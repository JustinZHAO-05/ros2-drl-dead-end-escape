# train_sac.py
import rclpy
import os,re
import argparse

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.logger import configure

from jetauto_env import JetAutoEnv  # 你的环境定义文件

from stable_baselines3.common.callbacks import BaseCallback

from serial_eval_callback import SerialEvalOnTrainEnv

# from gymnasium.wrappers import RecordEpisodeStatistics
# from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.utils import set_random_seed
set_random_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False






class MyInfoLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._norm_ep_return = 0.0   # 用来累计“归一化口径”的 episode 回报

    def _on_step(self) -> bool:
        # 单 env 时，locals["rewards"][0] 就是本 step 的 reward
        # 1) 归一化的 step 奖励（来自 VecNormalize 后的 reward）
        norm_step_rew = float(self.locals["rewards"][0])
        # self.logger.record("step/reward_norm", norm_step_rew)
        self._norm_ep_return += norm_step_rew

        # # 2) 原始的 step 奖励（来自 env.step() 塞进 info 的字段）
        # raw_step_rew = self.locals["infos"][0].get("raw_step_reward")
        # if raw_step_rew is not None:
        #     self.logger.record("step/reward_raw", float(raw_step_rew))

        # 3) 你之前的诊断项（照旧）
        info0 = self.locals["infos"][0]
        # window_bonus_ang = info0.get("window_bonus_ang")
        # window_bonus_dis = info0.get("window_bonus_dis")
        # if (window_bonus_ang is not None) and (window_bonus_dis is not None):
        #     self.logger.record("step/window_bonus_ang", float(window_bonus_ang))
        #     self.logger.record("step/window_bonus_dis", float(window_bonus_dis))



        # 4) episode 结束时，记录：
        #    - 原始 episode 回报：info['episode']['r']（VecMonitor提供）
        #    - 归一化 episode 回报：我们刚才累计的 _norm_ep_return
        done = bool(self.locals["dones"][0])
        if done:
            ep = info0.get("episode")
            if ep is not None:
                # 原始口径（已有）
                self.logger.record("episode/total_reward_raw", float(ep["r"]))
            # 归一化口径（新增）
            self.logger.record("episode/total_reward_norm", float(self._norm_ep_return))
            # 你自己的统计（照旧）

            rate = info0.get("config_rate")
            if rate is not None:
                self.logger.record("episode/config_success_rate", float(rate))

            contact_pen_ep_total = info0.get("contact_pen_ep_total")
            if contact_pen_ep_total is not None:
                self.logger.record("episode/contact_pen_ep_total", float(contact_pen_ep_total))

            hit_pen_ep_total = info0.get("hit_pen_ep_total")
            if hit_pen_ep_total is not None:
                self.logger.record("episode/hit_pen_ep_total", float(hit_pen_ep_total))

            stick_pen_ep_total = info0.get("stick_pen_ep_total")
            if stick_pen_ep_total is not None:
                self.logger.record("episode/stick_pen_ep_total", float(stick_pen_ep_total))

            stuck = info0.get("stuck")
            if stuck is not None:
                self.logger.record("episode/stuck", int(stuck))


            # 重置累计器
            self._norm_ep_return = 0.0

        # 5) dump 频率（可调）
        if done or (self.model.num_timesteps % 200 == 0):
            self.logger.dump(self.model.num_timesteps)

        return True



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--configs", 
        help="Path to configs.json",
        default="configs.json"
    )
    p.add_argument(
        "--total-timesteps", 
        type=int, 
        default=3_000_000,
        help="Total training timesteps"
    )
    p.add_argument(
        "--eval-freq",
        type=int,
        default=100_000,
        help="Evaluate every N timesteps"
    )
    p.add_argument(
        "--save-dir",
        default="./models/",
        help="Directory to save models and logs"
    )
    p.add_argument(
    "--checkpoint", 
    help="Path to an existing checkpoint .zip to resume from",
    default=None,
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    rclpy.init()

    # 1) 创建环境，并用 VecMonitor 跟踪 episode 信息
    def make_env(name):
        def _thunk():
            # JetAutoEnv 里新增一个 node_name 参数（很容易改）
            return JetAutoEnv(config_path=args.configs, node_name=f"jetauto_env_node_{name}")
        return _thunk

    # 训练环境
    # train_env = DummyVecEnv([make_env("train")])
    # train_env = VecMonitor(train_env)
    # train_env = VecNormalize(
    #     train_env,
    #     norm_obs=True,
    #     norm_reward=True,   # 训练时常开
    #     clip_obs=10.0
    # )



    # 2) 配置 logger，将 TensorBoard 日志存到 save_dir/log
    # 统一到 save_dir/tb
    # log_dir = os.path.join(args.save_dir, "tb")
    # new_logger = configure(log_dir, ["stdout", "tensorboard"])
    # new_logger = configure(args.save_dir, ["stdout", "tensorboard"])
    if args.checkpoint is not None:
        # model = SAC.load(
        #     args.checkpoint,
        #     env=train_env,
        #     device="auto",
        #     tensorboard_log=os.path.join(args.save_dir, "tb")
        # )
        ckpt_zip = args.checkpoint  # e.g. ./models/exp1/checkpoint_100000_steps.zip
        ckpt_dir = os.path.dirname(ckpt_zip)
        zip_stem = os.path.splitext(os.path.basename(ckpt_zip))[0]  # "checkpoint_100000_steps"

        # 从 zip 名字里提取 step 和前缀，例如 ("checkpoint", "100000")
        m = re.match(r"^(?P<prefix>.+)_(?P<step>\d+)_steps$", zip_stem)
        if not m:
            raise ValueError(f"Unexpected checkpoint name: {zip_stem}")
        prefix = m.group("prefix")                  # "checkpoint"
        step   = m.group("step")                    # "100000"

        # 根据你的命名规则拼出另外两个路径
        vecnorm_path = os.path.join(ckpt_dir, f"{prefix}_vecnormalize_{step}_steps.pkl")
        replay_path  = os.path.join(ckpt_dir, f"{prefix}_replay_buffer_{step}_steps.pkl")

        print(f"[Resume] ckpt:     {ckpt_zip}")
        print(f"[Resume] vecnorm:  {vecnorm_path} ({'OK' if os.path.exists(vecnorm_path) else 'MISSING'})")
        print(f"[Resume] replay:   {replay_path}  ({'OK' if os.path.exists(replay_path)  else 'MISSING'})")

        # 1) 先构建“基础向量环境”（不要一上来就新建 VecNormalize，否则统计会被覆盖）
        base_env = DummyVecEnv([make_env("train")])
        base_env = VecMonitor(base_env)

        # 2) 如果有保存过 VecNormalize 统计，就加载回去；否则新建一个
        if os.path.exists(vecnorm_path):
            print(f"[Resume] Loading VecNormalize stats from: {vecnorm_path}")
            train_env = VecNormalize.load(vecnorm_path, base_env)
        else:
            print("[Resume] VecNormalize stats not found, creating new statistics.")
            train_env = VecNormalize(base_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        # 训练阶段务必打开训练标志 & 奖励归一化
        train_env.training = True
        train_env.norm_reward = True

        # 3) 加载 SAC 模型（注意：这里传入的是已经含有统计的 train_env）
        model = SAC.load(
            ckpt_zip,
            env=train_env,
            device="auto",
            tensorboard_log=os.path.join(args.save_dir, "tb"),
        )

        # 4) 如有 Replay Buffer，则加载（可选，但推荐无缝续训时加载）
        if os.path.exists(replay_path):
            print(f"[Resume] Loading replay buffer from: {replay_path}")
            # truncate_last_trajectory=False 一般更稳妥；如上次中断在 episode 中间可改 True
            model.load_replay_buffer(replay_path, truncate_last_traj=False)
        else:
            print("[Resume] Replay buffer not found, starting with empty buffer.")

    else:

    # 3) 创建 SAC 模型
    #     model = SAC(
    #     "MlpPolicy",
    #     train_env,
    #     verbose=1,
    #     tensorboard_log=os.path.join(args.save_dir, "tb"),
    #     device="auto",
    #     learning_rate=3e-4,
    #     buffer_size=1_000_000,
    #     batch_size=256,
    #     gamma=0.99,
    #     tau=0.005,
    # )
    # ===== 新训练：创建 VecNormalize + SAC =====
        train_env = DummyVecEnv([make_env("train")])
        train_env = VecMonitor(train_env)
        train_env = VecNormalize(
            train_env, norm_obs=True, norm_reward=True, clip_obs=10.0
        )

        model = SAC(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log=os.path.join(args.save_dir, "tb"),
            device="auto",
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
        )



        # model.set_logger(new_logger)


    serial_eval_cb = SerialEvalOnTrainEnv(
        eval_freq=args.eval_freq,             # 例如 50_000
        n_eval_episodes=5,
        deterministic=True,
        best_model_save_path=os.path.join(args.save_dir, "best_model"),
        log_path=os.path.join(args.save_dir, "eval_logs"),
        eval_cfg_indices=[0, 50, 25, 75, 49, 99],   # ← 固定评估关卡索引
        verbose=1,
    )



    checkpoint_callback = CheckpointCallback(
        save_freq= 100_000,  # 每 100_000 步保存一次
        save_path=args.save_dir,
        name_prefix="checkpoint",
        save_replay_buffer= True,
        save_vecnormalize = True,
    )



    # 5) 开始训练
    model.learn(
        total_timesteps=args.total_timesteps,
        reset_num_timesteps=False,
        callback=[checkpoint_callback, serial_eval_cb, MyInfoLogger()],
        log_interval=1,
        tb_log_name="SAC_JetAuto"
    )

    # 6) 最后保存
    model.save(os.path.join(args.save_dir, "final_model"))
    print("Training completed. Models saved to", args.save_dir)
    train_env.save(os.path.join(args.save_dir, "vecnormalize.pkl"))
    train_env.close()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
