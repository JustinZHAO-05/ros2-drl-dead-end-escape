# train_sac.py
import rclpy
import os
import argparse

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.logger import configure

from jetauto_env import JetAutoEnv  # 你的环境定义文件

from stable_baselines3.common.callbacks import BaseCallback

from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.monitor import Monitor







class MyInfoLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # 单 env 时，locals["rewards"][0] 就是本 step 的 reward
        step_rew = float(self.locals["rewards"][0])
        self.logger.record("step/reward", step_rew)



        # 如果 done，就把 info['episode']['r'] 写出来
        for info in self.locals["infos"]:

            window_bonus_ang = info.get("window_bonus_ang")
            window_bonus_dis = info.get("window_bonus_dis")

            if window_bonus_ang is not None and window_bonus_dis:

                if window_bonus_ang > 0.0:
            
                    self.logger.record("step/window_bonus_ang",float(window_bonus_ang))

                
            
                    self.logger.record("step/window_bonus_dis",float(window_bonus_dis))

            ep = info.get("episode")
            if ep is not None:
                self.logger.record("episode/total_reward", float(ep["r"]))
                # 也可以记录 config idx、rate 等
                cfg = info.get("config_idx")
                if cfg is not None:
                    self.logger.record("episode/config_idx", float(cfg))
                rate = info.get("config_rate")
                if rate is not None:
                    self.logger.record("episode/config_success_rate",float(rate))
        current_step = self.model.num_timesteps
        self.logger.dump(current_step)        
        return True


# class TBRewardCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super().__init__(verbose)

#     def _on_step(self) -> bool:
#         # self.locals 中可以拿到 VecEnv 的 'rewards', 'infos', 'dones' 等
#         rewards = self.locals["rewards"]      # numpy array, shape=(n_envs,)
#         dones   = self.locals["dones"]        # boolean array
#         infos   = self.locals["infos"]        # list of dict

#         # 1) 每一步 reward
#         # 如果只有一个 env，就写 rewards[0]
#         self.logger.record("step/reward", float(rewards[0]))

#         # 2) 每个 env 的 episode 结束时，写 episode 回报
#         for idx, done in enumerate(dones):
#             if done:
#                 # info['episode']['r'] 是 RecordEpisodeStatistics 自动填的
#                 ep_r = infos[idx].get("episode", {}).get("r")
#                 if ep_r is not None:
#                     self.logger.record("episode/reward", float(ep_r))
#         current_step = self.model.num_timesteps
#         self.logger.dump(current_step)
#         return True

class TBRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # 1) 每一步都记录 step reward
        step_rew = float(self.locals["rewards"][0])
        self.logger.record("step/reward", step_rew, exclude=["stdout"])

        # 2) 如果这个 step 结束了一个 episode，就记录 episode reward
        done = self.locals["dones"][0]
        if done:
            info = self.locals["infos"][0]
            ep_r = info.get("episode", {}).get("r")
            if ep_r is not None:
                self.logger.record("episode/total_reward", float(ep_r))

        # 3) 强制把这次记录立刻 dump 到 TensorBoard
        #    这里用 model.num_timesteps 来替代 self.num_timesteps
        current_step = self.model.num_timesteps
        self.logger.dump(current_step)

        return True



class EpisodeAlignedEval(EvalCallback):
    def _on_step(self) -> bool:
        # self.locals["dones"] 是一个布尔列表，对应每个 env
        if not any(self.locals["dones"]):
            # 只有当至少有一个 env 正好 done 时，才继续执行父类逻辑
            return True
        return super()._on_step()



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
    def make_env():
        env = JetAutoEnv(config_path=args.configs)
        env = RecordEpisodeStatistics(env)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])
    env = VecMonitor(env)

    # 2) 配置 logger，将 TensorBoard 日志存到 save_dir/log
    new_logger = configure(args.save_dir, ["stdout", "tensorboard"])
    if args.checkpoint is not None:
        model = SAC.load(
            args.checkpoint,
            env=env,
            device="auto",
            tensorboard_log=os.path.join(args.save_dir, "tensorboard")
        )
    else:

    # 3) 创建 SAC 模型
        model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join(args.save_dir, "tensorboard"),
        device="auto",
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
    )
        model.set_logger(new_logger)

    # 4) 回调：定期存 checkpoint；定期做 Eval
    #   EvalCallback 会在 eval_env 上评估并把最佳模型复制到 best_model.zip
    eval_env = DummyVecEnv([make_env])
    eval_env = VecMonitor(eval_env)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.save_dir, "best_model"),
        log_path=os.path.join(args.save_dir, "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=10,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=800,
        save_path=args.save_dir,
        name_prefix="checkpoint",
        save_replay_buffer= True,
        save_vecnormalize = True,
    )

    # tb_callback = TBRewardCallback()

    # hist_cb = WeightsHistCallback()
    # 5) 开始训练
    model.learn(
        total_timesteps=args.total_timesteps,
        reset_num_timesteps=False,
        callback=[checkpoint_callback, eval_callback,MyInfoLogger()],
        log_interval=1,
        tb_log_name="SAC_JetAuto"
    )

    # 6) 最后保存
    model.save(os.path.join(args.save_dir, "final_model"))
    print("Training completed. Models saved to", args.save_dir)

if __name__ == "__main__":
    main()
