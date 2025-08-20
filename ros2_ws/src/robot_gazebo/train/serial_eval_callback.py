# serial_eval_callback.py
import os, random
import numpy as np
import torch
from typing import Optional, List
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize 

class SerialEvalOnTrainEnv(BaseCallback):
    """
    在训练过程中串行评估：达到 eval_freq 之后，等待一个 episode 结束，
    然后暂停训练、用“训练的同一个 env”连续跑 n_eval_episodes 局评估。
    评估时：不写回放、不反向、不改变训练随机轨迹（保存/恢复 RNG）。
    """
    def __init__(
        self,
        eval_freq: int = 50_000,
        n_eval_episodes: int = 5,
        deterministic: bool = True,
        best_model_save_path: str | None = None,
        log_path: str | None = None,
        eval_cfg_indices: Optional[List[int]] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.eval_cfg_indices = eval_cfg_indices
        self.best_mean_reward = -np.inf
        self._next_eval = eval_freq

    def _on_training_start(self) -> None:
        # 准备保存目录
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(self.log_path, exist_ok=True)

    def _on_step(self) -> bool:
        # 没到评估步数 → 继续训练
        if self.eval_freq is None or self.num_timesteps < self._next_eval:
            return True

        # 只在 episode 结束时评估，避免割裂
        dones = self.locals.get("dones", None)
        if dones is None or not any(dones):
            return True

        # 触发一次评估
        self._run_eval()

        # 下一次评估门槛
        self._next_eval += self.eval_freq
        return True

    @torch.no_grad()
    def _run_eval(self):
        self.model.policy.set_training_mode(False)

        # 训练时的向量环境
        train_vec = self.training_env
        # 必须是单环境 DummyVecEnv
        assert train_vec.num_envs == 1, "串行评估回调目前只支持单环境 DummyVecEnv"

        # === 如果用了 VecNormalize，拿到它；否则为 None ===
        vn = train_vec if isinstance(train_vec, VecNormalize) else None
        # 找到底层“原始”env（不带任何 vec 包装）
        # base_env = (vn.venv.envs[0] if vn is not None else train_vec.envs[0])

        # 冻结归一化统计，且评估时不归一化 reward
        if vn is not None:
            vn.training = False
            vn.norm_reward = False

        # === 开启 JetAutoEnv 的评估模式，并传入固定评估集 ===
        # 用 VecEnv 的 env_method，能穿透所有 wrapper 调到最底层单个 env
        train_vec.env_method(
            "set_eval_mode",
            True,
            cfg_indices=self.eval_cfg_indices, # 关键字参数，直接传
            indices=[0],
        )

        # 保存 RNG，评估后恢复
        py_state  = random.getstate()
        np_state  = np.random.get_state()
        th_state  = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        ep_returns, ep_lengths = [], []

        for _ in range(self.n_eval_episodes):
            # VecEnv 风格：返回 (1, obs_dim) 的观测（若 vn 存在，则已按训练统计归一化）
            obs = train_vec.reset()
            ep_r, ep_len = 0.0, 0
            done = False

            while not done:
                # model.predict 支持批量 obs，这里 obs 是 (1, obs_dim)，直接喂入即可
                action, _ = self.model.predict(obs, deterministic=self.deterministic)

                # 同样使用 VecEnv 的 step：返回 (1,) 的 rewards/dones 和 list 的 infos
                obs, rewards, dones, infos = train_vec.step(action)

                # rewards 已是不归一化的原始奖励（因为 vn.norm_reward=False）
                ep_r  += float(rewards[0])
                ep_len += 1
                done = bool(dones[0])

            ep_returns.append(ep_r)
            ep_lengths.append(ep_len)

        mean_r = float(np.mean(ep_returns))
        std_r  = float(np.std(ep_returns))
        mean_len = float(np.mean(ep_lengths))

        self.logger.record("eval/mean_reward", mean_r)
        self.logger.record("eval/std_reward", std_r)
        self.logger.record("eval/mean_length", mean_len)
        self.logger.record("eval/episodes", self.n_eval_episodes)
        self.logger.dump(self.num_timesteps)

        # 保存最优模型
        if mean_r > self.best_mean_reward and self.best_model_save_path is not None:
            self.best_mean_reward = mean_r
            path = os.path.join(self.best_model_save_path, "best_model.zip")
            self.model.save(path)
            if self.verbose > 0:
                print(f"[SerialEval] New best mean reward {mean_r:.3f}. Saved to {path}")

        # 恢复 RNG
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(th_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

        # === 关闭评估模式，恢复训练模式 ===
        train_vec.env_method("set_eval_mode", False, indices=[0]) 

        if vn is not None:
            vn.training = True
            vn.norm_reward = True
           

        self.model.policy.set_training_mode(True)
