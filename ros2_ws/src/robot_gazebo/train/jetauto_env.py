# jetauto_env.py
import os
import json
import math
import random


import numpy as np
import gymnasium 
from gymnasium import spaces



import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from transforms3d import euler
from geometry_msgs.msg import Twist
import time

from shapely.geometry import Polygon
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import ignsetpose as poset

from igncontactserver import IgnContactsWatcher as IgnContactsWatcher

from collections import deque











pi = math.pi

MAX_STEPS = 1000  # 最大步数（episode 长度）
# 超参
R_MAX  = 100
R_MIN  =  5
THRESH = 0.9

C_HIT  = 50.0     # 首次碰撞惩罚

# 时间窗 & 冷却（秒）
CONTACT_WINDOW_SEC     = 0.20   # watcher.collided_recently() 的检测窗口
CONTACT_COOLDOWN_SEC   = 0.40   # 首次碰撞大罚的冷却时间

# 单次接触事件的小罚上限与速率
STICK_MAX_PER_CONTACT  = 100.0   # 单次接触事件最多扣的小罚总额
C_STICK_PER_SEC        = 100.0   # 每秒持续接触的小罚（时间基，推荐）
APPLY_STICK_ON_FIRST_HIT = False  # 首次那一帧是否叠加小罚（一般 False 更直观）

# MIN_RANGE_THRESH = 0.12      # 雷达代理阈值（米）
# ENTER_SPEED_EPS  = 0.03      # 认为在“顶墙前进”的最小指令速度（m/s）

STUCK_STEPS      = 100        # 连续接触达到此步数，进入“卡死”判定窗口
STUCK_DIST_EPS   = 0.01      # 最近若干步平均位移 < 1cm，视作卡死
STUCK_WIN        = 8         # 统计位移的窗口大小（步）
STUCK_PENALTY    = 100.0       # 可选：卡死时一次性额外惩罚；不用就设 0


def angle_wrap(a):
    return (a + pi) % (2*pi) - pi


def compute_ackermann(v: float, delta: float, wheel_base: float) -> Tuple[float, float]:
    """给定油门 v 和前轮转角 delta，计算线速度 & 角速度 ω。"""
    if abs(wheel_base) < 1e-6:
        raise ValueError("wheel_base must be non-zero")
    ω = v * math.tan(delta) / wheel_base if abs(delta) > 1e-6 else 0.0
    return v, ω


class JetAutoEnv(gymnasium.Env):
    """Gym 环境：JetAuto 在 Ignition Gazebo 中的搬运 + SAC 训练接口。"""

    metadata = {"render.modes": []}

    def __init__(self, config_path: str,
                 wheel_base: float = 0.213,
                 max_v: float = 1,
                 max_delta_deg: float = 22.0,
                 max_steps: int = MAX_STEPS,
                 node_name="jetauto_env_node"):
        super().__init__()

        # 1) 初始化 ROS 2 节点（假设 rclpy.init() 已在外部调用）
        self._node = Node(node_name)

        # 2) 加载所有 configuration
        with open(config_path) as f:
            raw = json.load(f)



        # 我们用一个小结构来追踪每个 config 的试验次数和成功次数
        self._configs       = raw
        self._cfg_trials    = [0] * len(raw)
        self._cfg_successes = [0] * len(raw)
        self._pool          = list(range(len(raw)))  # 活跃的 config 索引

        # 供 reset / step 调用
        self._current_cfg_idx = None

        # 每条 episode 的累计回报
        self._episode_reward = 0.0
        self._prev_dist      = None

        # 3) 发布 / 订阅
        self._cmd_pub   = self._node.create_publisher(Twist,     '/controller/cmd_vel', 10)
        self._odom_sub  = self._node.create_subscription(Odometry, '/odom',            self._odom_cb,    10)
        self._scan_sub  = self._node.create_subscription(LaserScan, '/scan',            self._scan_cb,    10)
        self._contact_topic = "/world/all_training/model/all_walls_and_cylinders/link/single_link/sensor/sensor_contact/contact"
   
        self.watcher = IgnContactsWatcher(self._contact_topic)
        self.watcher.start()

        # 内部状态
        self._odom     = None
        self._scan     = None
        # self._collided = False
        # self._offground = False

        # 等待第一条激光消息到达，以便确定 observation 大小
        while self._scan is None:
            rclpy.spin_once(self._node, timeout_sec=0.1)
        n_rays = len(self._scan.ranges)

        # 4) 定义 Gym 的 action_space & observation_space
        self.wheel_base   = wheel_base
        max_delta = math.radians(max_delta_deg)
        self.action_space = spaces.Box(
            low=np.array([-max_v, -max_delta], dtype=np.float32),
            high=np.array([+max_v, +max_delta], dtype=np.float32),
            dtype=np.float32
        )
        # 观测：n_rays 激光 + 碰撞标志 + 目标位置（x,y）相对坐标
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_rays + 2,), dtype=np.float32
        )

        self._max_steps = max_steps
        self._step_cnt = 0
        self._dist_history   = []
        self._last_window_avg = None
        self._ang_history = []
        self._last_ang_window_avg = None
        self._prev_iou = None

        self._target_poly = None

        self.car_length = 0.316  # or 从 config 里读
        self.car_width  = 0.259

        self.sparse = None

        self.target_yaw = None

        self._was_in_contact = False
        self._t_prev_step = None                # 上一 step 的时间戳（用于计算 dt）
        self._contact_steps = 0
        self._last_hit_time = -10**9
        self._contact_pen_budget = 0.0            # 当前接触事件剩余预算
        self._episode_contact_pen_total = 0.0     # 本回合累计（仅用于统计/日志）
        self._episode_stick_pen_total = 0.0
        self._episode_hit_pen_total = 0.0
        self._disp_hist = deque(maxlen=STUCK_WIN)
        self._pos_last = None
        # self._last_min_range = float("inf")  # 由 _get_obs() 每步更新

        # --- Eval mode state ---
        self._eval_mode: bool = False
        self._eval_cfgs: list[int] = []   # 固定评估用的 cfg 索引列表
        self._eval_ptr: int = 0           # 轮转指针

    def set_eval_mode(self, flag: bool, cfg_indices: Optional[list[int]] = None):
        """
        切换训练/评估模式。
        flag=True：进入评估模式；flag=False：恢复训练模式。
        cfg_indices：固定评估集（configs.json 中的索引列表）。传 None 则沿用之前的列表。
        """
        self._eval_mode = bool(flag)
        if cfg_indices is not None:
            # 过滤非法索引，避免越界
            n = len(self._configs)
            self._eval_cfgs = [i for i in cfg_indices if 0 <= i < n]
        # 每次切换/设置都重置指针，从列表开头开始评估
        self._eval_ptr = 0




    def _odom_cb(self, msg: Odometry):
        self._odom = msg.pose.pose

    def _scan_cb(self, msg: LaserScan):
        self._scan = msg

 

    # def _robot_off_ground(self,z_thresh = 0.01 , ang_thresh = 0.05):
    #     pose = self._odom
    #     z    = pose.position.z  
    #     # print('z: ',z)
    #     # 四元数转 roll/pitch/yaw
    #     qx, qy, qz, qw = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
    #     # roll/pitch 公式
    #     roll  = math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
    #     pitch = math.asin( 2*(qw*qy - qz*qx) )
    #     # print("roll = ",roll,"pitch = ",pitch)
    #     if z > z_thresh or abs(roll) > ang_thresh or abs(pitch)>ang_thresh:
    #         self._offground = True # 或者直接 done=True
    #     else:
    #         self._offground = False


    def _wait_pose_stable(self,
                        x, y, yaw,
                        pos_tol=0.02, yaw_tol=0.05,
                        consecutive=3, timeout=2.0) -> bool:
        """等待 odom 连续 consecutive 帧落在 (x,y,yaw) 容差内。"""
        t0 = time.time()
        ok = 0
        while time.time() - t0 < timeout:
            rclpy.spin_once(self._node, timeout_sec=0.02)
            if self._odom is None:
                continue
            dx = self._odom.position.x - x
            dy = self._odom.position.y - y
            pos_err = math.hypot(dx, dy)
            yaw_err = abs(angle_wrap(self._robot_yaw() - yaw))
            if pos_err <= pos_tol and yaw_err <= yaw_tol:
                ok += 1
                if ok >= consecutive:
                    return True
            else:
                ok = 0
        return False

    def _wait_fresh_scan(self, timeout=1.0) -> bool:
        t0 = time.time()
        hdr  = getattr(self._scan, "header", None) if self._scan else None
        last = getattr(hdr, "stamp", None)
        while time.time() - t0 < timeout:
            rclpy.spin_once(self._node, timeout_sec=0.02)
            if self._scan is None:
                continue
            hdr = getattr(self._scan, "header", None)
            if hdr is None:
                return True
            stamp = hdr.stamp
            if (last is None) or (stamp.sec != last.sec or stamp.nanosec != last.nanosec):
                return True
        return False
    
        # 传送完成后的自检：用目标点与当前 odom 的距离做 sanity check
    def _check_and_wait(self, sx, sy, syaw, q, tx, ty, *,
                                max_retry=2, dist_thresh=1.5) -> None:
        """
        若起点传送后到目标距离 > dist_thresh（m），则重试 set_pose 并等待 odom 刷新；
        连续 max_retry 次失败后抛错。
        """
        for attempt in range(max_retry + 1):
            # --- 计算当前到目标的距离
            dx0 = tx - self._odom.position.x
            dy0 = ty - self._odom.position.y
            dist_to_tgt = math.hypot(dx0, dy0)

            if dist_to_tgt <= dist_thresh:
                # 检查通过
                return

            # 太远：打印并重试 set_pose 一次
            print(f"[reset] dist_to_tgt={dist_to_tgt:.3f} m > {dist_thresh:.3f} m; "
                f"teleport retry {attempt+1}/{max_retry}...")

            # 重新 set_pose 到期望起点
            # ok = poset.ign_set_pose("jetauto", sx, sy, 0.0, q[1], q[2], q[3], q[0])
            # if not ok:
            #     print("[reset] warn: ign_set_pose returned False")

            # 等待位姿和激光稳定（如果你已有这两个工具函数，直接用）
            # 等待 odom 连续稳定在起点附近
            self._wait_pose_stable(sx, sy, syaw, pos_tol=0.02, yaw_tol=0.05,
                                consecutive=3, timeout=2.0)
            # 等待一帧新的 scan（可选，但有助于一致性）
            self._wait_fresh_scan(timeout=1.0)

            # 小步 spin，确保回调收进最新消息
            rclpy.spin_once(self._node, timeout_sec=0.1)

        # 若走到这里，说明重试仍不通过
        raise RuntimeError("reset(): teleport likely failed after retries")




        
    def _robot_yaw(self):
        qx = self._odom.orientation.x
        qy = self._odom.orientation.y
        qz = self._odom.orientation.z
        qw = self._odom.orientation.w

        # 标准的 yaw 提取公式：
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw


    def _robot_poly(self,yaw) -> Polygon:
        """
        返回当前 odom.pose 下，机器人底盘在 XY 平面上的多边形轮廓。
        """
        if self._odom is None:
            # 还没收到任何里程计
            return Polygon()

        # 1) 读取位置
        x = self._odom.position.x
        y = self._odom.position.y

        # 2) 读取四元数，计算 yaw
        # yaw = self._robot_yaw()

        # 3) 调用已有工具，算出四个角点
        corners = _robot_corners(
            x, y, yaw,
            self.car_length,
            self.car_width
        )

        # 4) 构造并返回 Polygon
        return Polygon(corners)
    
    def iou(self,yaw) -> float:
        robot_poly = self._robot_poly(yaw)
        target_poly = self._target_poly
        inter = robot_poly.intersection(target_poly).area
        union = robot_poly.union(target_poly).area
        return float( inter / union if inter > 0 else 0.0)



    def reset(self,
                *,               # 这样可以强制把 seed 当关键字参数
              seed: Optional[int] = None,
              options: Optional[dict] = None):
        # 1) 如果外部指定了种子，就用它来初始化 Python 随机
        if seed is not None:
            random.seed(seed)
        # 1) 从活跃池里随机选一个配置

        if not self._pool:
            self._pool = list(range(len(self._configs)))
            # self._cfg_trials = [0] * len(self._configs)
            # self._cfg_successes = [0] * len(self._configs)
            print("reset(): pool was empty → reinitialized")


        # === 选择 cfg 索引：训练随机 vs 评估固定 ===
        if self._eval_mode and len(self._eval_cfgs) > 0:
            cfg_idx = self._eval_cfgs[self._eval_ptr]
            # 轮转前进
            self._eval_ptr = (self._eval_ptr + 1) % len(self._eval_cfgs)
            self._current_cfg_idx = cfg_idx
            # 评估时**不**去累计 trials/成功率，免得影响训练用的采样池逻辑
        else:
            cfg_idx = random.choice(self._pool)
            self._current_cfg_idx = cfg_idx
            # 只在训练模式下统计 trials
            self._cfg_trials[cfg_idx] += 1

        cfg = self._configs[cfg_idx]

        # print("cfg_idx:", cfg_idx, "trials:", self._cfg_trials[cfg_idx])

        # 3) 清零 episode reward 和 prev_dist
        self._episode_reward = 0.0
        self._prev_dist      = None
        # self._collided       = False
        self._step_cnt       = 0
        self._dist_history   = []
        self._last_window_avg = None
        self._ang_history = []
        self._last_ang_window_avg = None
        self._prev_iou = None

        self._was_in_contact = False
        self._contact_steps = 0
        self._last_hit_time = -10**9
        self._contact_pen_budget = 0.0  
        self._episode_contact_pen_total = 0.0
        self._t_prev_step = None                # 上一 step 的时间戳（用于计算 dt）
        self._disp_hist.clear()
        self._pos_last = None
        self._episode_stick_pen_total = 0.0
        self._episode_hit_pen_total = 0.0
        # self._last_min_range = float("inf")
        

        # 4) teleport 到起点
        sx, sy, syaw = cfg['start_pose']
        q = euler.euler2quat(0, 0, syaw)
        poset.ign_set_pose("jetauto", sx, sy, 0.0,
                     q[1], q[2], q[3], q[0])
        
        tw = Twist(); tw.linear.x = tw.angular.z = 0.0
        self._cmd_pub.publish(tw)


        # 5) 等待就位：odom 连续稳定 + 激光有新帧（不用固定 sleep）
        ok_pose = self._wait_pose_stable(sx, sy, syaw, pos_tol=0.02, yaw_tol=0.05,
                                        consecutive=3, timeout=2.0)
        ok_scan = self._wait_fresh_scan(timeout=1.0)
        if not (ok_pose and ok_scan):
            # 允许重试 1~2 次；仍失败就报错，避免带病开局
            for _ in range(2):
                poset.ign_set_pose("jetauto", sx, sy, 0.0, q[1], q[2], q[3], q[0])
                ok_pose = self._wait_pose_stable(sx, sy, syaw, timeout=2.0)
                ok_scan = self._wait_fresh_scan(timeout=1.0)
                if ok_pose and ok_scan:
                    break
            if not (ok_pose and ok_scan):
                raise RuntimeError("reset(): failed to stabilize pose/scan after teleport")

        # 5) 记录目标
        tx, ty = cfg['target_position']
        self._target = (tx, ty)
        self._target_poly = Polygon(cfg['target_poly'])
        self.target_yaw = cfg['target_yaw']

        # 如果前面的等待不充分，下面的检查会触发重试
        self._check_and_wait(sx, sy, syaw, q, tx, ty,
                                    max_retry=3, dist_thresh=2.0)
            

        # print(tx," ",ty)

        # 刷新一次传感器数据
        rclpy.spin_once(self._node, timeout_sec=0.5)
        # if self.iou(self.target_yaw) == 0.0 : # 计算初始 IOU


        obs = self._get_obs()
        # print('reset done')
        return obs, {}

    

    def _get_obs(self) -> np.ndarray:
        # raw ranges may contain inf/nan
        raw = np.array(self._scan.ranges, dtype=np.float32)

        # replace inf with max_range, nan with max_range (or some large finite value)
        # You can read range_max from the LaserScan message if you want.
        max_r = getattr(self._scan, 'range_max', 1.5)
        ranges = np.nan_to_num(raw,
                               nan=max_r,
                               posinf=max_r,
                               neginf=0.0)
        # print('start to check col')
        
        # print('checked is',self._collided)
        # col = np.array([1.0 if self._collided else 0.0], dtype=np.float32)
        dx = self._target[0] - self._odom.position.x
        dy = self._target[1] - self._odom.position.y
        tgt = np.array([dx, dy], dtype=np.float32)
        obs = np.concatenate([ranges,  tgt])
        # obs = ranges
        

        # sanity check for debugging:
        if not np.isfinite(obs).all():
            raise ValueError(f"Non-finite observation: {obs}")

        return obs.astype(np.float32, copy=False)


    def step(self, action):

        obs, reward,  info = None, 0.0,  {}    
        terminated = False
        truncated  = False


        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        v, delta = float(action[0]), float(action[1])
        # 转换成 (linear, angular)
        lin, ang = compute_ackermann(v, delta, self.wheel_base)

        # print(lin,ang)

        # 发布速度
        twist = Twist()
        twist.linear.x  = float(lin)
        twist.angular.z = float(ang)
        self._cmd_pub.publish(twist)

        # 等待一次仿真
        rclpy.spin_once(self._node, timeout_sec=0.01)
        self._step_cnt += 1


        obs = self._get_obs()

        idx = self._current_cfg_idx
        trials = self._cfg_trials[idx]
        succ = self._cfg_successes[idx]
        rate = succ / trials if trials > 0 else 0.0
        #done = False
        #reward = 0.0
        # self._collided = self.watcher.collided_recently(0.2)#ign_check_collision(self._contact_topic)

        # hit_flag = bool(self.watcher.collided_recently(0.2))
        # proxy_flag = (getattr(self, "_last_min_range", np.inf) < MIN_RANGE_THRESH) and (lin > ENTER_SPEED_EPS)
        t_now = time.monotonic()
        # 计算 dt（用于按秒积分的小罚）；第一步没有上一时间，dt 取 0
        dt = 0.0 if self._t_prev_step is None else max(0.0, t_now - self._t_prev_step)
        self._t_prev_step = t_now

        # 时间窗检测：过去 CONTACT_WINDOW_SEC 内是否发生过接触
        in_contact = self.watcher.collided_recently(CONTACT_WINDOW_SEC)

        # 首次接触（上升沿 + 时间冷却）
        first_hit = (
            in_contact
            and (not self._was_in_contact)
            and ((t_now - self._last_hit_time) > CONTACT_COOLDOWN_SEC)
        )

        # —— 大罚：首次接触（不占用预算）——
        if first_hit:
            reward -= C_HIT
            self._last_hit_time = t_now
            # 装满“本次接触事件”的小罚预算
            self._contact_pen_budget = STICK_MAX_PER_CONTACT
            self._episode_hit_pen_total += C_HIT
            

        # —— 持续接触：按时间积分的小罚（受本次事件预算限制）——
        if in_contact:
            # 是否在首次这一帧就给小罚
            give_stick_now = (not first_hit) or APPLY_STICK_ON_FIRST_HIT
            if give_stick_now and dt > 0.0 and self._contact_pen_budget > 1e-9:
                apply = min(C_STICK_PER_SEC * dt, self._contact_pen_budget)
                reward -= apply
                self._contact_pen_budget -= apply
                self._episode_contact_pen_total += apply
                self._episode_stick_pen_total += apply   # 统计持续接触惩罚

            self._contact_steps += 1
        else:
            # 离墙：结束当前接触事件，清空预算 & 连续计数
            self._contact_pen_budget = 0.0
            self._contact_steps = 0

        # 记录上一时刻接触状态
        self._was_in_contact = in_contact


        # 5) 位移统计（用于卡死判定）
        if self._pos_last is not None:
            disp = math.hypot(self._odom.position.x - self._pos_last[0],
                            self._odom.position.y - self._pos_last[1])
            self._disp_hist.append(disp)
        self._pos_last = (self._odom.position.x, self._odom.position.y)

        # 6) 卡死判定（不强制终止，就设 truncated；你也可以只加惩罚）
        stuck = (
            in_contact and
            self._contact_steps >= STUCK_STEPS and
            (len(self._disp_hist) >= STUCK_WIN) and
            (sum(self._disp_hist) / len(self._disp_hist) < STUCK_DIST_EPS)
        )
        if stuck:
            reward -= STUCK_PENALTY     # 可选
            truncated = True            # 成为“外因截断”，而非 terminated
            self._episode_reward += reward
            info.update({
                "config_idx": idx,
                "episode_reward": self._episode_reward,
                "config_trials": trials,
                "config_successes": succ,
                "config_rate": rate,
                "raw_step_reward": float(reward),


                # 其他诊断信息
                "in_contact": int(in_contact),
                "contact_steps": int(self._contact_steps),
                "contact_pen_ep_total": float(self._episode_hit_pen_total + self._episode_stick_pen_total),
                "hit_pen_ep_total": float(self._episode_hit_pen_total),
                "stick_pen_ep_total": float(self._episode_stick_pen_total),
                "contact_pen_budget": float(self._contact_pen_budget),
                "stuck": 1,
            })
            print("episode truncated: stuck in contact for too long")
            return obs, float(reward), terminated, truncated, info




        yaw = self._robot_yaw()


        yaw_dif = abs(angle_wrap(self.target_yaw - yaw))


        self._ang_history.append(yaw_dif)

        bonus_ang = 0.0
        window_len_ang = 10
        if self._step_cnt % (window_len_ang) == 0 and len(self._ang_history) >= window_len_ang:
            window_curr = self._ang_history[-4:]
            if len(window_curr) >= 1:
                avg_curr = sum(window_curr) / len(window_curr)
            else:
                avg_curr = 0.0

            if self._last_ang_window_avg == None:
                window_prev = self._ang_history[-window_len_ang:-window_len_ang+4]
                avg_prev =  sum(window_prev) /4.0

                avg_delta = avg_prev - avg_curr
            else:
                avg_delta = self._last_ang_window_avg - avg_curr

            K_ang = 300.0
            bonus_ang = K_ang*avg_delta

            reward += bonus_ang

            self._last_ang_window_avg = avg_curr

        dx   = self._target[0] - self._odom.position.x
        dy   = self._target[1] - self._odom.position.y
        dist = math.hypot(dx, dy)

        self._dist_history.append(dist)


        bonus_dis = 0.0
        window_len_dis = 10
        if self._step_cnt % (window_len_dis) == 0 and len(self._dist_history) >= window_len_dis:
            window_curr = self._dist_history[-4:]
            if len(window_curr) >= 1:
                avg_curr = sum(window_curr) / len(window_curr)
            else:
                avg_curr = 0.0

            if self._last_window_avg == None:
                window_prev = self._dist_history[-window_len_dis:-window_len_dis+4]
                avg_prev =  sum(window_prev) /4.0

                avg_delta = avg_prev - avg_curr
            else:
                avg_delta = self._last_window_avg - avg_curr

            K = 300.0
            bonus_dis = K*avg_delta

            reward += bonus_dis

            self._last_window_avg = avg_curr

        delta_r_iou = 0
        # r_iou = 0
        r_dist = 0
        r_time = 0


        # robot_poly = self._robot_poly(yaw)
        # target_poly = self._target_poly
        # inter = robot_poly.intersection(target_poly).area
        # union = robot_poly.union(target_poly).area
        # iou = inter / union if union > 0 else 0.0
        # w_iou = 20.0   # 你可以调这个权重
        # r_iou = w_iou * iou
        iou = self.iou(yaw)

        if self._prev_iou is None:

            delta_r_iou = 0.0
        else:
            delta_w_iou_p = 160.0   # 差分权重
            delta_w_iou_n = 80.0   # 差分权重
            diff_iou = iou - self._prev_iou
            if diff_iou > 0:
                
                delta_r_iou = delta_w_iou_p * (diff_iou)
            else:
                delta_r_iou = delta_w_iou_n * (diff_iou)

        self._prev_iou = iou

                # —— 2. 差分距离 Reward —— 
                # 上一步到目标的距离保存在 self._prev_dist
        if self._prev_dist is None:
                    # 第一步差分距离用 0
            r_dist = 0.0
        else:
            w_dist_p = 150.0  
            w_dist_n = 75.0 # 距离差分的权重
            dis_dif = self._prev_dist - dist
            if dis_dif >0:
                r_dist = w_dist_p * (dis_dif)
            else: r_dist = w_dist_n * dis_dif
                # 更新 prev_dist
        self._prev_dist = dist

                # —— 3. 时间惩罚 —— 
                # 随 step_cnt 增加，由 tanh 有界地增加惩罚
        alpha = 1.0    # 最大惩罚幅度
        beta  = 0.013   # 增长速率
        r_time = - alpha * math.tanh(beta * self._step_cnt)

                # 总 reward
        reward += (delta_r_iou + r_dist + r_time)        

            # 目标到达
        if dist < 0.35:
            reward += +100.0

            terminated = True
            truncated  = False

            self._cfg_successes[idx] += 1
            succ = self._cfg_successes[idx]
            rate = succ / trials if trials > 0 else 0.0


            # 超时
        elif self._step_cnt > self._max_steps:
            reward += -20.0
            terminated = False
            truncated  = True
            print("episode truncated: max steps reached")




        
        

        if (terminated or truncated) and ((trials > R_MAX) or (trials >= R_MIN and rate >= THRESH)):
            if idx in self._pool:
                self._pool.remove(idx)





        # print("\nstep:", self._step_cnt)
        #           "\noff_ground",self._offground,
        #       "\ncollided:", self._collided,
        #       "\ntarget_dist:", dist,
        #       "\nbonus_dis", bonus_dis,
        #       "\nbonus_ang",bonus_ang,
        #       "\ndelta_r_iou",delta_r_iou,
        #       "\nr_iou:", r_iou,
        #       "\nr_dist:", r_dist,
        #       "\nr_time",r_time,
        #       "\n----------",

        #       "\nreward",reward,
        #       "\nepisode_reward",self._episode_reward)



        self._episode_reward += reward




        info.update({
            "config_idx": idx,
            "episode_reward": self._episode_reward,
            "config_trials": trials,
            "config_successes": succ,
            "config_rate": rate,
            "raw_step_reward": float(reward),
            # 你的窗口奖励诊断项…
            'window_bonus_dis': bonus_dis,
            'window_bonus_ang': bonus_ang,
            # 其他诊断信息
            "in_contact": int(in_contact),
            "contact_steps": int(self._contact_steps),
            "contact_pen_ep_total": float(self._episode_hit_pen_total + self._episode_stick_pen_total),
            "hit_pen_ep_total": float(self._episode_hit_pen_total),
            "stick_pen_ep_total": float(self._episode_stick_pen_total),
            "contact_pen_budget": float(self._contact_pen_budget),
            "stuck": 0,
        })

        # return self._get_obs(), reward, done, info
        # Gymnasium expects: obs, reward, terminated, truncated, info
        
        return obs, float(reward), terminated, truncated, info
# --------------------------------------------------------





    def render(self, mode='human'):
        pass

    def close(self):
        try:
            self.watcher.stop()   # 你的 watcher 需要提供 stop

        except Exception:
            pass
        try:
            self._node.destroy_node()
        except:
            pass

def _robot_corners(x: float, y: float, yaw: float, L: float, W: float) -> List[Tuple[float, float]]:
    hl, hw = L/2.0, W/2.0
    corners_local = [( hl,  hw), ( hl, -hw), (-hl, -hw), (-hl,  hw)]
    corners = []
    c = math.cos(yaw)
    s = math.sin(yaw)
    for lx, ly in corners_local:
        gx = x + lx * c - ly * s
        gy = y + lx * s + ly * c
        corners.append((gx, gy))
    return corners