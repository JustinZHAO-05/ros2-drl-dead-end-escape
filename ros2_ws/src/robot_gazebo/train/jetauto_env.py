# jetauto_env.py
import os
import json
import math
import random
import subprocess
import shlex
import re
import numpy as np
import gymnasium 
from gymnasium import spaces


import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from transforms3d import euler
from geometry_msgs.msg import Twist, Pose

from shapely.geometry import Polygon
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional





import subprocess
import shlex
import re






pi = math.pi


# 超参
R_MAX  = 20
R_MIN  =  5
THRESH = 0.9




def ign_check_collision(topic: str, timeout: float = 0.5) -> bool:
    """
    调用 `ign topic` 抓一条 contact 消息（JSON 格式），
    如果 collision1/collision2 字段存在，则认为发生了碰撞。
    """
    cmd = [
        "ign", "topic",
        "-e",
        "-n", "1",                  # 只抓一条就退出
        "-t", topic,
        "-m","ignition.msgs.Contacts",
        "--json-output"
    ]
    try:
        # 捕获 stdout，忽略 stderr
        res = subprocess.run(cmd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL,
                             text=True,
                             timeout=timeout)
        if res.returncode != 0 or not res.stdout:
            return False
        msg = json.loads(res.stdout)
        # print(msg)
        # Ignition Contacts 消息里，collision1.name / collision2.name 存在时，说明有接触
        # contact = msg.get("contact", {})
        # name1 = contact.get("collision1", {}).get("name", "")
        # name2 = contact.get("collision2", {}).get("name", "")
        # return bool(name1 and name2)
        if (msg ):
            return True
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return False




import time
import subprocess, shlex, re

def ign_set_pose(entity_name: str,
                 x: float, y: float, z: float,
                 qx: float, qy: float, qz: str, qw: float,
                 world: str = "/world/all_training",
                 timeout_ms: int = 2000,
                 retries: int = 3,
                 retry_delay: float = 0.1) -> bool:
    """
    Calls `ign service -s {world}/set_pose` to teleport `entity_name` and returns True on success.
    If the service returns data: false, retry up to `retries` times (with delay).
    Raises RuntimeError on subprocess failure, ValueError on parse failure.
    """
    # 构造请求体
    req = f"""
name: "{entity_name}"
position {{
  x: {x}
  y: {y}
  z: {z}
}}
orientation {{
  x: {qx}
  y: {qy}
  z: {qz}
  w: {qw}
}}
""".strip()

    cmd = [
        "ign", "service", "-s", f"{world}/set_pose",
        "--reqtype", "ignition.msgs.Pose",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", str(timeout_ms),
        "--req", req
    ]

    print(cmd)

    for attempt in range(1, retries+1):
        result = subprocess.run(cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
        if result.returncode != 0:
            # 进程异常退出，直接报错
            raise RuntimeError(
                f"`{' '.join(shlex.quote(c) for c in cmd)}` failed:\n"
                f"{result.stderr}"
            )

        # 解析返回的 data: true/false
        m = re.search(r"data:\s*(true|false)", result.stdout)
        if not m:
            # raise ValueError(f"无法解析 ign 返回值：\n{result.stdout!r}")
            print(f"无法解析 ign 返回值：\n{result.stdout!r}")   
            time.sleep(retry_delay)         
            continue
        ok = (m.group(1) == "true")
        if ok:
            return True

        # 返回 false，准备重试
        if attempt < retries:
            time.sleep(retry_delay)

    # 连续 retries 次都失败
    return False




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
                 max_v: float = 0.6,
                 max_delta_deg: float = 22.0,
                 max_steps: int = 500):
        super().__init__()

        # 1) 初始化 ROS 2 节点（假设 rclpy.init() 已在外部调用）
        self._node = Node("jetauto_env_node")

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
   

        # 内部状态
        self._odom     = None
        self._scan     = None
        self._collided = False
        self._offground = False

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
            shape=(n_rays + 3,), dtype=np.float32
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

    def _odom_cb(self, msg: Odometry):
        self._odom = msg.pose.pose

    def _scan_cb(self, msg: LaserScan):
        self._scan = msg

    # def _touch_cb(self, msg: Bool):
        
    #     if msg.data:
    #         self._collided = True

        # else:
        #     self._collided = False  

    def _robot_off_ground(self,z_thresh = 0.01 , ang_thresh = 0.05):
        pose = self._odom
        z    = pose.position.z  
        # print('z: ',z)
        # 四元数转 roll/pitch/yaw
        qx, qy, qz, qw = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
        # roll/pitch 公式
        roll  = math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
        pitch = math.asin( 2*(qw*qy - qz*qx) )
        # print("roll = ",roll,"pitch = ",pitch)
        if z > z_thresh or abs(roll) > ang_thresh or abs(pitch)>ang_thresh:
            self._offground = True # 或者直接 done=True
        else:
            self._offground = False
        
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


    def _robot_poly(self) -> Polygon:
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
        yaw = self._robot_yaw()

        # 3) 调用已有工具，算出四个角点
        corners = _robot_corners(
            x, y, yaw,
            self.car_length,
            self.car_width
        )

        # 4) 构造并返回 Polygon
        return Polygon(corners)



    def reset(self,
                *,               # 这样可以强制把 seed 当关键字参数
              seed: Optional[int] = None,
              options: Optional[dict] = None):
        # 1) 如果外部指定了种子，就用它来初始化 Python 随机
        if seed is not None:
            random.seed(seed)
        # 1) 从活跃池里随机选一个配置
        cfg_idx = random.choice(self._pool)
        self._current_cfg_idx = cfg_idx

        cfg = self._configs[cfg_idx]

        # 2) 统计它的 trials
        self._cfg_trials[cfg_idx] += 1

        print("cfg_idx:", cfg_idx, "trials:", self._cfg_trials[cfg_idx])

        # 3) 清零 episode reward 和 prev_dist
        self._episode_reward = 0.0
        self._prev_dist      = None
        self._collided       = False
        self._step_cnt       = 0
        self._dist_history   = []
        self._last_window_avg = None
        self._ang_history = []
        self._last_ang_window_avg = None
        self._prev_iou = None
        

        # 4) teleport 到起点
        sx, sy, syaw = cfg['start_pose']
        q = euler.euler2quat(0, 0, syaw)
        ign_set_pose("jetauto", sx, sy, 0.0,
                     q[1], q[2], q[3], q[0])
        
        
        
                # wait for a valid odom message
        start = time.time()
        while self._odom is None:
            rclpy.spin_once(self._node, timeout_sec=0.1)
            if time.time() - start > 2.0:
                raise RuntimeError("Timeout waiting for odom in reset()")
            
        time.sleep(2)   
        rclpy.spin_once(self._node, timeout_sec=0.5) 
        # print(self._odom.position.x," ",self._odom.position.y)

        # 5) 记录目标
        tx, ty = cfg['target_position']
        self._target = (tx, ty)
        self._target_poly = Polygon(cfg['target_poly'])
        self.target_yaw = cfg['target_yaw'] - pi

        # print(tx," ",ty)

        # 刷新一次传感器数据
        rclpy.spin_once(self._node, timeout_sec=0.5)
        obs = self._get_obs()
        print('reset done')
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
        self._collided = ign_check_collision(self._contact_topic)
        # print('checked is',self._collided)
        col = np.array([1.0 if self._collided else 0.0], dtype=np.float32)
        dx = self._target[0] - self._odom.position.x
        dy = self._target[1] - self._odom.position.y
        tgt = np.array([dx, dy], dtype=np.float32)
        obs = np.concatenate([ranges, col, tgt])
        # obs = ranges
        

        # sanity check for debugging:
        if not np.isfinite(obs).all():
            raise ValueError(f"Non-finite observation: {obs}")

        return obs


    def step(self, action):

        obs, reward, done, info = None, 0.0, False, {}    

        v, delta = action
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
        #done = False
        #reward = 0.0

        yaw = self._robot_yaw()

        # print(self._odom.position.x," ",self._odom.position.y," ",self._odom.position.z," ",yaw)
        reward = 0.0



        
        # 在 step() 或者你计算 reward 的地方
        # --------------------------------------------------------
        # 1) 撞墙或圆柱立即终止
        if self._collided:
            # print('contact happen')
            reward += -50.0

        

            # 2) 计算到目标的欧氏距离

        # print("yaw = ",yaw)
        # print("target_yaw = ",self.target_yaw)


        yaw_dif = abs(self.target_yaw - yaw)

        # print('raw yaw difference = ',yaw_dif)

        if yaw_dif > pi:
            yaw_dif = pi - yaw_dif

        elif yaw_dif > pi/2 :
            yaw_dif = pi - yaw_dif
        # print('abs yaw difference = ',yaw_dif)

        self._ang_history.append(yaw_dif)

        bonus_ang = 0.0
        window_len_ang = 10
        if self._step_cnt % (window_len_ang) == 0 and len(self._dist_history) >= window_len_ang:
            window_curr = self._ang_history[-4:]
            avg_curr = sum(window_curr) /4.0

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
            avg_curr = sum(window_curr) /4.0

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
        r_iou = 0
        r_dist = 0
        r_time = 0

            # 目标到达
        if dist < 0.35:
            done   = True
            reward += +100.0

            # 超时
        elif self._step_cnt > self._max_steps:
            done   = True
            reward += -20.0

            # 常规 step，累加三部分 reward
        else:
            done = False

                # —— 1. IoU Reward —— 
                # robot_poly: 当前机器人底盘多边形
            robot_poly = self._robot_poly()
            target_poly = self._target_poly
            inter = robot_poly.intersection(target_poly).area
            union = robot_poly.union(target_poly).area
            iou = inter / union if union > 0 else 0.0
            w_iou = 20.0   # 你可以调这个权重
            r_iou = w_iou * iou

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
            reward += (delta_r_iou + r_iou + r_dist + r_time)

        self._robot_off_ground()
        if self._offground:
            # print('offground')
            reward += -10.0
            # 取当前位置的 x,y 和 yaw，强制 z=wheel_radius，roll=pitch=0
            ign_set_pose("jetauto",
                x=self._odom.position.x, y=self._odom.position.y, z=0,
                qx=0, qy=0, qz=math.sin(yaw/2), qw=math.cos(yaw/2))
            time.sleep(0.1)
            twist = Twist()
            twist.linear.x  = 0.0
            twist.angular.z = 0.0
            self._cmd_pub.publish(twist)
            rclpy.spin_once(self._node, timeout_sec=0.1)



        print("\nstep:", self._step_cnt)
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

        # print("coll:", self._collided)

        

        self._collided = False  # 重置碰撞标志
        idx = self._current_cfg_idx
        trials = self._cfg_trials[idx]
        succ = 0.0
        rate = 0.0
        # 2) 如果本 step 结束了
        if done:
            
            # 如果是成功到达
            if dist < 0.35 :
                self._cfg_successes[idx] += 1

            # 3) 检查是否要移除
            
            succ   = self._cfg_successes[idx]
            rate   = succ / trials
            if (trials > R_MAX) or (trials >= R_MIN and rate >= THRESH):
                self._pool.remove(idx)

            # 4) 在 info 里带上统计数据
        info['config_idx']      = idx
        info['episode_reward']  = self._episode_reward
        info['config_trials']   = trials
        info['config_successes']= succ
        info['config_rate']     = rate
        info['window_bonus_dis'] = bonus_dis
        info['window_bonus_ang'] = bonus_ang

        # return self._get_obs(), reward, done, info
        # Gymnasium expects: obs, reward, terminated, truncated, info
        terminated = done
        truncated  = False
        return obs, reward, terminated, truncated, info
# --------------------------------------------------------





    def render(self, mode='human'):
        pass

    def close(self):
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