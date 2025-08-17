import os
import math
import random
import xacro
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose
 # 假设这些消息类型已经由 ros_gz_bridge 生成
from ament_index_python.packages import get_package_share_directory
from transforms3d import euler
from ros_gz_interfaces.msg import Entity
from shapely.geometry import Polygon
from typing import List, Tuple, Optional



# 超参
R_MAX  = 20
R_MIN  =  5
THRESH = 0.9

import subprocess
import shlex
import re

def ign_set_pose(entity_name: str,
                 x: float, y: float, z: float,
                 qx: float, qy: float, qz: float, qw: float,
                 world: str = "/world/all_training",
                 timeout_ms: int = 2000) -> bool:
    """
    Calls `ign service -s {world}/set_pose` to teleport `entity_name` and returns True on success.
    Raises RuntimeError on subprocess failure, ValueError on parse failure.
    """
    # Construct the protobuf-style request payload
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

    # Run the command and capture its stdout/stderr
    result = subprocess.run(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"`{' '.join(shlex.quote(c) for c in cmd)}` failed:\n"
            f"{result.stderr}"
        )

    # The service prints something like:
    #   data: true
    # or
    #   data: false
    m = re.search(r"data:\s*(true|false)", result.stdout)
    if not m:
        raise ValueError(f"Could not parse ign response:\n{result.stdout!r}")
    return (m.group(1) == "true")


def compute_ackermann(v: float, delta: float, wheel_base: float) :
    if abs(wheel_base) < 1e-6:
        raise ValueError("wheel_base must be non-zero")
    linear_velocity = v
    angular_velocity = v * math.tan(delta) / wheel_base if abs(delta) >= 1e-6 else 0.0
    return linear_velocity, angular_velocity







class ConfigStatus:
    def __init__(self, cfg: dict, idx: int):
        self.cfg = cfg
        self.idx = idx
        self.trials  = 0
        self.success = 0

    @property
    def success_rate(self):
        return self.success / self.trials if self.trials>0 else 0.0

    def should_remove(self):
        return (self.trials > R_MAX) or \
               (self.trials >= R_MIN and self.success_rate >= THRESH)


class Trainer(Node):
    def __init__(self):
        super().__init__('trainer')
                # 记得把 xacro 模板路径保存一下
        pkg = get_package_share_directory('robot_gazebo')
        self.xacro_path = os.path.join(pkg, 'urdf', 'robot.gazebo.xacro')
        # 1）读所有 config



        # this gives you /home/ubuntu/ros2_ws/src/robot_gazebo/train
        BASEDIR = os.path.dirname(__file__)

        # later, instead of open('configs.json') use:
        config_path = os.path.join(BASEDIR, 'configs.json')

        with open(config_path) as f:
            raw = json.load(f)
        self.pool = [ConfigStatus(cfg,i) for i,cfg in enumerate(raw)]






        # 3）Publisher & Subscriber
        self.cmd_pub     = self.create_publisher(Twist,     '/controller/cmd_vel', 10)
        self.odom_sub    = self.create_subscription(Odometry, '/odom',            self.odom_cb,    10)
        self.joint_sub   = self.create_subscription(Odometry, '/joint_states',    self.joint_cb,   10)
        self.scan_sub    = self.create_subscription(LaserScan, '/scan',           self.scan_cb,    10)
        self.touch_sub   = self.create_subscription(Bool,      '/cfg/touched',    self.touch_cb,   10)
        #self.pose_sub    = self.create_subscription(Entity,   '/world/all_training/pose/info', self.on_pose, 10)
        # internal state
        self.collided = False
        self.odom     = None
        self.scan     = None
        self.spawned  = False

        self.car_length = 0.316  # or 从 config 里读
        self.car_width  = 0.259

    def _generate_model_xml(self):
        """用 xacro 把 robot.gazebo.xacro 展开成 XML 字符串（SDF/URDF 都行）"""
        doc = xacro.process_file(self.xacro_path,
                                 mappings={'sim_ign':'true'})
        return doc.toxml()

    def odom_cb(self, msg: Odometry):
        self.odom = msg.pose.pose

    def joint_cb(self, msg):  pass  # 如果你需要关节状态

    def scan_cb(self, msg: LaserScan):
        self.scan = msg

    def touch_cb(self, msg: Bool):
        if msg.data:
            self.collided = True

    def _robot_poly(self) -> Polygon:
        """
        返回当前 odom.pose 下，机器人底盘在 XY 平面上的多边形轮廓。
        """
        if self.odom is None:
            # 还没收到任何里程计
            return Polygon()

        # 1) 读取位置
        x = self.odom.position.x
        y = self.odom.position.y

        # 2) 读取四元数，计算 yaw
        qx = self.odom.orientation.x
        qy = self.odom.orientation.y
        qz = self.odom.orientation.z
        qw = self.odom.orientation.w

        # 标准的 yaw 提取公式：
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # 3) 调用已有工具，算出四个角点
        corners = _robot_corners(
            x, y, yaw,
            self.car_length,
            self.car_width
        )

        # 4) 构造并返回 Polygon
        return Polygon(corners)


    def teleport(self,  px: float, py: float, pyaw: float):
        entity_name = 'jetauto'          # e.g. "jetauto"


        # build the Pose
        q = euler.euler2quat(0, 0, pyaw)
        pqx = q[1]
        pqy = q[2]
        pqz = q[3]
        pqw = q[0]   



        # call and wait
        success = ign_set_pose(
            entity_name="jetauto",
            x=px, y=py, z=0.0,
            qx=pqx, qy=pqy, qz=pqz, qw=pqw
        )
        if success:print("teleport done to",px,py)



    def run_episode(self, cfg_stat: ConfigStatus):
        cfg_stat.trials += 1
        self.collided = False
        sx, sy, syaw = cfg_stat.cfg['start_pose']
        tx, ty       = cfg_stat.cfg['target_position']
        coords = cfg_stat.cfg['target_poly']  # [[x1,y1], [x2,y2], …]
        self._target_poly = Polygon(coords)

        # 放置小车
        self.teleport(sx, sy, syaw)

        max_steps = 300
        for step in range(max_steps):
            # 这里获取观测（用 self.scan, self.odom, joint_states…）
            obs_lidar = self.scan
            obs_collided = self.collided
            print(obs_collided)
            # agent 输出
            v, delta = 0.0,0.0
            #agent(obs_collided,obs_lidar)0.0,0.0

            v , omega=compute_ackermann(v,delta,0.213)

            # 发布到 /controller/cmd_vel
            twist = Twist()
            twist.linear.x  = v
            twist.angular.z = omega
            self.cmd_pub.publish(twist)

            rclpy.spin_once(self, timeout_sec=0.1)

            # 碰撞检测


            # 到达终点检测
            if self.odom is not None:
                dx = self.odom.position.x - tx
                dy = self.odom.position.y - ty
                if dx*dx + dy*dy < 0.05**2:
                    self.get_logger().info(f'[Cfg {cfg_stat.idx}] Success')
                    return True

        # 超时视为失败
        self.get_logger().info(f'[Cfg {cfg_stat.idx}] Timeout')
        return False

    def train(self):
        while self.pool:
            cfg = random.choice(self.pool)
            succ = self.run_episode(cfg)
            if succ:
                cfg.success += 1
            # 判断是否移除
            if cfg.should_remove():
                self.get_logger().info(f'Removing config {cfg.idx}')
                self.pool.remove(cfg)
        self.get_logger().info('*** All configs done ***')

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



def main(args=None):
    rclpy.init(args=args)
    trainer = Trainer()
    trainer.train()
    rclpy.shutdown()

if __name__=='__main__':
    main()
