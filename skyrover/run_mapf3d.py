#!/home/msh/miniconda3/envs/skyrover/bin/python
import sys
import os
sys.path.append(os.path.expanduser('~/miniconda3/envs/skyrover/lib/python3.12/site-packages'))
print(sys.executable)

import rclpy
from rclpy.node import Node as rosNode
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import os
from skyrover.pcd2grid import load_pcd,generate_3d_grid
from skyrover.wrapper.dcc_3d_wrapper import DCCAlgorithmWrapper
from skyrover.wrapper.cbs_3d_wrapper import CBSAlgorithmWrapper
from skyrover.wrapper.astar_3d_wrapper import AStarAlgorithmWrapper

import time
import math
import csv
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.vector3d_pb2 import Vector3d
from gz.msgs10.quaternion_pb2 import Quaternion
from gz.transport13 import Node as gzNode 
from datetime import datetime

import random

def set_entity_pose(entity, x, y, z, orientation_w=1.0):
    return

class Mapf3DExecutor(rosNode):
    def __init__(self):
        super().__init__('mapf_3d_publisher')

        # Declare algorithm parameter
        self.declare_parameter('alg', '3dcbs')
        self.algorithm_name = self.get_parameter('alg').get_parameter_value().string_value
        self.get_logger().info(f"Selected algorithm: {self.algorithm_name}")

        # Model path handling for 3ddcc
        self.model_path = None
        if self.algorithm_name == "3ddcc":
            self.declare_parameter('model_path', "65000.pth")
            self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
            # Convert model path to absolute path
            self.model_path = os.path.expanduser(self.model_path)  # Expands ~ to home directory if used
            self.model_path = os.path.abspath(self.model_path)      # Converts to absolute path
            self.get_logger().info(f"Using model path: {self.model_path}")

        # Declare parameter for PCD file path
        self.declare_parameter('pcd', 'map.pcd')
        self.pcd_file = self.get_parameter('pcd').get_parameter_value().string_value
        # Convert PCD file path to absolute path
        self.pcd_file = os.path.expanduser(self.pcd_file)  # Expands ~ to home directory if used
        self.pcd_file = os.path.abspath(self.pcd_file)      # Converts to absolute path
        self.get_logger().info(f"Using pcd path: {self.pcd_file}")
                               
        # Publisher for PointCloud2
        self.pc_publisher_ = self.create_publisher(PointCloud2, 'mapf_3d_pc', 10)
        self.grid_publisher_ = self.create_publisher(PointCloud2, 'mapf_3d_grid', 10)        
        self.agents_pos_publisher = self.create_publisher(PointCloud2, 'mapf_3d_agents_pos', 10)

        # Timer to periodically publish data
        self.timer = self.create_timer(1.0, self.timer_callback)

        # Load and process PCD data
        self.points = load_pcd(self.pcd_file)
        min_x = np.min(self.points[:, 0])
        min_y = np.min(self.points[:, 1])
        min_z = np.min(self.points[:, 2])

        print(f"Minimum X: {min_x}, Y: {min_y}, Z: {min_z}")
        self.min_bounds = [-21.0,-39.0,0.0]
        self.max_bounds = [21.0,23.0,15.0]
        self.grid = generate_3d_grid(self.points,np.array(self.min_bounds),np.array(self.max_bounds), 1.0)
        print(f"grid shape: {self.grid.shape}, self.min_bounds:{self.min_bounds}")
        self.obstacles = np.argwhere(self.grid == 1)*1.0+self.min_bounds

        
        # 找到所有空闲位置
        free_positions = np.argwhere(self.grid == 0)  # 位置值为0表示空闲
        free_positions = free_positions * 1.0 + self.min_bounds  # 转换为世界坐标系

        # 将位置转换为元组，并且限制z坐标不超过8
        free_positions = [tuple(pos) for pos in free_positions if pos[2] <= 8]

        # 随机选择3000个起始位置
        start_positions = random.sample(free_positions, 3000)

        # 从剩余的空闲位置中选择3000个目标位置
        # 排除已选择的起点，确保终点和起点不重复
        remaining_positions = [pos for pos in free_positions if pos not in start_positions]
        goal_positions = random.sample(remaining_positions, 3000)

        # 创建任务列表，每个任务包含起始位置和目标位置
        tasks = []
        for i in range(3000):
            task = {
                "name": f"a{i+1}",
                "start": start_positions[i],
                "goal": goal_positions[i]
            }
            tasks.append(task)

        # save tasks to file (你可以选择保存为JSON或其他格式)
        with open("tasks.json", "w") as f:
            import json
            json.dump(tasks, f, indent=4)

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        agents_pos_data = pc2.create_cloud_xyz32(header,start_positions)
        self.agents_pos_publisher.publish(agents_pos_data)  # Publish the agents' positions

        self.tasks = tasks
        self.reset_planner(self.tasks)
        self.step_count = 0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"positions_{timestamp}.csv"
        self.first_write = True


    def reset_planner(self,tasks):
        for item in tasks:
            item["start"] = self.world2planner(item["start"])
            item["goal"] = self.world2planner(item["goal"])
        for it in tasks:
            p = self.planner2world(it["start"])
            set_entity_pose(it["name"],p[0],p[1],p[2])

        if self.algorithm_name == "3dastar":
            self.planner = AStarAlgorithmWrapper(tasks,self.grid.shape,
                                                 [self.world2planner((p[0],p[1],p[2])) for p in self.obstacles])
        elif self.algorithm_name == "3dcbs":
            self.planner = CBSAlgorithmWrapper(tasks,self.grid.shape,
                                                [self.world2planner((p[0],p[1],p[2])) for p in self.obstacles])
        elif self.algorithm_name == "3ddcc":
            self.planner = DCCAlgorithmWrapper(tasks,self.grid.shape,
                                                [self.world2planner((p[0],p[1],p[2])) for p in self.obstacles])
            self.planner.init(self.model_path)
        print(f"planner init done")


    def world2planner(self,w):
        return (w[0] - self.min_bounds[0],w[1] - self.min_bounds[1],w[2] - self.min_bounds[2])

    def planner2world(self,p):
        return (p[0] + self.min_bounds[0],p[1] + self.min_bounds[1],p[2] + self.min_bounds[2])

    def save_positions_to_csv(self, step, positions):
        mode = 'w' if self.first_write else 'a'
        with open(self.filename, mode, newline='') as file:
            writer = csv.writer(file)

            if self.first_write:
                header = ['Step'] + list(positions.keys())
                writer.writerow(header)
                self.first_write = False

            row = [step] + [f"({p[0]},{p[1]},{p[2]})" for p in positions.values()]
            writer.writerow(row)

    def timer_callback(self):
        """Publish PointCloud2 message."""
        # Create PointCloud2 message
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        
        # Create point cloud data
        points = self.obstacles.reshape(-1, 3)
        pc_data = pc2.create_cloud_xyz32(header, points)
        
        self.grid_publisher_.publish(pc_data)
        pc_data = pc2.create_cloud_xyz32(header, self.points)
        self.pc_publisher_.publish(pc_data)
        self.get_logger().info('Publishing 3D grid points')
        if not self.planner.done:
            self.step_count += 1
            current_positions, done = self.planner.step()
            print(f"Step {self.step_count}: {current_positions}")
            self.save_positions_to_csv(self.step_count, current_positions)
            self.agent_positions = []
            for p in current_positions.values():
                p = self.planner2world(p)
                self.agent_positions.append([p[0], p[1], p[2]])  # Assuming p is a tuple (x, y, z)

            # Publish the agents' positions as a PointCloud2 message
            agents_pc_data = pc2.create_cloud_xyz32(header, self.agent_positions)
            self.agents_pos_publisher.publish(agents_pc_data)  # Publish the agents' positions

            for n,p in current_positions.items():
                p = self.planner2world(p)
                set_entity_pose(n,p[0],p[1],p[2])


def test_mapf(args=None):
    rclpy.init(args=args)
    node = Mapf3DExecutor()

    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()

def test_script(args=None):
    pass

def main(args=None):
    test_mapf(args=args)

if __name__ == '__main__':
    main()
