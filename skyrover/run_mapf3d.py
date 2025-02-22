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

def set_entity_pose(entity, x, y, z, orientation_w=1.0):
    """
    Set the entity pose by calling the Gazebo service.

    Parameters:
    - entity: The name of the entity (e.g., x500_0, delivery_robot_0).
    - x, y, z: The position coordinates.
    - orientation_w: The w component of the quaternion orientation (default is 1.0).
    """
    node = gzNode()
    service_name = "/world/warehouse/set_pose"
    request = Pose()
    request.name = entity
    request.position.x = x
    request.position.y = y
    request.position.z = z
    request.orientation.x = 0.0
    request.orientation.y = 0.0
    request.orientation.z = 0.0
    request.orientation.w = orientation_w
    response = Boolean()
    timeout = 200  # Timeout in milliseconds

    result, response = node.request(service_name, request, Pose, Boolean, timeout)
    # print(f"Set {entity} position to ({x}, {y}, {z}), Result:", result, "\nResponse:", response.data)

init_pose = [
    {"name": "x500_0", "start": (7, -28, 3), "goal": (7, -28, 3)},
    {"name": "x500_1", "start": (11, -27, 3), "goal": (11, -27, 3)},
    {"name": "x500_2", "start": (11, -30, 3), "goal": (11, -30, 3)},
    {"name": "x500_3", "start": (-13, -27, 3), "goal": (-13, -27, 3)},
    {"name": "x500_4", "start": (-13, -30, 3), "goal": (-13, -30, 3)},
    {"name": "x500_5", "start":  (-9, -30, 3), "goal": (-9, -30, 3)},
    {"name": "delivery_robot_0", "start": (9, 0, 0), "goal": (9, 0, 0)},
    {"name": "delivery_robot_1", "start": (9, -1, 0), "goal": (9, -1, 0)},
    {"name": "delivery_robot_2", "start": (9, -2, 0), "goal": (9, -2, 0)},
    {"name": "delivery_robot_3", "start": (9, -3, 0), "goal": (9, -3, 0)},
    {"name": "delivery_robot_4", "start": (9, -4, 0), "goal": (9, -4, 0)},
    {"name": "delivery_robot_5", "start": (9, -5, 0), "goal": (9, -5, 0)},
    {"name": "delivery_robot_6", "start": (9, -6, 0), "goal": (9, -6, 0)},
    {"name": "delivery_robot_7", "start": (9, -7, 0), "goal": (9, -7, 0)},
    {"name": "delivery_robot_8", "start": (9, -8, 0), "goal": (9, -8, 0)},
    {"name": "delivery_robot_9", "start": (9, -9, 0), "goal": (9, -9, 0)},
    {"name": "delivery_robot_10", "start": (9, -10, 0), "goal":(9, -10, 0)},
    {"name": "delivery_robot_11", "start": (9, -11, 0), "goal": (9, -11, 0)},
    {"name": "delivery_robot_12", "start": (9, -12, 0), "goal": (9, -12, 0)},
    {"name": "delivery_robot_13", "start": (9, -13, 0), "goal": (9, -13, 0)},
    {"name": "delivery_robot_14", "start": (9, -14, 0), "goal": (9, -14, 0)},
    {"name": "delivery_robot_15", "start": (9, -15, 0), "goal": (9, -15, 0)},
]

task_phrase_1 = [
    {"name": "x500_0", "start": (11, 0, 3), "goal": (7, -28, 3)},
    {"name": "x500_1", "start": (11, -1, 3), "goal": (11, -27, 3)},
    {"name": "x500_2", "start": (11, -2, 3), "goal": (11, -30, 3)},
    {"name": "x500_3", "start": (11, -3, 3), "goal": (-13, -27, 3)},
    {"name": "x500_4", "start": (11, -4, 3), "goal": (-13, -30, 3)},
    {"name": "x500_5", "start": (11, -5, 3), "goal": (-9, -30, 3)},
    {"name": "delivery_robot_0", "start": (9, 0, 0), "goal": (9, 0, 0)},
    {"name": "delivery_robot_1", "start": (9, -1, 0), "goal": (9, -1, 0)},
    {"name": "delivery_robot_2", "start": (9, -2, 0), "goal": (9, -2, 0)},
    {"name": "delivery_robot_3", "start": (9, -3, 0), "goal": (9, -3, 0)},
    {"name": "delivery_robot_4", "start": (9, -4, 0), "goal": (9, -4, 0)},
    {"name": "delivery_robot_5", "start": (9, -5, 0), "goal": (9, -5, 0)},
    {"name": "delivery_robot_6", "start": (9, -6, 0), "goal": (9, -6, 0)},
    {"name": "delivery_robot_7", "start": (9, -7, 0), "goal": (9, -7, 0)},
    {"name": "delivery_robot_8", "start": (9, -8, 0), "goal": (9, -8, 0)},
    {"name": "delivery_robot_9", "start": (9, -9, 0), "goal": (9, -9, 0)},
    {"name": "delivery_robot_10", "start": (9, -10, 0), "goal":(9, -10, 0)},
    {"name": "delivery_robot_11", "start": (9, -11, 0), "goal": (9, -11, 0)},
    {"name": "delivery_robot_12", "start": (9, -12, 0), "goal": (9, -12, 0)},
    {"name": "delivery_robot_13", "start": (9, -13, 0), "goal": (9, -13, 0)},
    {"name": "delivery_robot_14", "start": (9, -14, 0), "goal": (9, -14, 0)},
    {"name": "delivery_robot_15", "start": (9, -15, 0), "goal": (9, -15, 0)},
]

task_phrase_2 = [
    {"name": "x500_0", "start": (7, -28, 3), "goal": (7, -28, 3)},
    {"name": "x500_1", "start": (11, -27, 3), "goal": (11, -27, 3)},
    {"name": "x500_2", "start": (11, -30, 3), "goal": (11, -30, 3)},
    {"name": "x500_3", "start": (-13, -27, 3), "goal": (-13, -27, 3)},
    {"name": "x500_4", "start": (-13, -30, 3), "goal": (-13, -30, 3)},
    {"name": "x500_5", "start":  (-9, -30, 3), "goal": (-9, -30, 3)},
    {"name": "delivery_robot_0", "start": (9, 0, 0), "goal": (7, -27, 0)},
    {"name": "delivery_robot_1", "start": (9, -1, 0), "goal": (8, -27, 0)},
    {"name": "delivery_robot_2", "start": (9, -2, 0), "goal": (9, -27, 0)},
    {"name": "delivery_robot_3", "start": (9, -3, 0), "goal": (10, -27, 0)},
    {"name": "delivery_robot_4", "start": (9, -4, 0), "goal": (7, -30, 0)},
    {"name": "delivery_robot_5", "start": (9, -5, 0), "goal": (8, -30, 0)},
    {"name": "delivery_robot_6", "start": (9, -6, 0), "goal": (9, -30, 0)},
    {"name": "delivery_robot_7", "start": (9, -7, 0), "goal": (10, -30, 0)},
    {"name": "delivery_robot_8", "start": (9, -8, 0), "goal": (-9, -27, 0)},
    {"name": "delivery_robot_9", "start": (9, -9, 0), "goal": (-10, -27, 0)},
    {"name": "delivery_robot_10", "start": (9, -10, 0), "goal": (-11, -27, 0)},
    {"name": "delivery_robot_11", "start": (9, -11, 0), "goal": (-12, -27, 0)},
    {"name": "delivery_robot_12", "start": (9, -12, 0), "goal": (-9, -30, 0)},
    {"name": "delivery_robot_13", "start": (9, -13, 0), "goal": (-10, -30, 0)},
    {"name": "delivery_robot_14", "start": (9, -14, 0), "goal": (-11, -30, 0)},
    {"name": "delivery_robot_15", "start": (9, -15, 0), "goal": (-12, -30, 0)},
]

task_phrase_3 = [
    {"name": "x500_0", "start": (7, -28, 3), "goal": (0, 5, 3)},
    {"name": "x500_1", "start": (11, -27, 3), "goal": (0, 8, 3)},
    {"name": "x500_2", "start": (11, -30, 3), "goal": (0, 11, 3)},
    {"name": "x500_3", "start": (-13, -27, 3), "goal": (0, -5, 3)},
    {"name": "x500_4", "start": (-13, -30, 3), "goal": (0, -8, 3)},
    {"name": "x500_5", "start":  (-9, -30, 3), "goal": (0, -11, 3)},
    {"name": "delivery_robot_0", "start": (7, -27, 0), "goal": (0,5,0)},
    {"name": "delivery_robot_1", "start": (8, -27, 0), "goal": (0,9,0)},
    {"name": "delivery_robot_2", "start": (9, -27, 0), "goal": (0,13,0)},
    {"name": "delivery_robot_3", "start": (10, -27, 0), "goal": (0,17,0)},
    {"name": "delivery_robot_4", "start": (7, -30, 0), "goal": (0,-5,0)},
    {"name": "delivery_robot_5", "start": (8, -30, 0), "goal": (0,-9,0)},
    {"name": "delivery_robot_6", "start": (9, -30, 0), "goal": (0,-13,0)},
    {"name": "delivery_robot_7", "start": (10, -30, 0), "goal": (0,-17,0)},
    {"name": "delivery_robot_8", "start": (-9, -27, 0), "goal": (-6,5,0)},
    {"name": "delivery_robot_9", "start": (-10, -27, 0), "goal": (-6,9,0)},
    {"name": "delivery_robot_10", "start": (-11, -27, 0), "goal": (-6,13,0)},
    {"name": "delivery_robot_11", "start": (-12, -27, 0), "goal": (-6,17,0)},
    {"name": "delivery_robot_12", "start": (-9, -30, 0), "goal": (-6,-5,0)},
    {"name": "delivery_robot_13", "start": (-10, -30, 0), "goal": (-6,-9,0)},
    {"name": "delivery_robot_14", "start": (-11, -30, 0), "goal": (-6,-13,0)},
    {"name": "delivery_robot_15", "start": (-12, -30, 0), "goal": (-6,-17,0)},
]

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
        # print([(p[0],p[1],p[2]) for p in self.obstacles])
        self.task_stage = [init_pose,task_phrase_1,task_phrase_2,task_phrase_3]
        self.reset_planner(self.task_stage[0])
        self.task_stage = self.task_stage[1:]
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

            for n,p in current_positions.items():
                p = self.planner2world(p)
                set_entity_pose(n,p[0],p[1],p[2])
        else:
            if self.task_stage:
                self.reset_planner(self.task_stage[0])
                self.task_stage = self.task_stage[1:]


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
