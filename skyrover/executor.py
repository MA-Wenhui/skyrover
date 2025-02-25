import os
import csv
import numpy as np
import random
from datetime import datetime
from rclpy.node import Node as rosNode
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.boolean_pb2 import Boolean
from gz.transport13 import Node as gzNode 


from skyrover.pcd2grid import load_pcd, generate_3d_grid
from skyrover.wrapper.dcc_3d_wrapper import DCCAlgorithmWrapper
from skyrover.wrapper.cbs_3d_wrapper import CBSAlgorithmWrapper
from skyrover.wrapper.astar_3d_wrapper import AStarAlgorithmWrapper


def set_entity_pose(entity, x, y, z, orientation_w=1.0,service_name="/world/warehouse/set_pose"):
    """
    Set the entity pose by calling the Gazebo service.

    Parameters:
    - entity: The name of the entity (e.g., x500_0, delivery_robot_0).
    - x, y, z: The position coordinates.
    - orientation_w: The w component of the quaternion orientation (default is 1.0).
    """
    node = gzNode()
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


class Mapf3DExecutor(rosNode):
    def __init__(self,alg,pcd,model,pub_gz=False):
        super().__init__('mapf_3d_publisher')
        self.pub_gz = pub_gz
        self.algorithm_name = alg
        self.get_logger().info(f"Selected algorithm: {self.algorithm_name}")

        self.model_path = None
        if self.algorithm_name == "3ddcc":
            self.model_path = model
            # Convert model path to absolute path
            self.model_path = os.path.expanduser(self.model_path)
            self.model_path = os.path.abspath(self.model_path)
            self.get_logger().info(f"Using model path: {self.model_path}")

        # Declare parameter for PCD file path
        self.pcd_file = pcd
        self.pcd_file = os.path.expanduser(self.pcd_file)
        self.pcd_file = os.path.abspath(self.pcd_file) 
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
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"positions_{timestamp}.csv"
        self.first_write = True

        self.planner = None

    

    def set_tasks(self,tasks):
        self.agent_positions = []
        for item in tasks:
            item["start"] = self.world2planner(item["start"])
            item["goal"] = self.world2planner(item["goal"])
            self.agent_positions.append(item["start"])

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
        else:
            raise Exception(f"unsupport alg: {self.algorithm_name}")


        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        agents_pos_data = pc2.create_cloud_xyz32(header,self.agent_positions)
        self.agents_pos_publisher.publish(agents_pos_data)  # Publish the agents' positions

        
        self.step_count = 0
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
        if self.planner and not self.planner.done:
            self.step_count += 1
            cur, done = self.planner.step()

            print(f"Step {self.step_count}: {cur}")

            self.save_positions_to_csv(self.step_count, cur)

            self.agent_positions = []
            for p in cur.values():
                p = self.planner2world(p)
                self.agent_positions.append([p[0], p[1], p[2]])  # Assuming p is a tuple (x, y, z)

            # Publish the agents' positions as a PointCloud2 message
            agents_pc_data = pc2.create_cloud_xyz32(header, self.agent_positions)
            self.agents_pos_publisher.publish(agents_pc_data)  # Publish the agents' positions
            
            if self.pub_gz:
                for n,p in cur.items():
                    p = self.planner2world(p)
                    set_entity_pose(n,p[0],p[1],p[2],"/world/warehouse/set_pose")
