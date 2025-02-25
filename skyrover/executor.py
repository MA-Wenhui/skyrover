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
    - service_name: The Gazebo service name for setting the pose.
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
    def __init__(self,alg,grid,world_origin,model,pub_gz=False):
        """
        Initialize the 3D Multi-Agent Path Finding (MAPF) executor.

        Parameters:
        - alg: The selected pathfinding algorithm (e.g., 3ddcc, 3dcbs, 3dastar).
        - grid: The grid representation of the environment.
        - world_origin: The origin coordinates in the world frame.
        - model: The path to the model (used for 3ddcc algorithm).
        - pub_gz: Boolean indicating whether to publish agent positions to Gazebo.
        """
        super().__init__('mapf_3d_publisher')
        self.grid = grid
        self.pub_gz = pub_gz
        self.algorithm_name = alg
        self.get_logger().info(f"Selected algorithm: {self.algorithm_name}")
        self.world_origin = world_origin
        self.model_path = None
        if self.algorithm_name == "3ddcc":
            self.model_path = model
            # Convert model path to absolute path
            self.model_path = os.path.expanduser(self.model_path)
            self.model_path = os.path.abspath(self.model_path)
            self.get_logger().info(f"Using model path: {self.model_path}")

        # Publisher for PointCloud2
        self.pc_publisher_ = self.create_publisher(PointCloud2, 'mapf_3d_pc', 10)
        self.grid_publisher_ = self.create_publisher(PointCloud2, 'mapf_3d_grid', 10)        
        self.agents_pos_publisher = self.create_publisher(PointCloud2, 'mapf_3d_agents_pos', 10)

        # Timer to periodically publish data
        self.timer = self.create_timer(1.0, self.timer_callback)

 
        self.obstacles = np.argwhere(self.grid == 1)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"positions_{timestamp}.csv"
        self.first_write = True

        self.planner = None

    

    def set_tasks(self,tasks):
        """
        Set the tasks for the planner by converting world coordinates to planner coordinates.

        Parameters:
        - tasks: A list of task dictionaries with 'start' and 'goal' positions.
        """
        self.agent_positions = []
        planner_tasks = tasks
        for item in planner_tasks:
            self.agent_positions.append(item["start"])
            item["start"] = self.world2planner(item["start"])
            item["goal"] = self.world2planner(item["goal"])

        if self.algorithm_name == "3dastar":
            self.planner = AStarAlgorithmWrapper(planner_tasks,self.grid.shape,
                                                 [(p[0],p[1],p[2]) for p in self.obstacles])
        elif self.algorithm_name == "3dcbs":
            self.planner = CBSAlgorithmWrapper(planner_tasks,self.grid.shape,
                                                [(p[0],p[1],p[2]) for p in self.obstacles])
        elif self.algorithm_name == "3ddcc":
            self.planner = DCCAlgorithmWrapper(planner_tasks,self.grid.shape,
                                                [(p[0],p[1],p[2]) for p in self.obstacles])
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
        return (w[0] - self.world_origin[0],w[1] - self.world_origin[1],w[2] - self.world_origin[2])

    def planner2world(self,p):
        return (p[0] + self.world_origin[0],p[1] + self.world_origin[1],p[2] + self.world_origin[2])

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
        # Create PointCloud2 message
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        
        obs = pc2.create_cloud_xyz32(header, self.obstacles.reshape(-1, 3))
        self.grid_publisher_.publish(obs)

        self.get_logger().info('Publishing 3D grid points')
        if self.planner and not self.planner.done:
            self.step_count += 1
            cur, done = self.planner.step()

            print(f"Step {self.step_count}: {cur}")

            self.save_positions_to_csv(self.step_count, cur)

            self.agent_positions = []
            for p in cur.values():
                self.agent_positions.append([p[0], p[1], p[2]])  # Assuming p is a tuple (x, y, z)

            # Publish the agents' positions as a PointCloud2 message
            agents_pc_data = pc2.create_cloud_xyz32(header, self.agent_positions)
            self.agents_pos_publisher.publish(agents_pc_data)  # Publish the agents' positions
            
            if self.pub_gz:
                for n,p in cur.items():
                    p = self.planner2world(p)
                    set_entity_pose(n,p[0],p[1],p[2],1.0,"/world/warehouse/set_pose")
