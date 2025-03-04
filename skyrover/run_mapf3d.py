
import argparse
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header

from skyrover.interfaces import gz_srv
from skyrover.executor import Mapf3DExecutor
from skyrover.pcd2grid import load_pcd, generate_3d_grid

tasks = [
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

def process_pcd(pcd_file,min_bounds,max_bounds,resolution=1.0):
    # Load and process PCD data
    points = load_pcd(pcd_file)
    grid = generate_3d_grid(points,np.array(min_bounds),np.array(max_bounds), resolution)
    return grid



class Mapf3DROSNode(Node):
    def __init__(self,alg,grid,world_origin,model,pub_gz):
        super().__init__('mapf_3d_node')
        self.pc_publisher_ = self.create_publisher(PointCloud2, 'mapf_3d_pc', 10)
        self.grid_publisher_ = self.create_publisher(PointCloud2, 'mapf_3d_grid', 10)
        self.agents_pos_publisher = self.create_publisher(PointCloud2, 'mapf_3d_agents_pos', 10)

        # Initialize the executor
        self.executor = Mapf3DExecutor(
            alg,
            grid,
            world_origin,
            model
        )
        self.pub_gz = pub_gz
        self.executor.set_tasks(tasks)

        # Timer to periodically call step
        self.timer = self.create_timer(1.0, self.timer_callback)


    def timer_callback(self):
        agents_pos, obstacle = self.executor.step()
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        agents_pc_data = pc2.create_cloud_xyz32(header, agents_pos)
        self.agents_pos_publisher.publish(agents_pc_data)
        grid_pc_data = pc2.create_cloud_xyz32(header, obstacle)
        self.grid_publisher_.publish(grid_pc_data)
        if self.pub_gz:
            for n, p in agents_pos.items():
                p = self.executor.planner2world(p)
                gz_srv.set_entity_pose(n, p[0], p[1], p[2], 1.0, "/world/warehouse/set_pose")

def main(args=None):
    parser = argparse.ArgumentParser(description="Mapf3DExecutor Node")
    parser.add_argument("--alg", type=str, default="3dcbs", help="algorithm for 3D MAPF")
    parser.add_argument("--pcd", type=str, default="map.pcd", help="point cloud file")
    parser.add_argument("--model", type=str, default="model.pth", help="model used for 3D DCC")
    parser.add_argument("--pub_gz", type=bool, default=False, help="whether to publish gz manipulation")

    known_args, unknown_args = parser.parse_known_args()
    min_bounds = [-21.0,-39.0,0.0]
    max_bounds = [21.0,23.0,15.0]
    print(f"Custom param received: {known_args.alg}") 
    grid = process_pcd(known_args.pcd,min_bounds,max_bounds)

    rclpy.init(args=args)
    node = Mapf3DROSNode(known_args.alg,grid,min_bounds,known_args.model,known_args.pub_gz)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
