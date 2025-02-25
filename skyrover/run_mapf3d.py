#!/home/msh/miniconda3/envs/skyrover/bin/python
import sys
import os
sys.path.append(os.path.expanduser('~/miniconda3/envs/skyrover/lib/python3.12/site-packages'))
print(sys.executable)

import rclpy
import argparse
from skyrover.executor import Mapf3DExecutor

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

def main(args=None):

    parser = argparse.ArgumentParser(description="Mapf3DExecutor Node")
    parser.add_argument("--alg", type=str, default="3dcbs", help="algorithm for 3D MAPF")
    parser.add_argument("--pcd", type=str, default="map.pcd", help="point cloud file")
    parser.add_argument("--model", type=str, default="model.pth", help="model used for 3D DCC")
    parser.add_argument("--pub_gz", type=bool, default=False, help="whether to publish gz manipulation")

    known_args, unknown_args = parser.parse_known_args()
    print(f"Custom param received: {known_args.alg}") 

    rclpy.init(args=unknown_args)
    e = Mapf3DExecutor(known_args.alg,known_args.pcd,known_args.model,known_args.pub_gz)
    e.set_tasks(tasks)
    rclpy.spin(e)
    
    e.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
