
import os
import csv
import numpy as np
from datetime import datetime

from skyrover.wrapper.dcc_3d_wrapper import DCCAlgorithmWrapper
from skyrover.wrapper.cbs_3d_wrapper import CBSAlgorithmWrapper
from skyrover.wrapper.astar_3d_wrapper import AStarAlgorithmWrapper

class Mapf3DExecutor:
    def __init__(self, alg, grid, world_origin, model, publish_callback=None):
        """
        Initialize the 3D Multi-Agent Path Finding (MAPF) executor.

        Parameters:
        - alg: The selected pathfinding algorithm (e.g., 3ddcc, 3dcbs, 3dastar).
        - grid: The grid representation of the environment.
        - world_origin: The origin coordinates in the world frame.
        - model: The path to the model (used for 3ddcc algorithm).
        - publish_callback: A callback function to publish data (e.g., PointCloud2 messages).
        """
        self.grid = grid
        self.algorithm_name = alg
        self.world_origin = world_origin
        self.model_path = None
        self.publish_callback = publish_callback
        self.tasks = None

        if self.algorithm_name == "3ddcc":
            self.model_path = model
            # Convert model path to absolute path
            self.model_path = os.path.expanduser(self.model_path)
            self.model_path = os.path.abspath(self.model_path)

        self.obstacles = np.argwhere(self.grid == 1)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"positions_{timestamp}.csv"
        self.first_write = True

        self.planner = None

    def set_tasks(self, tasks):
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
        
        self.tasks = planner_tasks
        if self.algorithm_name == "3dastar":
            self.planner = AStarAlgorithmWrapper(planner_tasks, self.grid.shape,
                                                 [(p[0], p[1], p[2]) for p in self.obstacles])
        elif self.algorithm_name == "3dcbs":
            self.planner = CBSAlgorithmWrapper(planner_tasks, self.grid.shape,
                                               [(p[0], p[1], p[2]) for p in self.obstacles])
        elif self.algorithm_name == "3ddcc":
            self.planner = DCCAlgorithmWrapper(planner_tasks, self.grid.shape,
                                               [(p[0], p[1], p[2]) for p in self.obstacles])
            self.planner.init(self.model_path)
        else:
            raise Exception(f"Unsupported algorithm: {self.algorithm_name}")

        if self.publish_callback:
            self.publish_callback("agents_pos", self.agent_positions)  # Publish initial agent positions

        self.step_count = 0
        print("Planner initialization done.")

    def world2planner(self, w):
        return (w[0] - self.world_origin[0], w[1] - self.world_origin[1], w[2] - self.world_origin[2])

    def planner2world(self, p):
        return (p[0] + self.world_origin[0], p[1] + self.world_origin[1], p[2] + self.world_origin[2])

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
    def get_obstacles(self):
        return self.obstacles.reshape(-1, 3)

    def step(self):
        """
        Perform a single step of the planner and publish the results.
        """
        if self.planner and not self.planner.done:
            self.step_count += 1
            cur, done = self.planner.step()

            print(f"Step {self.step_count}: {cur}")

            self.save_positions_to_csv(self.step_count, cur)

            self.agent_positions = []
            for p in cur.values():
                self.agent_positions.append([p[0], p[1], p[2]])  # Assuming p is a tuple (x, y, z)

            return cur,self.obstacles.reshape(-1, 3)
