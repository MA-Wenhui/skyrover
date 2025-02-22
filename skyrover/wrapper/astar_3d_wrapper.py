from .algo_wrapper import AlgorithmWrapperBase
from .astar_3d import a_star_3d
import numpy as np
import random

class AStarAlgorithmWrapper(AlgorithmWrapperBase):
    def __init__(self, agents, space_dim, obstacles):
        """
        初始化AStarAlgorithmWrapper

        :param agents: 代理信息列表，格式为 [{"name": str, "start": tuple, "goal": tuple}, ...]
        :param space_dim: 3D空间的维度，格式为 (x, y, z)
        :param obstacles: 初始障碍物列表，格式为 [(x, y, z), ...]
        """
        self.agents = agents
        self.space_dim = space_dim
        self.obstacles = set(obstacles)
        self.grid = [[[0 for _ in range(space_dim[2])] for _ in range(space_dim[1])] for _ in range(space_dim[0])]
        self.planned_paths = {}
        self.current_positions = {}

        # 初始化每个代理的路径
        self.plan_paths()
        self.done = False
        self.episode_length = 0

    def plan_paths(self):
        """使用A*算法为所有代理规划路径"""
        remaining_agents = self.agents[:]
        max_attempts = len(self.agents)  # 最大尝试次数，避免死循环
        attempts = 0

        while remaining_agents and attempts < max_attempts:
            random.shuffle(remaining_agents)  # 随机顺序规划路径
            next_remaining_agents = []

            for agent in remaining_agents:
                start = tuple(agent["start"])
                goal = tuple(agent["goal"])

                # 调用A*算法进行路径规划
                path = a_star_3d(self.grid, start, goal, self.obstacles)

                if path is None:
                    print(f"Path planning failed for agent {agent['name']}. Retrying...")
                    next_remaining_agents.append(agent)  # 将失败的代理加入下一轮重试
                else:
                    # 存储规划好的路径和当前代理的位置
                    self.planned_paths[agent["name"]] = path
                    self.current_positions[agent["name"]] = start

                    # 将路径添加到障碍物，避免路径冲突
                    self.obstacles.update(path)

            remaining_agents = next_remaining_agents
            attempts += 1

        # 检查是否仍有未规划的代理
        if remaining_agents:
            print(f"Failed to find paths for some agents: {[agent['name'] for agent in remaining_agents]}")
        
    def init(self):
        """初始化动作和状态"""
        # self.actions = {agent["name"]: [] for agent in self.agents}
        self.done = False
        self.episode_length = 0

    def reset(self, agents, space_dim, obstacles):
        """
        初始化AStarAlgorithmWrapper

        :param agents: 代理信息列表，格式为 [{"name": str, "start": tuple, "goal": tuple}, ...]
        :param space_dim: 3D空间的维度，格式为 (x, y, z)
        :param obstacles: 初始障碍物列表，格式为 [(x, y, z), ...]
        """
        self.agents = agents
        self.space_dim = space_dim
        self.obstacles = set(obstacles)
        self.grid = [[[0 for _ in range(space_dim[2])] for _ in range(space_dim[1])] for _ in range(space_dim[0])]
        self.planned_paths = {}
        self.current_positions = {}

        # 初始化每个代理的路径
        self.plan_paths()
        self.done = False
        self.episode_length = 0

    def step(self):
        """
        获取所有代理的下一个位置，更新位置状态
        """
        if self.done:
            print("All agents have reached their goals.")
            return self.current_positions, self.done

        all_done = True

        for agent in self.agents:
            name = agent["name"]
            path = self.planned_paths.get(name, [])
            current_position = self.current_positions.get(name)

            # 检查代理是否已经完成
            if current_position == tuple(agent["goal"]):
                continue

            all_done = False

            # 获取路径中的下一个位置
            if path:
                next_position = path.pop(0)  # 从路径中取出下一个位置
                self.current_positions[name] = next_position
                # self.actions[name] = next_position
            else:
                print(f"Agent {name} has no more moves but has not reached the goal.")

        self.episode_length +=1
        self.done = all_done
        return self.current_positions, self.done


