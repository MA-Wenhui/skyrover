from .algo_wrapper import AlgorithmWrapperBase
from ..src.dcc_3d.environment3d import Environment3D
from ..src.dcc_3d.model import Network

import numpy as np
import torch

class DCCAlgorithmWrapper(AlgorithmWrapperBase):
    def __init__(self,agents, space_dim, obstacles):
        self.agents = agents
        self.space_dim = space_dim
        self.obstacles = set(obstacles)

        space = np.zeros(space_dim, dtype=int)
        # print(space)
        # print(obstacles)
        for obstacle in obstacles:
            obstacle = tuple(map(int, obstacle))  # Convert float to int
            print(obstacle)
            space[obstacle] = 1

        agents_pos = np.array([agent["start"] for agent in agents], dtype=int)
        goals_pos = np.array([agent["goal"] for agent in agents], dtype=int)

        self.env = Environment3D()
        self.env.load(space, agents_pos, goals_pos)
        self.current_positions = {}
        for agent in self.agents:
            self.current_positions[agent["name"]] = agent["start"]
        self.done = False
        self.episode_length = 0


    def init(self,network_path):
        print("Initializing DCC Algorithm and Environment")
        self.network_path = network_path
        self.device = torch.device("cpu")
        self.network = Network()
        self.network.eval()
        self.network.to(self.device)
        state_dict = torch.load(self.network_path , map_location=self.device)
        self.network.load_state_dict(state_dict)
        self.done = False
        self.episode_length = 0

    def reset(self,agents, space_dim, obstacles):
        self.agents = agents
        self.space_dim = space_dim
        self.obstacles = set(obstacles)

        space = np.zeros(space_dim, dtype=int)
        print(space)
        print(obstacles)
        for obstacle in obstacles:
            space[obstacle] = 1

        agents_pos = np.array([agent["start"] for agent in agents], dtype=int)
        goals_pos = np.array([agent["goal"] for agent in agents], dtype=int)

        self.env = Environment3D()
        self.env.load(space, agents_pos, goals_pos)
        self.current_positions = {}
        for agent in self.agents:
            self.current_positions[agent["name"]] = agent["start"]
        self.done = False
        self.episode_length = 0


    def step(self):
        if self.done:
            print("DCC Algorithm has completed the episode.")
            return self.current_positions, self.done
        if self.done:
            return
        obs, last_act, pos = self.env.observe()
        actions, q_val, _, _, comm_mask = self.network.step(
            torch.as_tensor(obs.astype(np.float32)).to(self.device),
            torch.as_tensor(last_act.astype(np.float32)).to(self.device),
            torch.as_tensor(pos.astype(int)).to(self.device),
        )
        (obs, last_act, pos), _, self.done, info = self.env.step(actions)
        for i, p in enumerate(pos):
            # self.current_positions[f"agent_{i}"] = tuple(p)
            self.current_positions[self.agents[i]["name"]] = tuple(p)
        self.episode_length +=1
        return self.current_positions, self.done

