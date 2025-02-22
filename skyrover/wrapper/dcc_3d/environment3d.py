import matplotlib.pyplot as plt
import numpy as np
import random
from typing import List
from . import config as config

ACTION_LIST = np.array(
    [
        [0, 0, 1],
        [0, 0, -1],  # up, down (z-axis)
        [-1, 0, 0],
        [1, 0, 0],  # left, right (x-axis)
        [0, 1, 0],
        [0, -1, 0],  # front, back (y-axis)
        [0, 0, 0],  # stay
    ],
    dtype=int,
)

DIRECTION_TO_ACTION = {
    (0, 0, 1): 0,  # up
    (0, 0, -1): 1,  # down
    (-1, 0, 0): 2,  # left
    (1, 0, 0): 3,  # right
    (0, 1, 0): 4,  # front
    (0, -1, 0): 5,  # back
    (0, 0, 0): 6,  # stay
}

ACTION_TO_STR = {
    0: "up",  # 向上移动
    1: "down",  # 向下移动
    2: "left",  # 向左移动
    3: "right",  # 向右移动
    4: "front",  # 向前移动
    5: "back",  # 向后移动
    6: "stay",  # 停留
}


class Environment3D:

    def __init__(
        self,
        num_agents: int = config.init_env_settings[0],
        space_length: int = config.init_env_settings[1],
        obs_radius: int = config.obs_radius,
        reward_fn: dict = config.reward_fn,
        curriculum=False,
        init_env_settings_set=config.init_env_settings,
    ):

        self.curriculum = curriculum
        if curriculum:
            self.env_set = [init_env_settings_set]
            self.num_agents = init_env_settings_set[0]
            self.space_size = (init_env_settings_set[1], init_env_settings_set[1],init_env_settings_set[1])
        else:
            self.num_agents = num_agents
            self.space_size = (space_length, space_length,space_length)

        self.obstacle_density = 0.3
        self.space = np.random.choice(
            2, self.space_size, p=[1 - self.obstacle_density, self.obstacle_density]
        ).astype(int)

        self.obs_radius = obs_radius
        self.reward_fn = reward_fn
        partition_list = self._space_partition()
        while len(partition_list) == 0:
            self.space = np.random.choice(
                2, self.space_size, p=[1 - self.obstacle_density, self.obstacle_density]
            ).astype(int)
            partition_list = self._space_partition()

        self.agents_pos = np.empty((self.num_agents, 3), dtype=int)
        self.goals_pos = np.empty((self.num_agents, 3), dtype=int)
        pos_num = sum([len(partition) for partition in partition_list])

        for i in range(self.num_agents):

            pos_idx = random.randint(0, pos_num - 1)
            partition_idx = 0
            for partition in partition_list:
                if pos_idx >= len(partition):
                    pos_idx -= len(partition)
                    partition_idx += 1
                else:
                    break

            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.agents_pos[i] = np.asarray(pos, dtype=int)

            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.goals_pos[i] = np.asarray(pos, dtype=int)

            partition_list = [
                partition for partition in partition_list if len(partition) >= 2
            ]
            pos_num = sum([len(partition) for partition in partition_list])

        self.steps = 0
        self.last_actions = np.zeros((self.num_agents, config.action_dim), dtype=bool)

    def load(self, space: np.ndarray, agents_pos: np.ndarray, goals_pos: np.ndarray):
        self.space = np.copy(space)
        self.agents_pos = np.copy(agents_pos)
        self.goals_pos = np.copy(goals_pos)
        self.num_agents = agents_pos.shape[0]
        self.space_size = (self.space.shape[0], self.space.shape[1], self.space.shape[2])
        self.steps = 0
        self.last_actions = np.zeros((self.num_agents, config.action_dim), dtype=bool)

    def set_goal(self,agent,pos):
        assert agent < self.num_agents
        self.goals_pos[agent] = pos

    def update_env_settings_set(self, new_env_settings_set):
        self.env_set = new_env_settings_set

    def reset(self, num_agents=None, space_length=None, space_shape=None):
        if self.curriculum:
            rand = random.choice(self.env_set)
            self.num_agents = rand[0]
            self.space_size = (rand[1], rand[1],rand[1])
        elif num_agents is not None and space_length is not None:
            self.num_agents = num_agents
            self.space_size = (space_length, space_length, space_length)
        if space_shape:
            self.num_agents = num_agents
            self.space_size = (space_shape[0], space_shape[1], space_shape[2])

        self.space = np.random.choice(
            2, self.space_size, p=[1 - self.obstacle_density, self.obstacle_density]
        ).astype(int)

        partition_list = self._space_partition()
        while len(partition_list) == 0:
            self.space = np.random.choice(
                2, self.space_size, p=[1 - self.obstacle_density, self.obstacle_density]
            ).astype(int)
            partition_list = self._space_partition()

        self.agents_pos = np.empty((self.num_agents, 3), dtype=int)
        self.goals_pos = np.empty((self.num_agents, 3), dtype=int)
        pos_num = sum([len(partition) for partition in partition_list])

        for i in range(self.num_agents):

            pos_idx = random.randint(0, pos_num - 1)
            partition_idx = 0
            for partition in partition_list:
                if pos_idx >= len(partition):
                    pos_idx -= len(partition)
                    partition_idx += 1
                else:
                    break

            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.agents_pos[i] = np.asarray(pos, dtype=int)

            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.goals_pos[i] = np.asarray(pos, dtype=int)

            partition_list = [
                partition for partition in partition_list if len(partition) >= 2
            ]
            pos_num = sum([len(partition) for partition in partition_list])

        self.steps = 0
        self.last_actions = np.zeros((self.num_agents, config.action_dim), dtype=bool)
        return self.observe()

    def step(self, actions: List[int]):
        """
            (0, 0, 1):  0,  # up
            (0, 0, -1): 1,  # down
            (-1, 0, 0): 2,  # left
            (1, 0, 0):  3,  # right
            (0, 1, 0):  4,  # front
            (0, -1, 0): 5,  # back
            (0, 0, 0):  6,  # stay
        """
        assert (
            len(actions) == self.num_agents
        ), "only {} actions as input while {} agents in environment".format(
            len(actions), self.num_agents
        )
        
        checking_list = [i for i in range(self.num_agents)]
        rewards = []
        next_pos = np.copy(self.agents_pos)
        for agent_id in checking_list.copy():
            if actions[agent_id] == 6:
                # unmoving
                if np.array_equal(self.agents_pos[agent_id], self.goals_pos[agent_id]):
                    rewards.append(self.reward_fn["stay_on_goal"])
                else:
                    rewards.append(self.reward_fn["stay_off_goal"])
                checking_list.remove(agent_id)
            else:
                # move
                next_pos[agent_id] += ACTION_LIST[actions[agent_id]]
                rewards.append(self.reward_fn["move"])

        # first round check, these two conflicts have the highest priority
        for agent_id in checking_list.copy():
            if (
                np.any(next_pos[agent_id] < 0)
                or next_pos[agent_id][0] >= self.space_size[0]
                or next_pos[agent_id][1] >= self.space_size[1]
                or next_pos[agent_id][2] >= self.space_size[2]
            ):
                # agent out of map range
                rewards[agent_id] = self.reward_fn["collision"]
                next_pos[agent_id] = self.agents_pos[agent_id]
                print(f"Environment3D: move outbound")
                checking_list.remove(agent_id)

            elif self.space[tuple(next_pos[agent_id])] == 1:
                # collide obstacle
                rewards[agent_id] = self.reward_fn["collision"]
                print(f"Environment3D: move to obstacle")
                next_pos[agent_id] = self.agents_pos[agent_id]
                checking_list.remove(agent_id)

        # second round check, agent swapping conflict
        no_conflict = False
        while not no_conflict:
            no_conflict = True
            for agent_id in checking_list:
                target_agent_id = np.where(
                    np.all(next_pos[agent_id] == self.agents_pos, axis=1)
                )[0]
                if target_agent_id:
                    target_agent_id = target_agent_id.item()
                    if np.array_equal(
                        next_pos[target_agent_id], self.agents_pos[agent_id]
                    ):
                        assert (
                            target_agent_id in checking_list
                        ), "target_agent_id should be in checking list"
                        next_pos[agent_id] = self.agents_pos[agent_id]
                        rewards[agent_id] = self.reward_fn["collision"]
                        next_pos[target_agent_id] = self.agents_pos[target_agent_id]
                        rewards[target_agent_id] = self.reward_fn["collision"]
                        print(f"Environment3D: swap {agent_id} {target_agent_id}")
                        checking_list.remove(agent_id)
                        checking_list.remove(target_agent_id)
                        no_conflict = False
                        break
        # third round check, agent collision conflict
        no_conflict = False
        while not no_conflict:
            no_conflict = True
            for agent_id in checking_list:
                collide_agent_id = np.where(
                    np.all(next_pos == next_pos[agent_id], axis=1)
                )[0].tolist()
                if len(collide_agent_id) > 1:
                    # if all agents in collide agent are in checking list
                    all_in_checking = True
                    for id in collide_agent_id.copy():
                        if id not in checking_list:
                            all_in_checking = False
                            collide_agent_id.remove(id)
                    if all_in_checking:
                        collide_agent_pos = next_pos[collide_agent_id].tolist()
                        for pos, id in zip(collide_agent_pos, collide_agent_id):
                            pos.append(id)
                        collide_agent_pos.sort(
                            key=lambda x: x[0] * self.space_size[0]
                            + x[1] ** self.space_size[1]
                            + x[2]
                        )
                        collide_agent_id.remove(collide_agent_pos[0][3])
                    next_pos[collide_agent_id] = self.agents_pos[collide_agent_id]
                    for id in collide_agent_id:
                        rewards[id] = self.reward_fn["collision"]
                        print(f"Environment3D: colide {agent_id} {collide_agent_id}")
                    for id in collide_agent_id:
                        checking_list.remove(id)

                    no_conflict = False
                    break
        self.agents_pos = np.copy(next_pos)
        self.steps += 1
        # check done
        if np.array_equal(self.agents_pos, self.goals_pos):
            done = True
            rewards = [self.reward_fn["finish"] for _ in range(self.num_agents)]
        else:
            done = False

        info = {"step": self.steps - 1}
        # make sure no overlapping agents
        assert np.unique(self.agents_pos, axis=0).shape[0] == self.num_agents
        # update last actions
        self.last_actions = np.zeros((self.num_agents, len(ACTION_LIST)), dtype=bool)
        self.last_actions[np.arange(self.num_agents), np.array(actions)] = 1
        return self.observe(), rewards, done, info

    def observe(self):
        obs = np.zeros(
            (
                self.num_agents,
                3,
                2 * self.obs_radius + 1,
                2 * self.obs_radius + 1,
                2 * self.obs_radius + 1,
            ),
            dtype=bool,
        )
        obstacle_map = np.pad(
            self.space, ((self.obs_radius,), (self.obs_radius,), (self.obs_radius,)), "constant", constant_values=0
        )

        agent_map = np.zeros((self.space_size), dtype=bool)
        agent_map[
            self.agents_pos[:, 0], self.agents_pos[:, 1], self.agents_pos[:, 2]
        ] = 1
        agent_map = np.pad(agent_map, self.obs_radius, "constant", constant_values=0)

        for i, agent_pos in enumerate(self.agents_pos):
            x, y, z = agent_pos
            obs[i, 0] = agent_map[
                x : x + 2 * self.obs_radius + 1,
                y : y + 2 * self.obs_radius + 1,
                z : z + 2 * self.obs_radius + 1,
            ]
            # set center as 0
            obs[i, 0, self.obs_radius, self.obs_radius, self.obs_radius] = 0
            obs[i, 1] = obstacle_map[
                x : x + 2 * self.obs_radius + 1,
                y : y + 2 * self.obs_radius + 1,
                z : z + 2 * self.obs_radius + 1,
            ]
            obs[i, 2] = self._get_projective_obs_3d(agent_pos, self.goals_pos[i])

        return obs, np.copy(self.last_actions), np.copy(self.agents_pos)

    def _get_projective_obs_3d(self, agent_pos, target):
        agent_x = agent_pos[0] + self.obs_radius
        agent_y = agent_pos[1] + self.obs_radius
        agent_z = agent_pos[2] + self.obs_radius

        targets_obs = np.zeros((2 * self.obs_radius + 1, 2 * self.obs_radius + 1, 2 * self.obs_radius + 1))

        target_x, target_y, target_z = target[0] + self.obs_radius, target[1] + self.obs_radius, target[2] + self.obs_radius
        # print(f"target_pos expanded: [{target_x},{target_y},{target_z}]")
        # 判断目标是否在FOV内
        if (
            agent_x - self.obs_radius <= target_x <= agent_x + self.obs_radius
        ) and (agent_y - self.obs_radius <= target_y <= agent_y + self.obs_radius) and (
            agent_z - self.obs_radius <= target_z <= agent_z + self.obs_radius
        ):
            # 计算目标在FOV数组中的位置
            rel_x = target_x - (agent_x - self.obs_radius)
            rel_y = target_y - (agent_y - self.obs_radius)
            rel_z = target_z - (agent_z - self.obs_radius)
            targets_obs[rel_x, rel_y, rel_z] = 1
        else:
            # 计算目标点在FOV边界上的投影位置
            offset_x = target_x - agent_x
            offset_y = target_y - agent_y
            offset_z = target_z - agent_z
            border_x = 0 if offset_x < 0 else 2 * self.obs_radius
            border_y = 0 if offset_y < 0 else 2 * self.obs_radius
            border_z = 0 if offset_z < 0 else 2 * self.obs_radius

            if abs(offset_x) >= abs(offset_y) and abs(offset_x) >= abs(offset_z):
                border_y = (
                    int((offset_y / abs(offset_x)) * self.obs_radius)
                    + self.obs_radius
                )
                border_z = (
                    int((offset_z / abs(offset_x)) * self.obs_radius)
                    + self.obs_radius
                )
            elif abs(offset_y) >= abs(offset_x) and abs(offset_y) >= abs(offset_z):
                border_x = (
                    int((offset_x / abs(offset_y)) * self.obs_radius)
                    + self.obs_radius
                )
                border_z = (
                    int((offset_z / abs(offset_y)) * self.obs_radius)
                    + self.obs_radius
                )
            else:
                border_x = (
                    int((offset_x / abs(offset_z)) * self.obs_radius)
                    + self.obs_radius
                )
                border_y = (
                    int((offset_y / abs(offset_z)) * self.obs_radius)
                    + self.obs_radius
                )

            # 将距离编码到FOV的边界上
            targets_obs[border_x, border_y, border_z] = 1

        return targets_obs

    def _space_partition(self):
        # print(f"self.space.shape: {self.space.shape}")
        empty_list = np.argwhere(self.space == 0).tolist()
        empty_pos = set([tuple(pos) for pos in empty_list])

        if not empty_pos:
            raise RuntimeError("no empty position")

        partition_list = list()
        while empty_pos:

            start_pos = empty_pos.pop()

            open_list = list()
            open_list.append(start_pos)
            close_list = list()

            while open_list:
                current_pos = open_list.pop(0)
                close_list.append(current_pos)

                for action in ACTION_LIST[:-1]:  # Exclude the 'stay' action
                    # print(f"current_pos:{current_pos}, action:{action}")
                    neighbor = tuple(np.array(current_pos) + action)
                    if neighbor in empty_pos:
                        empty_pos.remove(neighbor)
                        open_list.append(neighbor)

            if len(close_list) >= 2:
                partition_list.append(close_list)

        return partition_list
