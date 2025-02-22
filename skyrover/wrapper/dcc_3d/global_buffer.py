from . import config as config
import numpy as np
import ray
import threading
import time
import torch
from typing import Tuple
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
import os

@dataclass
class EpisodeData:
    __slots__ = (
        "actor_id",
        "num_agents",
        "map_len",
        "obs",
        "last_act",
        "actions",
        "rewards",
        "hiddens",
        "relative_pos",
        "comm_mask",
        "gammas",
        "td_errors",
        "sizes",
        "done",
    )
    actor_id: int                    # Actor 的 ID，用于标识生成该数据的具体 Actor
    num_agents: int                  # 当前回合中的智能体（agent）数量
    map_len: int                     # 地图长度，通常用于描述当前环境的规模
    obs: np.ndarray                  # 观测值（observation）的数组，每个时间步的观测数据
    last_act: np.ndarray             # 最后执行的动作的数组，记录了上一时间步的动作，用于 RNN 等需要考虑历史信息的场景
    actions: np.ndarray              # 动作序列，记录了 agent 在整个回合中执行的所有动作
    rewards: np.ndarray              # 奖励值的数组，记录了每个时间步对应的奖励，用于强化学习中的回报计算
    hiddens: np.ndarray              # 隐藏状态的数组，通常用于存储 RNN 中的隐藏状态，用于保存历史信息
    relative_pos: np.ndarray         # 相对位置的数组，记录 agent 和其他实体之间的相对位置信息
    comm_mask: np.ndarray            # 通信掩码的数组，表示 agent 之间是否能够通信
    gammas: np.ndarray               # 折扣因子的数组，用于对未来的回报进行折扣计算，表示从当前时间步开始的折扣因子
    td_errors: np.ndarray            # TD-误差数组，存储每个样本的 Temporal Difference 误差，表示当前策略与目标值之间的差距
    sizes: np.ndarray                # 每个 chunk 的大小数组，表示将整个回合分块后的每个块中的数据数量，用于经验存储的管理
    done: bool                       # 回合结束标志，True 表示回合已结束，False 表示回合未结束

class SumTree:
    """used for prioritized experience replay"""

    def __init__(self, capacity: int):
        layer = 1
        while 2 ** (layer - 1) < capacity:
            layer += 1
        assert 2 ** (layer - 1) == capacity, "capacity only allow n**2 size"
        self.layer = layer
        self.tree = np.zeros(2**layer - 1, dtype=np.float64)
        self.capacity = capacity
        self.size = 0

    def sum(self):
        assert (
            np.sum(self.tree[-self.capacity :]) - self.tree[0] < 0.1
        ), "sum is {} but root is {}".format(
            np.sum(self.tree[-self.capacity :]), self.tree[0]
        )
        return self.tree[0]

    def __getitem__(self, idx: int):
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity - 1 + idx] #叶子结点从self.capacity - 1开始

    def batch_sample(self, batch_size: int):
        # 获取树根节点的值，即所有叶节点的优先级和（总和）
        p_sum = self.tree[0]
        
        # 将总和 p_sum 分成 batch_size 份，得到每个采样间隔
        interval = p_sum / batch_size

        # 生成 prefixsums 数组，其中每个元素是每个间隔的起始点，加上一个在 [0, interval] 范围内的随机数
        # 目的是保证采样的均匀性并引入随机性
        prefixsums = np.arange(0, p_sum, interval, dtype=np.float64) + np.random.uniform(
            0, interval, batch_size
        )

        # 初始化采样索引数组，初始位置从根节点开始（索引为 0）
        idxes = np.zeros(batch_size, dtype=int)
        
        # 遍历每一层，直到叶节点，逐层向下选择节点，层数为 self.layer - 1
        for _ in range(self.layer - 1):
            # 获取每个 idxes 对应的左子节点的值
            nodes = self.tree[idxes * 2 + 1]
            
            # 如果 prefixsum 小于当前节点的左子节点值，则选择左子节点（索引为 idxes * 2 + 1）
            # 否则，选择右子节点（索引为 idxes * 2 + 2）
            idxes = np.where(prefixsums < nodes, idxes * 2 + 1, idxes * 2 + 2)
            
            # 如果选择了右子节点，则更新 prefixsums，减去左子节点的值
            # 这是因为在向右走时，已经跳过了左子节点，因此要从前缀和中减去对应的值
            prefixsums = np.where(
                idxes % 2 == 0, prefixsums - self.tree[idxes - 1], prefixsums
            )

        # 最终找到的 idxes 数组是叶节点的位置（在数组中）
        # 获取对应叶节点的优先级值
        priorities = self.tree[idxes]
        
        # 将 idxes 从树的索引转换为叶节点在树的叶节点部分的索引
        # 叶节点从 self.capacity - 1 开始存储，因此要减去 self.capacity - 1
        idxes -= self.capacity - 1

        # 确保所有的优先级大于 0，否则会引发错误
        assert np.all(priorities > 0), "idx: {}, priority: {}".format(idxes, priorities)
        # 确保所有索引在 [0, capacity) 范围内
        assert np.all(idxes >= 0) and np.all(idxes < self.capacity)

        # 返回采样的索引以及它们对应的优先级
        return idxes, priorities


    def batch_update(self, idxes: np.ndarray, priorities: np.ndarray):
        assert idxes.shape[0] == priorities.shape[0]
        idxes += self.capacity - 1
        self.tree[idxes] = priorities

        for _ in range(self.layer - 1):
            idxes = (idxes - 1) // 2
            idxes = np.unique(idxes)
            self.tree[idxes] = self.tree[2 * idxes + 1] + self.tree[2 * idxes + 2]

        # check
        assert (
            np.sum(self.tree[-self.capacity :]) - self.tree[0] < 0.1
        ), "sum is {} but root is {}".format(
            np.sum(self.tree[-self.capacity :]), self.tree[0]
        )



@ray.remote(num_cpus=1)
class GlobalBuffer:
    def __init__(
        self,
        log_dir,
        buffer_capacity=config.buffer_capacity,
        init_env_settings=tuple(config.init_env_settings),
        alpha=config.prioritized_replay_alpha,
        beta=config.prioritized_replay_beta,
        chunk_capacity=config.chunk_capacity,
    ):

        self.capacity = buffer_capacity
        self.chunk_capacity = chunk_capacity
        self.num_chunks = buffer_capacity // chunk_capacity
        self.ptr = 0 #chunk指针

        # prioritized experience replay
        self.priority_tree = SumTree(buffer_capacity)
        self.alpha = alpha
        self.beta = beta

        self.counter = 0
        self.batched_data = []
        self.stat_dict = {init_env_settings: []}
        self.lock = threading.Lock()
        self.env_settings_set = ray.put([init_env_settings])

        self.obs_buf = [None] * self.num_chunks
        self.last_act_buf = [None] * self.num_chunks
        self.act_buf = np.zeros((buffer_capacity), dtype=np.uint8)
        self.rew_buf = np.zeros((buffer_capacity), dtype=np.float16)
        self.hid_buf = [None] * self.num_chunks
        self.size_buf = np.zeros(self.num_chunks, dtype=np.uint8)
        self.relative_pos_buf = [None] * self.num_chunks
        self.comm_mask_buf = [None] * self.num_chunks
        self.gamma_buf = np.zeros((self.capacity), dtype=np.float16)
        self.num_agents_buf = np.zeros((self.num_chunks), dtype=np.uint8)
        self.learner_epoch = 0
        self.writer = SummaryWriter(log_dir)

    def __len__(self):
        return np.sum(self.size_buf)

    def run(self):
        self.background_thread = threading.Thread(target=self._prepare_data, daemon=True)
        self.background_thread.start()

    def _prepare_data(self):
        while True:
            if len(self.batched_data) <= 4:
                data = self.sample_batch(config.batch_size)
                data_id = ray.put(data)
                self.batched_data.append(data_id)

            else:
                time.sleep(0.1)

    def sample_batch(self, *args, **kwargs):  # NOTE: NEW, for avoiding bugs
        return self._sample_batch(*args, **kwargs)

    def get_batched_data(self, learner_epoch):
        """
        get one batch of data, called by learner.
        """
        self.learner_epoch = learner_epoch
        if len(self.batched_data) == 0:
            print("no prepared data")
            data = self._sample_batch(config.batch_size)
            data_id = ray.put(data)
            return data_id
        else:
            return self.batched_data.pop(0)

    def add(self, data: EpisodeData):
        """
        Add one episode data into replay buffer, called by actor if actor finished one episode.

        data: actor_id 0, num_agents 1, map_len 2, obs_buf 3, act_buf 4, rew_buf 5,
                hid_buf 6, comm_mask_buf 8, gamma 9, td_errors 10, sizes 11, done 12
        """
        if data.actor_id >= 18:  # eps-greedy < 0.01 
        # if True:
            stat_key = (data.num_agents, data.map_len)
            if stat_key in self.stat_dict:
                self.stat_dict[stat_key].append(data.done)
                if len(self.stat_dict[stat_key]) == config.cl_history_size + 1:
                    self.stat_dict[stat_key].pop(0)

        with self.lock:
            for i, size in enumerate(data.sizes):
                size = int(size)
                idxes = np.arange(
                    self.ptr * self.chunk_capacity, (self.ptr + 1) * self.chunk_capacity
                )
                start_idx = self.ptr * self.chunk_capacity
                # update buffer size
                self.counter += size

                self.priority_tree.batch_update(
                    idxes,
                    data.td_errors[
                        i * self.chunk_capacity : (i + 1) * self.chunk_capacity
                    ]
                    ** self.alpha,
                )

                self.obs_buf[self.ptr] = np.copy(
                    data.obs[
                        i * self.chunk_capacity : (i + 1) * self.chunk_capacity
                        + config.burn_in_steps
                        + config.forward_steps
                    ]
                )
                self.last_act_buf[self.ptr] = np.copy(
                    data.last_act[
                        i * self.chunk_capacity : (i + 1) * self.chunk_capacity
                        + config.burn_in_steps
                        + config.forward_steps
                    ]
                )
                # print(f"self.act_buf.shape: {self.act_buf.shape}")
                # print(f"self.act_buf: {self.act_buf}")
                # print(f"start_idx: {start_idx}")
                # print(f"size: {size}")
                # print(f"type(start_idx):{type(start_idx)}")
                # print(f"type(size):{type(size)}")
                # print(f"type(self.chunk_capacity):{type(self.chunk_capacity)}")
                # print(f"type(i):{type(i)}")
                self.act_buf[start_idx : start_idx + size] = data.actions[
                    i * self.chunk_capacity : i * self.chunk_capacity + size
                ]
                self.rew_buf[start_idx : start_idx + size] = data.rewards[
                    i * self.chunk_capacity : i * self.chunk_capacity + size
                ]
                self.hid_buf[self.ptr] = np.copy(
                    data.hiddens[
                        i * self.chunk_capacity : i * self.chunk_capacity
                        + size
                        + config.forward_steps
                    ]
                )
                self.size_buf[self.ptr] = size
                self.relative_pos_buf[self.ptr] = np.copy(
                    data.relative_pos[
                        i * self.chunk_capacity : (i + 1) * self.chunk_capacity
                        + config.burn_in_steps
                        + config.forward_steps
                    ]
                )
                self.comm_mask_buf[self.ptr] = np.copy(
                    data.comm_mask[
                        i * self.chunk_capacity : (i + 1) * self.chunk_capacity
                        + config.burn_in_steps
                        + config.forward_steps
                    ]
                )
                self.gamma_buf[start_idx : start_idx + size] = data.gammas[
                    i * self.chunk_capacity : i * self.chunk_capacity + size
                ]
                self.num_agents_buf[self.ptr] = data.num_agents

                # 环形缓冲
                self.ptr = (self.ptr + 1) % self.num_chunks

            del data

    def _sample_batch(self, batch_size: int) -> Tuple:

        b_obs, b_last_act, b_steps, b_relative_pos, b_comm_mask = [], [], [], [], []
        b_hidden = []
        idxes, priorities = [], []

        with self.lock:

            idxes, priorities = self.priority_tree.batch_sample(batch_size)
            global_idxes = idxes // self.chunk_capacity
            local_idxes = idxes % self.chunk_capacity
            max_num_agents = np.max(self.num_agents_buf[global_idxes])

            for global_idx, local_idx in zip(global_idxes.tolist(), local_idxes.tolist()):
                # print(f"local_idx: {local_idx}")
                # print(f"global_idx: {global_idx}")
                # print(f"global_idx: {global_idx}")
                # print(f"self.size_buf[{global_idx}],: {self.size_buf[global_idx]}")

                assert (
                    local_idx < self.size_buf[global_idx]
                ), "index is {} but size is {}".format(
                    local_idx, self.size_buf[global_idx]
                )

                steps = min(
                    config.forward_steps, self.size_buf[global_idx].item() - local_idx
                )

                relative_pos = self.relative_pos_buf[global_idx][
                    local_idx : local_idx + config.burn_in_steps + steps + 1
                ]
                comm_mask = self.comm_mask_buf[global_idx][
                    local_idx : local_idx + config.burn_in_steps + steps + 1
                ]
                obs = self.obs_buf[global_idx][
                    local_idx : local_idx + config.burn_in_steps + steps + 1
                ]
                last_act = self.last_act_buf[global_idx][
                    local_idx : local_idx + config.burn_in_steps + steps + 1
                ]
                hidden = self.hid_buf[global_idx][local_idx]

                if steps < config.forward_steps:
                    pad_len = config.forward_steps - steps
                    obs = np.pad(obs, ((0, pad_len), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)))
                    last_act = np.pad(last_act, ((0, pad_len), (0, 0), (0, 0)))
                    relative_pos = np.pad(
                        relative_pos, ((0, pad_len), (0, 0), (0, 0), (0, 0))
                    )
                    comm_mask = np.pad(comm_mask, ((0, pad_len), (0, 0), (0, 0)))

                if self.num_agents_buf[global_idx] < max_num_agents:
                    pad_len = max_num_agents - self.num_agents_buf[global_idx].item()
                    obs = np.pad(obs, ((0, 0), (0, pad_len), (0, 0), (0, 0), (0, 0), (0, 0)))
                    last_act = np.pad(last_act, ((0, 0), (0, pad_len), (0, 0)))
                    relative_pos = np.pad(
                        relative_pos, ((0, 0), (0, pad_len), (0, pad_len), (0, 0))
                    )
                    comm_mask = np.pad(comm_mask, ((0, 0), (0, pad_len), (0, pad_len)))
                    hidden = np.pad(hidden, ((0, pad_len), (0, 0)))

                b_obs.append(obs)
                b_last_act.append(last_act)
                b_steps.append(steps)
                b_relative_pos.append(relative_pos)
                b_comm_mask.append(comm_mask)
                b_hidden.append(hidden)

            # importance sampling weight
            min_p = np.min(priorities)
            weights = np.power(priorities / min_p, -self.beta)

            b_action = self.act_buf[idxes]
            b_reward = self.rew_buf[idxes]
            b_gamma = self.gamma_buf[idxes]

            data = (
                torch.from_numpy(np.stack(b_obs)).transpose(1, 0).contiguous(),
                torch.from_numpy(np.stack(b_last_act)).transpose(1, 0).contiguous(),
                torch.from_numpy(b_action).unsqueeze(1),
                torch.from_numpy(b_reward).unsqueeze(1),
                torch.from_numpy(b_gamma).unsqueeze(1),
                torch.ByteTensor(b_steps),
                torch.from_numpy(np.concatenate(b_hidden, axis=0)),
                torch.from_numpy(np.stack(b_relative_pos)),
                torch.from_numpy(np.stack(b_comm_mask)),
                idxes,
                torch.from_numpy(weights.astype(np.float16)).unsqueeze(1),
                self.ptr,
            )

            return data

    def update_priorities(self, idxes: np.ndarray, priorities: np.ndarray, old_ptr: int):
        """Update priorities of sampled transitions"""
        with self.lock:

            # discard the indices that already been discarded in replay buffer during training
            if self.ptr > old_ptr:
                # range from [old_ptr, self.ptr)
                mask = (idxes < old_ptr * self.chunk_capacity) | (
                    idxes >= self.ptr * self.chunk_capacity
                )
                idxes = idxes[mask]
                priorities = priorities[mask]
            elif self.ptr < old_ptr:
                # range from [0, self.ptr) & [old_ptr, self,capacity)
                mask = (idxes < old_ptr * self.chunk_capacity) & (
                    idxes >= self.ptr * self.chunk_capacity
                )
                idxes = idxes[mask]
                priorities = priorities[mask]

            self.priority_tree.batch_update(
                np.copy(idxes), np.copy(priorities) ** self.alpha
            )

    def stats(self, interval: int):
        """
        Print log
        """
        print("buffer update speed: {}/s".format(self.counter / interval))
        print("buffer size: {}".format(np.sum(self.size_buf)))

        print("  ", end="")
        for i in range(config.init_env_settings[1], config.max_map_length + 1, 5):
            print("   {:2d}   ".format(i), end="")
        print()

        for num_agents in range(config.init_env_settings[0], config.max_num_agents + 1):
            print("{:2d}".format(num_agents), end="")
            for map_len in range(
                config.init_env_settings[1], config.max_map_length + 1, 5
            ):
                if (num_agents, map_len) in self.stat_dict:
                    print(
                        "{:4d}/{:<3d}".format(
                            sum(self.stat_dict[(num_agents, map_len)]),
                            len(self.stat_dict[(num_agents, map_len)]),
                        ),
                        end="",
                    )
                    self.writer.add_scalar(f'statistics/num_agents_{num_agents}_map_len_{map_len}_sr', 
                                           sum(self.stat_dict[(num_agents, map_len)]), 
                                           self.learner_epoch)
                else:
                    print("   N/A  ", end="")
            print()

        for key, val in self.stat_dict.copy().items():
            if (
                len(val) == config.cl_history_size
                and sum(val) >= config.cl_history_size * config.pass_rate
            ):
                # add number of agents
                add_agent_key = (key[0] + 1, key[1])
                if (
                    add_agent_key[0] <= config.max_num_agents
                    and add_agent_key not in self.stat_dict
                ):
                    self.stat_dict[add_agent_key] = []

                if key[1] < config.max_map_length:
                    add_map_key = (key[0], key[1] + 5)
                    if add_map_key not in self.stat_dict:
                        self.stat_dict[add_map_key] = []

        self.env_settings_set = ray.put(list(self.stat_dict.keys()))

        self.counter = 0

    def ready(self):
        if len(self) >= config.learning_starts:
            return True
        else:
            return False

    def get_env_settings(self):
        return self.env_settings_set
