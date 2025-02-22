
import ray
import torch
import numpy as np
import random
import math
from global_buffer import GlobalBuffer,EpisodeData
from learner import Learner
from model import Network
from environment3d import Environment3D
import config as config

class LocalBuffer:
    __slots__ = (
        "actor_id",
        "map_len",
        "num_agents",
        "obs_buf",
        "act_buf",
        "rew_buf",
        "hidden_buf",
        "forward_steps",
        "relative_pos_buf",
        "q_buf",
        "capacity",
        "size",
        "done",
        "burn_in_steps",
        "chunk_capacity",
        "last_act_buf",
        "comm_mask_buf",
    )

    def __init__(
        self,
        actor_id: int,
        num_agents: int,
        map_len: int,
        init_obs: np.ndarray,
        forward_steps=config.forward_steps,
        capacity: int = config.max_episode_length,
        burn_in_steps=config.burn_in_steps,
        obs_shape=config.obs_shape,
        hidden_dim=config.hidden_dim,
        action_dim=config.action_dim,
    ):
        """
        buffer for each episode
        """
        self.actor_id = actor_id
        self.num_agents = num_agents
        self.map_len = map_len

        self.burn_in_steps = burn_in_steps
        self.forward_steps = forward_steps

        self.chunk_capacity = config.chunk_capacity

        self.obs_buf = np.zeros(
            (burn_in_steps + capacity + 1, num_agents, *obs_shape), dtype=bool
        )
        self.last_act_buf = np.zeros(
            (burn_in_steps + capacity + 1, num_agents, config.action_dim), dtype=bool
        )
        self.act_buf = np.zeros((capacity), dtype=np.uint8)
        self.rew_buf = np.zeros((capacity + forward_steps - 1), dtype=np.float16)
        self.hidden_buf = np.zeros(
            (burn_in_steps + capacity + 1, num_agents, hidden_dim), dtype=np.float16
        )
        self.relative_pos_buf = np.zeros(
            (burn_in_steps + capacity + 1, num_agents, num_agents, 3), dtype=np.int8
        )
        self.comm_mask_buf = np.zeros(
            (burn_in_steps + capacity + 1, num_agents, num_agents), dtype=bool
        )
        self.q_buf = np.zeros((capacity + 1, action_dim), dtype=np.float32)

        self.capacity = capacity
        self.size = 0

        self.obs_buf[: burn_in_steps + 1] = init_obs

    def add(
        self,
        q_val,
        action: int,
        last_act,
        reward: float,
        next_obs,
        hidden,
        relative_pos,
        comm_mask,
    ):
        assert self.size < self.capacity

        self.act_buf[self.size] = action
        self.rew_buf[self.size] = reward
        self.obs_buf[self.burn_in_steps + self.size + 1] = next_obs
        self.last_act_buf[self.burn_in_steps + self.size + 1] = last_act
        self.q_buf[self.size] = q_val
        self.hidden_buf[self.burn_in_steps + self.size + 1] = hidden
        self.relative_pos_buf[self.burn_in_steps + self.size] = relative_pos
        self.comm_mask_buf[self.burn_in_steps + self.size] = comm_mask

        self.size += 1

    def finish(self, last_q_val=None, last_relative_pos=None, last_comm_mask=None):
        forward_steps = min(self.size, self.forward_steps)
        cumulated_gamma = [
            config.gamma**forward_steps for _ in range(self.size - forward_steps)
        ]

        # last q value is None if done
        if last_q_val is None:
            done = True
            cumulated_gamma.extend([0 for _ in range(forward_steps)])

        else:
            done = False
            self.q_buf[self.size] = last_q_val
            self.relative_pos_buf[self.burn_in_steps + self.size] = last_relative_pos
            self.comm_mask_buf[self.burn_in_steps + self.size] = last_comm_mask
            cumulated_gamma.extend(
                [config.gamma**i for i in reversed(range(1, forward_steps + 1))]
            )

        num_chunks = math.ceil(self.size / config.chunk_capacity) # 切分成小块，便于global buffer维护

        cumulated_gamma = np.array(cumulated_gamma, dtype=np.float16)
        self.obs_buf = self.obs_buf[: self.burn_in_steps + self.size + 1]
        self.last_act_buf = self.last_act_buf[: self.burn_in_steps + self.size + 1]
        self.act_buf = self.act_buf[: self.size]
        self.rew_buf = self.rew_buf[: self.size + self.forward_steps - 1]
        self.hidden_buf = self.hidden_buf[: self.size]
        self.relative_pos_buf = self.relative_pos_buf[
            : self.burn_in_steps + self.size + 1
        ]
        self.comm_mask_buf = self.comm_mask_buf[: self.burn_in_steps + self.size + 1]

        self.rew_buf = np.convolve(
            self.rew_buf,
            [
                config.gamma ** (self.forward_steps - 1 - i)
                for i in range(self.forward_steps)
            ],
            "valid",
        ).astype(np.float16)

        # caculate td errors for prioritized experience replay
        # max_q评估当前状态下所有可能动作的最大 Q 值
        max_q = np.max(self.q_buf[forward_steps : self.size + 1], axis=1)
        # 填充 max_q 的最后一个元素,简化计算
        max_q = np.concatenate(
            (max_q, np.array([max_q[-1] for _ in range(forward_steps - 1)]))
        )
        # 当前状态下实际选择的动作对应的 Q 值
        target_q = self.q_buf[np.arange(self.size), self.act_buf]
        td_errors = np.zeros(num_chunks * self.chunk_capacity, dtype=np.float32)
        #  self.rew_buf + max_q * cumulated_gamma 当前状态及未来状态的预期总回报
        #  target_q 当前策略下实际选择的动作的估计 Q 值
        #  td_errors目的是为了缩小当前估计与目标值之间的差距
        td_errors[: self.size] = np.abs(
            self.rew_buf + max_q * cumulated_gamma - target_q
        ).clip(1e-6)
        sizes = np.array(
            [
                min(self.chunk_capacity, self.size - i * self.chunk_capacity)
                for i in range(num_chunks)
            ],
            dtype=np.uint8,
        )

        data = EpisodeData(
            self.actor_id,
            self.num_agents,
            self.map_len,
            self.obs_buf,
            self.last_act_buf,
            self.act_buf,
            self.rew_buf,
            self.hidden_buf,
            self.relative_pos_buf,
            self.comm_mask_buf,
            cumulated_gamma,
            td_errors,
            sizes,
            done,
        )

        return data


@ray.remote(num_cpus=1)
class Actor:
    def __init__(
        self, worker_id: int, epsilon: float, learner: Learner, buffer: GlobalBuffer
    ):
        print(f"Actor[{worker_id}], epsilon:{epsilon}")
        self.id = worker_id
        self.model = Network()
        if config.load_model:
            checkpoint = torch.load(config.load_model, map_location="cpu")
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.env = Environment3D(curriculum=True)
        self.epsilon = epsilon
        self.learner = learner
        self.global_buffer = buffer
        self.max_episode_length = config.max_episode_length
        self.counter = 0

    def run(self):
        done = False
        obs, last_act, pos, local_buffer = self._reset()

        while True:
            # print(obs)
            # sample action
            actions, q_val, hidden, relative_pos, comm_mask = self.model.step(
                torch.from_numpy(obs.astype(np.float32)),
                torch.from_numpy(last_act.astype(np.float32)),
                torch.from_numpy(pos.astype(int)),
            )

            if random.random() < self.epsilon:
                # Note: only one agent do random action in order to keep the environment stable
                actions[0] = np.random.randint(0, config.action_dim)

            # take action in env
            (next_obs, last_act, next_pos), rewards, done, _ = self.env.step(actions)
            # print(f"rewards: {rewards}")

            # return data and update observation
            local_buffer.add(
                q_val[0],
                actions[0],
                last_act,
                rewards[0],
                next_obs,
                hidden,
                relative_pos,
                comm_mask,
            )

            if done == False and self.env.steps < self.max_episode_length:
                obs, pos = next_obs, next_pos
            else:
                # finish and send buffer
                if done:
                    data = local_buffer.finish()
                else:
                    _, q_val, _, relative_pos, comm_mask = self.model.step(
                        torch.from_numpy(next_obs.astype(np.float32)),
                        torch.from_numpy(last_act.astype(np.float32)),
                        torch.from_numpy(next_pos.astype(int)),
                    )
                    data = local_buffer.finish(q_val[0], relative_pos, comm_mask)

                self.global_buffer.add.remote(data)
                done = False
                obs, last_act, pos, local_buffer = self._reset()

            self.counter += 1
            if self.counter == config.actor_update_steps:
                self._update_weights()
                self.counter = 0

    def _update_weights(self):
        """load weights from learner"""
        # update network parameters
        weights_id = ray.get(self.learner.get_weights.remote())
        weights = ray.get(weights_id)
        self.model.load_state_dict(weights)
        # update environment settings set (number of agents and map size)
        new_env_settings_set = ray.get(self.global_buffer.get_env_settings.remote())
        self.env.update_env_settings_set(ray.get(new_env_settings_set))

    def _reset(self):
        self.model.reset()
        obs, last_act, pos = self.env.reset()
        local_buffer = LocalBuffer(
            self.id, self.env.num_agents, self.env.space_size[0], obs
        )
        return obs, last_act, pos, local_buffer
