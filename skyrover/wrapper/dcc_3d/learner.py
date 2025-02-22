import ray
import torch
from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
import torch.nn.functional as F
from model import Network
from global_buffer import GlobalBuffer
import config as config

import subprocess
import os
import threading
from copy import deepcopy

import json
from torch.utils.tensorboard import SummaryWriter

def is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False
    

@ray.remote
def execute_command(command):
    try:
        subprocess.run(command)
    except subprocess.CalledProcessError:
        pass


@ray.remote(num_cpus=1, num_gpus=1)
class Learner:
    def __init__(self, buffer: GlobalBuffer,log_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Network()
        if config.load_model:
            checkpoint = torch.load(config.load_model, map_location=self.device)
            self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.tar_model = deepcopy(self.model)
        self.optimizer = Adam(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scheduler = MultiStepLR(
            self.optimizer,
            milestones=config.lr_scheduler_milestones,
            gamma=config.lr_scheduler_gamma,
        )
        self.buffer = buffer
        self.counter = 0
        self.last_counter = 0
        self.done = False
        self.loss = 0

        self.data_list = []

        self.store_weights()

        self.writer = SummaryWriter(log_dir)
        
        config_variables = {
            key: value for key, value in vars(config).items()
            if not key.startswith("__") and is_json_serializable(value)
        }
        with open(f"{log_dir}/config.json", "w") as file:
            json.dump(config_variables, file, indent=4)  

    def get_weights(self):
        return self.weights_id

    def store_weights(self):
        state_dict = self.model.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        self.weights_id = ray.put(state_dict)

    def run(self):
        self.learning_thread = threading.Thread(target=self._train, daemon=True)
        self.learning_thread.start()

    def _train(self):
        scaler = GradScaler()
        b_seq_len = torch.LongTensor(config.batch_size)
        b_seq_len[:] = config.burn_in_steps + 1

        checkpoint_path = None

        for i in range(1, config.training_steps + 1):

            data_id = ray.get(self.buffer.get_batched_data.remote(i))
            data = ray.get(data_id)

            (
                b_obs,
                b_last_act,
                b_action,
                b_reward,
                b_gamma,
                b_steps,
                b_hidden,
                b_relative_pos,
                b_comm_mask,
                idxes,
                weights,
                old_ptr,
            ) = data
            b_obs, b_last_act, b_action, b_reward = (
                b_obs.to(self.device),
                b_last_act.to(self.device),
                b_action.to(self.device),
                b_reward.to(self.device),
            )
            b_gamma, weights = b_gamma.to(self.device), weights.to(self.device)
            b_hidden = b_hidden.to(self.device)
            b_relative_pos, b_comm_mask = b_relative_pos.to(
                self.device
            ), b_comm_mask.to(self.device)

            b_action = b_action.long()

            b_obs, b_last_act = b_obs.half(), b_last_act.half()

            b_next_seq_len = b_seq_len + b_steps

            with torch.no_grad():
                b_q_ = self.tar_model(
                    b_obs,
                    b_last_act,
                    b_next_seq_len,
                    b_hidden,
                    b_relative_pos,
                    b_comm_mask,
                ).max(1, keepdim=True)[0]

            target_q = b_reward + b_gamma * b_q_

            b_q = self.model(
                b_obs[: -config.forward_steps],
                b_last_act[: -config.forward_steps],
                b_seq_len,
                b_hidden,
                b_relative_pos[:, : -config.forward_steps],
                b_comm_mask[:, : -config.forward_steps],
            ).gather(1, b_action)

            td_error = target_q - b_q

            priorities = (
                td_error.detach().clone().squeeze().abs().clamp(1e-6).cpu().numpy()
            )

            loss = F.mse_loss(b_q, target_q)
            self.loss += loss.item()

            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_norm_dqn)
            # 打印每一层的梯度平均值
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.writer.add_scalar(f"Gradients/{name}", param.grad.norm(), i)
            self.writer.add_scalar("Loss/Total", loss.mean().item(), i)
            # print(f"loss: {loss}")
            scaler.step(self.optimizer)
            scaler.update()

            self.scheduler.step()

            # store new weights in shared memory
            if i % 2 == 0:
                self.store_weights()

            self.buffer.update_priorities.remote(idxes, priorities, old_ptr)

            self.counter += 1

            # update target net, save model
            if i % config.target_network_update_freq == 0:
                self.tar_model.load_state_dict(self.model.state_dict())

            if i % config.save_interval == 0:
                # create save path if not exist
                if not os.path.exists(config.save_path):
                    os.makedirs(config.save_path)
                checkpoint_path = os.path.join(config.save_path, f"{self.counter}.pth")
                torch.save(self.model.state_dict(), checkpoint_path)

            # if i % config.val_interval == 0:
            #     assert (
            #         checkpoint_path is not None
            #     ), "A checkpoint path must be saved before validation"
            #     self.validate(checkpoint_path, self.counter, config.run_id, config.name)

        self.done = True

    def validate(self, checkpoint_path, step, run_id, name):
        """Run subprocess in background and detach directly"""
        print("Validation subprocess started")
        command = [
            "python",
            "validate.py",
            "--checkpoint_path",
            checkpoint_path,
            "--step",
            str(step),
            "--run_id",
            run_id,
            "--name",
            name,
        ]
        print(" ".join(command))

        # Note: took me forever to get this working
        # subprocess does not work apparently, so need to do stuff with Ray
        future = execute_command.remote(command)

    def stats(self, interval: int):
        """
        print log
        """
        print("number of updates: {}".format(self.counter))
        print(
            "update speed: {}/s".format((self.counter - self.last_counter) / interval)
        )
        if self.counter != self.last_counter:
            print("loss: {:.4f}".format(self.loss / (self.counter - self.last_counter)))


        self.last_counter = self.counter
        self.loss = 0
        return self.done
