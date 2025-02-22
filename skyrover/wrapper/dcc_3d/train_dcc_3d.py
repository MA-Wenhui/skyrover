import os
import random
import time

import numpy as np
import ray
import torch

from . import config as config
from .actor import Actor
from .global_buffer import GlobalBuffer
from .learner import Learner


os.environ["OMP_NUM_THREADS"] = "1"
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def main(num_actors=config.num_actors, log_interval=config.log_interval):
    ray.init()

    tensorboard_log_dir = f"logs/"        
    os.makedirs(tensorboard_log_dir,exist_ok=True)
    total_files = len([file for file in os.listdir(tensorboard_log_dir)])
    result_dir = os.path.join(tensorboard_log_dir, f"{total_files + 1}")
    os.makedirs(result_dir,exist_ok=True)

    buffer = GlobalBuffer.remote(f"{result_dir}/buffer")
    learner = Learner.remote(buffer,f"{result_dir}/learner")
    time.sleep(1)
    actors = [
        Actor.remote(i, 0.4 ** (1 + (i / (num_actors - 1)) * 7), learner, buffer)
        for i in range(num_actors)
    ]
    
    try:
        for actor in actors:
            actor.run.remote()

        while not ray.get(buffer.ready.remote()):
            time.sleep(5)
            ray.get(learner.stats.remote(5))
            ray.get(buffer.stats.remote(5))

        print("start training")
        buffer.run.remote()
        learner.run.remote()

        done = False
        while not done:
            time.sleep(log_interval)
            done = ray.get(learner.stats.remote(log_interval))
            ray.get(buffer.stats.remote(log_interval))
            print()

    except ray.exceptions.RayActorError as e:
        print(f"An actor has crashed: {e}")
        print("Shutting down the cluster...")
        ray.shutdown()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Shutting down the cluster...")
        ray.shutdown()
    else:
        ray.shutdown()


if __name__ == "__main__":
    main()
