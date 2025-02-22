import copy
import pickle
import random

import numpy as np

from tqdm import tqdm
import os

MAX_ITERATIONS = 1000
FREE_SPACE = 0
OBSTACLE = 1


def flood_fill(matrix, x, y, z, old_value, new_value):
    stack = [(x, y, z)]
    while stack:
        cx, cy, cz = stack.pop()
        if (
            cx < 0 or cx >= matrix.shape[0]
            or cy < 0 or cy >= matrix.shape[1]
            or cz < 0 or cz >= matrix.shape[2]
            or matrix[cx, cy, cz] != old_value
        ):
            continue
        matrix[cx, cy, cz] = new_value
        stack.append((cx + 1, cy, cz))
        stack.append((cx - 1, cy, cz))
        stack.append((cx, cy + 1, cz))
        stack.append((cx, cy - 1, cz))
        stack.append((cx, cy, cz + 1))
        stack.append((cx, cy, cz - 1))

def generate_map(width, height, depth, density, tolerance=0.005):
    iteration = 0

    while iteration < MAX_ITERATIONS:
        matrix = np.random.choice(
            [0, 1], size=(width, height, depth), p=[1 - density, density]
        )

        filled_matrix = matrix.copy()
        flood_fill(filled_matrix, 0, 0, 0, 0, 2)
        total_free_space = np.sum(filled_matrix == 2)
        total_space = width * height * depth
        actual_density = 1 - total_free_space / total_space
        if abs(actual_density - density) < tolerance:
            filled_matrix[filled_matrix == 0] = 1
            filled_matrix[filled_matrix == 2] = 0
            return filled_matrix
        iteration += 1

    raise ValueError(
        f"Unable to generate a grid with the desired density of {density} after {MAX_ITERATIONS} iterations."
    )


def move(loc, d):
    directions = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]
    return (
        loc[0] + directions[d][0],
        loc[1] + directions[d][1],
        loc[2] + directions[d][2],
    )


def space_partition(grid_map):
    empty_spots = np.argwhere(np.array(grid_map) == FREE_SPACE).tolist()
    empty_spots = [tuple(pos) for pos in empty_spots]
    partitions = []
    while empty_spots:
        start_loc = empty_spots.pop()
        open_list = [start_loc]
        close_list = []
        while open_list:
            loc = open_list.pop(0)
            for d in range(6):  # Six directions for 3D
                child_loc = move(loc, d)
                if (
                    child_loc[0] < 0
                    or child_loc[0] >= len(grid_map)
                    or child_loc[1] < 0
                    or child_loc[1] >= len(grid_map[0])
                    or child_loc[2] < 0
                    or child_loc[2] >= len(grid_map[0][0])
                ):
                    continue
                if grid_map[child_loc[0]][child_loc[1]][child_loc[2]] == OBSTACLE:
                    continue
                if child_loc in empty_spots:
                    empty_spots.remove(child_loc)
                    open_list.append(child_loc)
            close_list.append(loc)
        partitions.append(close_list)
    return partitions


def generate_random_agents(grid_map, map_partitions, num_agents):
    starts, goals = [], []
    counter = 0
    partitions = copy.deepcopy(map_partitions)
    while counter < num_agents:
        partitions = [p for p in partitions if len(p) >= 2]
        partition_index = random.randint(0, len(partitions) - 1)
        si, sj, sk = random.choice(partitions[partition_index])
        partitions[partition_index].remove((si, sj, sk))
        gi, gj, gk = random.choice(partitions[partition_index])
        partitions[partition_index].remove((gi, gj, gk))
        starts.append((si, sj, sk))
        goals.append((gi, gj, gk))
        counter += 1
    # convert to numpy array
    starts = np.array(starts, dtype=int)
    goals = np.array(goals, dtype=int)

    return starts, goals


### Main function to generate data and save into pickle files
def main(width, height, depth, density, num_agents, num_instances):
    instances = []

    print(
        f"Generating instances for {width}x{height}x{depth} map with {num_agents} agents ..."
    )
    for _ in tqdm(range(num_instances)):
        map_ = generate_map(width, height, depth, density)
        map_partitions = space_partition(map_)
        starts, goals = generate_random_agents(map_, map_partitions, num_agents)
        instances.append((map_, starts, goals))
    os.makedirs("./test_set/",exist_ok=True)
    file_name = f"./test_set/{width}x{height}x{depth}_{num_agents}agents_{density}density.pth"
    with open(file_name, "wb") as f:
        pickle.dump(instances, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate random 3D maps and agents for testing"
    )
    parser.add_argument("--width", type=int, default=20, help="Width of the map")
    parser.add_argument("--height", type=int, default=20, help="Height of the map")
    parser.add_argument("--depth", type=int, default=20, help="Depth of the map")
    parser.add_argument("--density", type=float, default=0.3, help="Density of the map")
    parser.add_argument("--num_agents", type=int, default=5, help="Number of agents")
    parser.add_argument(
        "--num_instances", type=int, default=100, help="Number of instances"
    )
    args = parser.parse_args()

    main(
        args.width,
        args.height,
        args.depth,
        args.density,
        args.num_agents,
        args.num_instances,
    )
