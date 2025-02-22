import heapq
import random

class Node:
    def __init__(self, x, y, z, cost=0, heuristic=0, parent=None):
        self.x = x
        self.y = y
        self.z = z
        self.cost = cost  # g: cost from start to this node
        self.heuristic = heuristic  # h: heuristic to the goal
        self.parent = parent

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)


def heuristic(node, goal):
    # Using Manhattan distance for heuristic
    return abs(node.x - goal.x) + abs(node.y - goal.y) + abs(node.z - goal.z)


def get_neighbors(node, grid, obstacles):
    # Generate valid neighbors
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    neighbors = []
    for dx, dy, dz in directions:
        nx, ny, nz = node.x + dx, node.y + dy, node.z + dz
        if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and 0 <= nz < len(grid[0][0]) and (nx, ny, nz) not in obstacles:
            neighbors.append(Node(nx, ny, nz))
    return neighbors


def a_star_3d(grid, start, goal, obstacles):
    open_set = []
    closed_set = set()

    start_node = Node(start[0], start[1], start[2], cost=0, heuristic=heuristic(Node(*start), Node(*goal)))
    heapq.heappush(open_set, start_node)

    while open_set:
        current_node = heapq.heappop(open_set)
        # print((current_node.x, current_node.y, current_node.z),goal)
        if (current_node.x, current_node.y, current_node.z) == goal:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y, current_node.z))
                current_node = current_node.parent
            return path[::-1]

        closed_set.add((current_node.x, current_node.y, current_node.z))

        for neighbor in get_neighbors(current_node, grid, obstacles):
            if (neighbor.x, neighbor.y, neighbor.z) in closed_set:
                continue

            neighbor.cost = current_node.cost + 1
            neighbor.heuristic = heuristic(neighbor, Node(*goal))
            neighbor.parent = current_node

            if any((n.x, n.y, n.z) == (neighbor.x, neighbor.y, neighbor.z) and n.cost + n.heuristic <= neighbor.cost + neighbor.heuristic for n in open_set):
                continue

            heapq.heappush(open_set, neighbor)

    return None  # No path found


def plan_agents_paths(grid, agents,static_obstacles):
    obstacles = set(static_obstacles)
    planned_paths = {}
    random.shuffle(agents)

    for agent in agents:
        start, goal = agent
        path = a_star_3d(grid, start, goal, obstacles)
        if path is None:
            print(f"No path found for agent from {start} to {goal}")
            continue

        planned_paths[(start, goal)] = path
        obstacles.update(path)  # Add the path as obstacles for subsequent agents

    return planned_paths


# Example Usage
if __name__ == "__main__":
    # Define a 10x10x10 grid
    grid_size = (10,10,10)
    grid = [[[0 for _ in range(grid_size[0])] for _ in range(grid_size[1])] for _ in range(grid_size[2])]
    static_obstacles = [
        (1, 1, 1), (1, 1, 2), (1, 1, 3),  # Example obstacle line
        (2, 2, 2), (2, 3, 2), (2, 4, 2),  # Example obstacle line
    ]

    # Define agents with (start, goal) positions
    agents = [((0, 0, 0), (9, 9, 9)), ((9, 0, 0), (0, 9, 9)), ((0, 9, 0), (9, 0, 9))]

    # Plan paths
    planned_paths = plan_agents_paths(grid, agents,static_obstacles)

    # Print results
    for agent, path in planned_paths.items():
        print(f"Agent from {agent[0]} to {agent[1]} path: {path}")
