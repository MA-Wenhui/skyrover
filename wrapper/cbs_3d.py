"""
based on https://github.com/atb033/multi_agent_path_planning/blob/master/centralized/cbs/cbs.py
"""

from math import fabs
from itertools import combinations
from copy import deepcopy


class Location(object):
    def __init__(self, x=-1, y=-1, z=-1):
        self.x = x
        self.y = y
        self.z = z
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z
    def __str__(self):
        return str((self.x, self.y, self.z))


class State(object):
    def __init__(self, time, location):
        self.time = time
        self.location = location
    def __eq__(self, other):
        return self.time == other.time and self.location == other.location
    def __hash__(self):
        return hash(str(self.time)+str(self.location.x) + str(self.location.y) + str(self.location.z))
    def is_equal_except_time(self, state):
        return self.location == state.location
    def __str__(self):
        return str((self.time, self.location.x, self.location.y, self.location.z))

class Conflict(object):
    VERTEX = 1
    EDGE = 2
    def __init__(self):
        self.time = -1
        self.type = -1

        self.agent_1 = ''
        self.agent_2 = ''

        self.location_1 = Location()
        self.location_2 = Location()

    def __str__(self):
        return '(' + str(self.time) + ', ' + self.agent_1 + ', ' + self.agent_2 + \
             ', '+ str(self.location_1) + ', ' + str(self.location_2) + ')'


class VertexConstraint(object):
    def __init__(self, time, location):
        self.time = time
        self.location = location

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location
    def __hash__(self):
        return hash(str(self.time)+str(self.location))
    def __str__(self):
        return '(' + str(self.time) + ', '+ str(self.location) + ')'

class EdgeConstraint(object):
    def __init__(self, time, location_1, location_2):
        self.time = time
        self.location_1 = location_1
        self.location_2 = location_2
    def __eq__(self, other):
        return self.time == other.time and self.location_1 == other.location_1 \
            and self.location_2 == other.location_2
    def __hash__(self):
        return hash(str(self.time) + str(self.location_1) + str(self.location_2))
    def __str__(self):
        return '(' + str(self.time) + ', '+ str(self.location_1) +', '+ str(self.location_2) + ')'


class Constraints(object):
    def __init__(self):
        self.vertex_constraints = set()
        self.edge_constraints = set()

    def add_constraint(self, other):
        self.vertex_constraints |= other.vertex_constraints
        self.edge_constraints |= other.edge_constraints

    def __str__(self):
        return "VC: " + str([str(vc) for vc in self.vertex_constraints])  + \
            "EC: " + str([str(ec) for ec in self.edge_constraints])


class Environment(object):
    def __init__(self, dimension, agents, obstacles):
        self.dimension = dimension
        self.obstacles = obstacles

        self.agents = agents
        self.agent_dict = {}

        self.make_agent_dict()

        self.constraints = Constraints()
        self.constraint_dict = {}

        self.low_level = LowLevelSearch(self)

    def get_neighbors(self, state):
        neighbors = []

        # Wait action
        n = State(state.time + 1, state.location)
        if self.state_valid(n):
            neighbors.append(n)
        # Movements in 3D space
        directions = [
            (1, 0, 0), (-1, 0, 0),  # Move along x-axis
            (0, 1, 0), (0, -1, 0),  # Move along y-axis
            (0, 0, 1), (0, 0, -1)   # Move along z-axis
        ]
        for dx, dy, dz in directions:
            new_location = Location(state.location.x + dx, state.location.y + dy, state.location.z + dz)
            n = State(state.time + 1, new_location)
            if self.state_valid(n) and self.transition_valid(state, n):
                neighbors.append(n)
        return neighbors

    def get_first_conflict(self, solution):
        max_t = max([len(plan) for plan in solution.values()])
        result = Conflict()
        for t in range(max_t):
            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1 = self.get_state(agent_1, solution, t)
                state_2 = self.get_state(agent_2, solution, t)
                if state_1.is_equal_except_time(state_2):
                    result.time = t
                    result.type = Conflict.VERTEX
                    result.location_1 = state_1.location
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2
                    return result

            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1a = self.get_state(agent_1, solution, t)
                state_1b = self.get_state(agent_1, solution, t + 1)

                state_2a = self.get_state(agent_2, solution, t)
                state_2b = self.get_state(agent_2, solution, t + 1)

                if state_1a.is_equal_except_time(
                    state_2b
                ) and state_1b.is_equal_except_time(state_2a):
                    result.time = t
                    result.type = Conflict.EDGE
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2
                    result.location_1 = state_1a.location
                    result.location_2 = state_1b.location
                    return result
        return False

    def create_constraints_from_conflict(self, conflict):
        constraint_dict = {}
        if conflict.type == Conflict.VERTEX:
            v_constraint = VertexConstraint(conflict.time, conflict.location_1)
            constraint = Constraints()
            constraint.vertex_constraints |= {v_constraint}
            constraint_dict[conflict.agent_1] = constraint
            constraint_dict[conflict.agent_2] = constraint

        elif conflict.type == Conflict.EDGE:
            constraint1 = Constraints()
            constraint2 = Constraints()

            e_constraint1 = EdgeConstraint(
                conflict.time, conflict.location_1, conflict.location_2
            )
            e_constraint2 = EdgeConstraint(
                conflict.time, conflict.location_2, conflict.location_1
            )

            constraint1.edge_constraints |= {e_constraint1}
            constraint2.edge_constraints |= {e_constraint2}

            constraint_dict[conflict.agent_1] = constraint1
            constraint_dict[conflict.agent_2] = constraint2

        return constraint_dict

    def get_state(self, agent_name, solution, t):
        if t < len(solution[agent_name]):
            return solution[agent_name][t]
        else:
            return solution[agent_name][-1]

    def state_valid(self, state: State):
        return state.location.x >= 0 and state.location.x < self.dimension[0] \
            and state.location.y >= 0 and state.location.y < self.dimension[1] \
            and state.location.z >= 0 and state.location.z < self.dimension[2] \
            and VertexConstraint(state.time, state.location) not in self.constraints.vertex_constraints \
            and (state.location.x, state.location.y, state.location.z) not in self.obstacles

    def transition_valid(self, state_1: State, state_2: State):
        return (
            EdgeConstraint(state_1.time, state_1.location, state_2.location)
            not in self.constraints.edge_constraints
        )

    def admissible_heuristic(self, state: State, agent_name):
        goal = self.agent_dict[agent_name]["goal"]
        return (
            fabs(state.location.x - goal.location.x)
            + fabs(state.location.y - goal.location.y)
            + fabs(state.location.z - goal.location.z)
        )

    def is_at_goal(self, state:State, agent_name):
        goal_state = self.agent_dict[agent_name]["goal"]
        return state.is_equal_except_time(goal_state)

    def make_agent_dict(self):
        for agent in self.agents:
            start_state = State(
                0, Location(agent["start"][0], agent["start"][1], agent["start"][2])
            )
            goal_state = State(
                0, Location(agent["goal"][0], agent["goal"][1], agent["goal"][2])
            )
            self.agent_dict.update(
                {agent["name"]: {"start": start_state, "goal": goal_state}}
            )

    def compute_solution(self):
        solution = {}
        for agent in self.agent_dict.keys():
            self.constraints = self.constraint_dict.setdefault(agent, Constraints())
            local_solution = self.low_level.search(agent)
            if not local_solution:
                return False
            solution.update({agent: local_solution})
        return solution

    def compute_solution_cost(self, solution):
        return sum([len(path) for path in solution.values()])


class LowLevelSearch:
    def __init__(self, env:Environment):
        self.agent_dict = env.agent_dict
        self.admissible_heuristic = env.admissible_heuristic
        self.is_at_goal = env.is_at_goal
        self.get_neighbors = env.get_neighbors

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def search(self, agent_name):
        """
        low level search
        """
        initial_state = self.agent_dict[agent_name]["start"]
        step_cost = 1

        closed_set = set()
        open_set = {initial_state}

        came_from = {}

        g_score = {}
        g_score[initial_state] = 0

        f_score = {}

        f_score[initial_state] = self.admissible_heuristic(initial_state, agent_name)

        while open_set:
            temp_dict = {
                open_item: f_score.setdefault(open_item, float("inf"))
                for open_item in open_set
            }
            current = min(temp_dict, key=temp_dict.get)

            if self.is_at_goal(current, agent_name):
                return self.reconstruct_path(came_from, current)

            open_set -= {current}
            closed_set |= {current}

            neighbor_list = self.get_neighbors(current)

            for neighbor in neighbor_list:
                if neighbor in closed_set:
                    continue

                tentative_g_score = (
                    g_score.setdefault(current, float("inf")) + step_cost
                )

                if neighbor not in open_set:
                    open_set |= {neighbor}
                elif tentative_g_score >= g_score.setdefault(neighbor, float("inf")):
                    continue

                came_from[neighbor] = current

                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.admissible_heuristic(
                    neighbor, agent_name
                )
        return False


class HighLevelNode(object):
    def __init__(self):
        self.solution = {}
        self.constraint_dict = {}
        self.cost = 0

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.solution == other.solution and self.cost == other.cost

    def __hash__(self):
        return hash((self.cost))

    def __lt__(self, other):
        return self.cost < other.cost


class CBS_3D(object):
    def __init__(self, environment: Environment):
        self.env = environment
        self.open_set = set()
        self.closed_set = set()

    def search(self):
        start = HighLevelNode()
        # TODO: Initialize it in a better way
        start.constraint_dict = {}
        for agent in self.env.agent_dict.keys():
            start.constraint_dict[agent] = Constraints()
        start.solution = self.env.compute_solution()
        if not start.solution:
            return {}
        start.cost = self.env.compute_solution_cost(start.solution)

        self.open_set |= {start}

        while self.open_set:
            P: HighLevelNode = min(self.open_set)
            self.open_set -= {P}
            self.closed_set |= {P}

            self.env.constraint_dict = P.constraint_dict
            conflict_dict = self.env.get_first_conflict(P.solution)
            if not conflict_dict:
                print("solution found")

                return self.generate_plan(P.solution)

            constraint_dict = self.env.create_constraints_from_conflict(conflict_dict)

            for agent in constraint_dict.keys():
                new_node = deepcopy(P)
                new_node.constraint_dict[agent].add_constraint(constraint_dict[agent])

                self.env.constraint_dict = new_node.constraint_dict
                new_node.solution = self.env.compute_solution()
                if not new_node.solution:
                    continue
                new_node.cost = self.env.compute_solution_cost(new_node.solution)

                # TODO: ending condition
                if new_node not in self.closed_set:
                    self.open_set |= {new_node}

        return {}

    def generate_plan(self, solution):
        plan = {}
        for agent, path in solution.items():
            path_dict_list = [
                {
                    "t": state.time,
                    "x": state.location.x,
                    "y": state.location.y,
                    "z": state.location.z,
                }
                for state in path
            ]
            plan[agent] = path_dict_list
        return plan


def main():
    # 定义3D网格的维度
    dimension = (5, 5, 5)  # 5x5x5的3D空间

    # 定义代理
    agents = [
        {"name": "agent_1", "start": (0, 0, 0), "goal": (4, 4, 4)},
        {"name": "agent_2", "start": (4, 4, 0), "goal": (0, 0, 4)},
        {"name": "agent_3", "start": (0, 4, 4), "goal": (4, 0, 0)},
    ]

    # 定义障碍物
    obstacles = [
        (2, 2, 2),  # 中间的一个障碍物
        (3, 3, 3),  # 另一个障碍物
    ]

    # 创建环境
    environment = Environment(dimension, agents, obstacles)
    low_level = LowLevelSearch(environment)
    agent_name = "agent_1"
    path = low_level.search(agent_name)    
    if path:
        print(f"LowLevelSearch Path found for {agent_name}: {[str(s) for s in path]}")
    else:
        print(f"LowLevelSearch No path found for {agent_name}.")


    # 创建CBS对象并运行搜索
    cbs_3d_solver = CBS_3D(environment)
    plan = cbs_3d_solver.search()

    # 打印结果
    if plan:
        print("Solution found!")
        for agent, path in plan.items():
            print(f"Agent {agent}'s path:")
            for step in path:
                print(f"t={step['t']}, x={step['x']}, y={step['y']}, z={step['z']}")
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()
