from .algo_wrapper import AlgorithmWrapperBase
from .cbs_3d import CBS_3D,Environment

class CBSAlgorithmWrapper(AlgorithmWrapperBase):
    def __init__(self, agents, space_dim, obstacles):
        """
        初始化CBSAlgorithmWrapper
        :param agents: 代理信息列表，格式为 [{"name": str, "start": tuple, "goal": tuple}, ...]
        :param space_dim: 3D空间的维度，格式为 (x, y, z)
        :param obstacles: 初始障碍物列表，格式为 [(x, y, z), ...]
        """
        self.agents = agents
        self.space_dim = space_dim
        self.obstacles = set(obstacles)
        self.environment = self.create_environment()
        self.planned_paths = {}
        self.current_positions = {}

        # 初始化 CBS 并规划路径
        self.plan_paths()
        self.done = False
        self.episode_length = 0

    def create_environment(self):
        """创建环境对象，供CBS使用"""
        # 根据现有的Environment类初始化3D空间环境
        return Environment(self.space_dim, self.agents, self.obstacles)

    def plan_paths(self):
        """使用CBS算法为所有代理规划路径"""
        print(f"CBSAlgorithmWrapper plan_paths")
        cbs_solver = CBS_3D(self.environment)
        plan = cbs_solver.search()

        if plan:
            # 存储每个代理的路径
            for agent in self.agents:
                name = agent["name"]
                self.planned_paths[name] = [
                    (step["x"], step["y"], step["z"]) for step in plan.get(name, [])
                ]
                self.current_positions[name] = tuple(agent["start"])
        else:
            print("CBS failed to find a solution.")
            self.planned_paths = {}
            self.current_positions = {}

    def init(self):
        """初始化动作和状态"""
        # self.actions = {agent["name"]: [] for agent in self.agents}
        self.done = False
        self.plan_paths()
        self.episode_length = 0

    def reset(self, agents, space_dim, obstacles):
        self.agents = agents
        self.space_dim = space_dim
        self.obstacles = set(obstacles)
        self.environment = self.create_environment()
        self.planned_paths = {}
        self.current_positions = {}

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
