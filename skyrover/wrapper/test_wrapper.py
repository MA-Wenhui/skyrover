from astar_3d_wrapper import AStarAlgorithmWrapper
from cbs_3d_wrapper import CBSAlgorithmWrapper
from dcc_3d_wrapper import DCCAlgorithmWrapper

# 定义3D空间维度
dimension = (5, 5, 5)

# 定义代理
agents = [
    {"name": "agent_0", "start": (0, 0, 0), "goal": (4, 4, 4)},
    {"name": "agent_1", "start": (4, 4, 0), "goal": (0, 0, 4)},
    {"name": "agent_2", "start": (0, 4, 4), "goal": (4, 0, 0)},
]

# 定义初始障碍物
obstacles = [
    (2, 2, 2),  # 一个障碍物
    (3, 3, 3),  # 另一个障碍物
]

def test_astar():
    print("=============== test_astar =================")

    # 创建A*包装器实例
    wrapper = AStarAlgorithmWrapper(agents, dimension, obstacles)
    wrapper.init()

    # 打印初始化状态
    print("Initial positions:")
    for agent in agents:
        print(f"{agent['name']} starts at {agent['start']}")

    # 打印规划路径
    print("\nPlanned paths:")
    for agent_name, path in wrapper.planned_paths.items():
        print(f"{agent_name}: {path}")

    # 模拟路径执行过程
    print("\nSimulation steps:")
    step_count = 0
    while not wrapper.done:
        step_count += 1
        current_positions, done = wrapper.step()
        print(f"Step {step_count}: {current_positions}")

    print("\nFinal positions:")
    for agent in agents:
        print(f"{agent['name']} ends at {agent['goal']}")


def test_cbs():
    print("=============== test_cbs =================")

    # 创建 CBS 算法包装器
    cbs_wrapper = CBSAlgorithmWrapper(agents, dimension, obstacles)

    # 初始化
    cbs_wrapper.init()

    # 按步执行路径规划
    while not cbs_wrapper.done:
        positions, done = cbs_wrapper.step()
        print("Current positions:", positions)

def test_dcc():
    # 创建DCC包装器
    dcc_wrapper = DCCAlgorithmWrapper(agents, dimension, obstacles)

    # 初始化网络路径（假设已经训练好的模型保存为network.pth）
    network_path = "src/dcc_3d/data/64000.pth"

    # 初始化DCC算法
    dcc_wrapper.init(network_path)

    # 模拟运行
    while not dcc_wrapper.done:
        positions, done = dcc_wrapper.step()
        print("Current positions:", positions)
        if done:
            print("All agents have reached their goals.")

    print("DCC Algorithm test complete.")


if __name__ == "__main__":
    test_astar()
    test_cbs()
    test_dcc()