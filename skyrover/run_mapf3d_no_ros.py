import argparse
import numpy as np
import time
import threading
from queue import Queue
import random

from panda3d.core import Geom, GeomNode, GeomVertexFormat, GeomVertexData, GeomVertexWriter, GeomPoints

from panda3d.core import RigidBodyCombiner, NodePath, RenderModeAttrib, Geom, GeomNode
from panda3d.core import GeomVertexFormat, GeomVertexData, GeomTriangles, GeomVertexWriter
from panda3d.core import Point3
from panda3d.core import LineSegs
from panda3d.core import LineSegs, Vec3, NodePath
from direct.showbase.ShowBase import ShowBase
from direct.task import Task

from skyrover.executor import Mapf3DExecutor
from skyrover.pcd2grid import load_pcd, generate_3d_grid

import numpy as np


def generate_random_tasks(grid, num_tasks=100):
    # 找到所有可通过的点 (grid == 0) 表示可通过区域
    free_points = np.argwhere(grid == 0)  # 获取所有值为0的位置，表示可通过区域
    np.random.shuffle(free_points)  # 使用 NumPy 自带的随机打乱
    print(free_points)
    tasks = []
    for i in range(num_tasks):
        if i >= len(free_points):  # 如果可用点不足，则结束
            break
        
        # 起始点
        start_point = tuple(free_points[i])

        # 目标点：选择一个距离较远的点来避免聚集
        goal_point = None
        while goal_point == start_point or goal_point is None:
            # 随机选择目标点
            goal_point = tuple(random.choice(free_points))
        
        task_name = f"x{i}"  # 给任务命名
        tasks.append({
            "name": task_name,
            "start": start_point,
            "goal": goal_point
        })

    return tasks

def process_pcd(pcd_file, min_bounds, max_bounds, resolution=1.0):

    print(f"load_pcd...") 
    points = load_pcd(pcd_file)
    print(f"generate_3d_grid...") 
    grid = generate_3d_grid(points, np.array(min_bounds), np.array(max_bounds), resolution)
    return grid

class Mapf3DRenderer(ShowBase):
    def __init__(self, executor,min_bounds, max_bounds):
        ShowBase.__init__(self)
        self.executor = executor
        self.win.setClearColor((0, 0, 0, 1))  # 设置背景为纯黑色 (RGBA)

        # 共享队列用于存储智能体最新位置
        self.position_queue = Queue()

        # 开启后台线程，不阻塞渲染线程
        self.running = True
        self.update_thread = threading.Thread(target=self.update_agent_positions, daemon=True)
        self.update_thread.start()

        
        print(f"create_obstacles...")
        self.create_obstacles()

        # 创建多个智能体（小球）
        self.agent_models = {}
        for task in self.executor.tasks:
            agent = self.loader.loadModel("models/smiley")
            agent.setScale(0.5)
            agent.reparentTo(self.render)
            self.agent_models[task["name"]] = agent

        self.taskMgr.add(self.update_agents_task, "update_agents_task")

    def update_camera_from_mouse(self):
        """根据鼠标位置更新相机视角"""
        mouse_pos = self.mouseWatcherNode.getMouse()
        if mouse_pos:
            # 获取鼠标位置，范围是 [-1, 1]
            x = mouse_pos.getX()
            y = mouse_pos.getY()

            # 设定相机偏航角度和俯仰角度（控制水平和垂直方向的视角）
            pitch = y * 45  # 控制俯仰角度（上下）
            yaw = x * 45    # 控制偏航角度（左右）

            # 更新相机的方向
            self.camera.setHpr(yaw, pitch, 0)
            self.camera.lookAt(0, 0, 0)  # 或者根据需要选择目标位置

    def task_camera_control(self, task):
        """更新每帧鼠标控制的相机视角"""
        self.update_camera_from_mouse()
        return Task.cont  # 继续执行
    def create_obstacles(self):
        """使用 RigidBodyCombiner 合并障碍物，减少绘制开销"""
        rbc = RigidBodyCombiner("obstacles")  
        root = NodePath(rbc)  

        obs = self.executor.get_obstacles()  # 获取障碍物点

        # 只创建一个 cube 模型，然后克隆位置
        base_cube = self.loader.loadModel("models/box")
        base_cube.setScale(0.8)
        base_cube.clearTexture()
        base_cube.setColor(0.7, 0.7, 0.7, 0.1)  # 设定实体颜色（灰色）

        for pos in obs:
            x, y, z = pos
            cube = base_cube.copyTo(root)  # 直接克隆
            cube.setPos(x, y, z)

        root.reparentTo(self.render)
        rbc.collect()  # 进行优化合并

    def update_agent_positions(self):
        """后台线程持续计算智能体新位置，并存入队列"""
        while self.running:
            agents_pos, _ = self.executor.step()
            self.position_queue.put(agents_pos)
            time.sleep(1)  # 控制更新频率，防止 CPU 过载

    def update_agents_task(self, task):
        """从队列获取最新智能体位置，并更新 Panda3D 场景"""
        while not self.position_queue.empty():
            cam_pos = self.camera.getPos()
            cam_look = self.camera.getQuat().getForward()  # 获取相机朝向的前向向量
            print(f"Camera Position: {cam_pos}, Look At Direction: {cam_look}")
            agents_pos = self.position_queue.get()
            for agent_name, pos in agents_pos.items():
                if agent_name in self.agent_models:
                    self.agent_models[agent_name].setPos(*pos)
        return Task.cont  # 继续执行

    def destroy(self):
        """关闭时停止后台线程"""
        self.running = False
        self.update_thread.join()

def main(args=None):
    parser = argparse.ArgumentParser(description="Mapf3DExecutor Node")
    parser.add_argument("--alg", type=str, default="3dcbs", help="algorithm for 3D MAPF")
    parser.add_argument("--pcd", type=str, default="map.pcd", help="point cloud file")
    parser.add_argument("--model", type=str, default="model.pth", help="model used for 3D DCC")
    parser.add_argument("--min_bound", type=float, nargs=3, default=[-21.0, -39.0, 0.0], help="minimum bounds (x, y, z)")
    parser.add_argument("--max_bound", type=float, nargs=3, default=[21.0, 23.0, 15.0], help="maximum bounds (x, y, z)")


    known_args, unknown_args = parser.parse_known_args()
    
    # min_bounds = [0, 0, 0]
    # max_bounds = [123, 321, 9]
    min_bounds = known_args.min_bound
    max_bounds = known_args.max_bound

    print(f"Custom param received: {known_args.alg}") 
    print(f"Min bounds: {min_bounds}, Max bounds: {max_bounds}")


    print(f"Custom param received: {known_args.alg}") 
    print(f"process_pcd...") 
    grid = process_pcd(known_args.pcd, min_bounds, max_bounds)
    print(grid)
    print(f"generate_random_tasks...") 
    tasks = generate_random_tasks(grid,10)  # Generate random tasks

    print(f"Init Mapf3DExecutor...") 
    executor = Mapf3DExecutor(known_args.alg, grid, min_bounds, known_args.model)

    print(f"set_tasks...") 
    executor.set_tasks(tasks)

    # 启动渲染
    app = Mapf3DRenderer(executor,min_bounds,max_bounds)
    try:
        app.run()
    finally:
        app.destroy()

if __name__ == '__main__':
    main()
