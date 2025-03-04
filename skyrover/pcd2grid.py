
import numpy as np

def load_pcd(file_path):
    """Load point cloud from a PCD file using NumPy for faster processing."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 找到数据部分的起始索引
    data_start = next(i for i, line in enumerate(lines) if line.strip().startswith('DATA')) + 1
    
    # 直接用 NumPy 读取数据部分
    points = np.loadtxt(lines[data_start:], dtype=np.float32)

    # 计算点云范围
    min_bounds = points.min(axis=0)
    max_bounds = points.max(axis=0)
    print(f"load_pcd min_bounds:{min_bounds}")
    print(f"load_pcd max_bounds:{max_bounds}")

    return points


import numpy as np

def generate_3d_grid(points, min_bounds, max_bounds, grid_size):
    # 计算网格的中心点
    center_min = np.floor(min_bounds / grid_size).astype(int)
    center_max = np.floor(max_bounds / grid_size).astype(int)

    # 计算网格的维度
    grid_dimensions = (center_max - center_min + 1)
    grid_dimensions = tuple(grid_dimensions.astype(int))

    print(f"generate_3d_grid grid_dimensions: {grid_dimensions}")

    # 初始化 3D 网格
    grid = np.zeros(grid_dimensions, dtype=np.uint8)

    # 计算所有点的网格索引（不做偏移）
    indices = np.floor(points / grid_size).astype(int) - center_min

    # 过滤掉越界的点
    valid_mask = np.all((indices >= 0) & (indices < grid_dimensions), axis=1)
    valid_indices = indices[valid_mask]

    # 将有效点的对应网格位置设为 1
    grid[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] = 1

    return grid
