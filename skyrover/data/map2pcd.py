import numpy as np
import open3d as o3d

# 读取文件并解析
def read_map(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 找到“map”部分开始的位置
    map_start_index = lines.index('map\n') + 1  # 跳过“map”行
    # 从“map”部分开始读取地图数据
    map_lines = lines[map_start_index:]

    # 将地图数据转换为numpy矩阵
    map_data = np.array([list(line.strip()) for line in map_lines])  # 逐行读取并移除换行符

    # 将 'T' 替换为 1, '.' 替换为 0
    map_numeric = np.where(map_data == 'T', 1, 0)

    return map_numeric

# 读取并转换地图
file_path = 'warehouse-20-40-10-2-1.map'  # 替换为你的文件路径
map_matrix = read_map(file_path)

# 在z方向扩展，复制10次
z_height = 10
map_3d = np.repeat(map_matrix[np.newaxis, :, :], z_height, axis=0)  # 结果是 (10, height, width)

# 调整为 (height, width, z) 以符合 (x, y, z) 访问顺序
map_3d = np.transpose(map_3d, (1, 2, 0))  # 变为 (height, width, z)

# 细分因子
scale_factor = 5

# 获取所有值为 1 的点的位置 (x, y, z)
points = np.argwhere(map_3d == 1)  # 只保留障碍物点云

# 生成细分坐标
step = 1.0 / scale_factor  # 每个小网格的步长 0.1

# 使用 numpy 高效地创建细分后的所有坐标
fine_points = []
for x, y, z in points:
    x_fine = np.linspace(x, x + step * (scale_factor - 1), scale_factor)
    y_fine = np.linspace(y, y + step * (scale_factor - 1), scale_factor)
    z_fine = np.linspace(z, z + step * (scale_factor - 1), scale_factor)
    
    # 利用 meshgrid 创建所有细分坐标
    grid_x, grid_y, grid_z = np.meshgrid(x_fine, y_fine, z_fine)
    
    # 过滤掉边界点，边界值设为0
    # 边界判断：如果坐标处于细分区域的最小值或最大值，设为0
    mask = ((grid_x == grid_x.min()) | (grid_x == grid_x.max()) |
            (grid_y == grid_y.min()) | (grid_y == grid_y.max()) |
            (grid_z == grid_z.min()) | (grid_z == grid_z.max()))
    
    # 设置边界的细分点为 0
    grid_x[mask] = 0
    grid_y[mask] = 0
    grid_z[mask] = 0
    
    fine_points.append(np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T)

# 合并所有细分点
fine_points = np.vstack(fine_points)

# 创建 Open3D 点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(fine_points.astype(np.float32))  # 添加点云数据

# 使用二进制格式保存 PCD，提高保存速度
pcd_filename = "map_pointcloud_fine.pcd"
o3d.io.write_point_cloud(pcd_filename, pcd, write_ascii=True)  # 使用二进制格式

print(f"Point cloud saved as {pcd_filename}, with {len(fine_points)} points.")
