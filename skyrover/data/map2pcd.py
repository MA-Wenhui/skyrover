import numpy as np
import open3d as o3d
import argparse

def read_map(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    map_start_index = lines.index('map\n') + 1
    map_lines = lines[map_start_index:]
    map_data = np.array([list(line.strip()) for line in map_lines])
    return np.where(map_data == 'T', 1, 0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, help="Path to the map file")
    parser.add_argument("--out", type=str, help="Output point cloud filename")
    args = parser.parse_args()

    map_matrix = read_map(args.map)
    map_3d = np.repeat(map_matrix[np.newaxis, :, :], 10, axis=0)
    map_3d = np.transpose(map_3d, (1, 2, 0))
    
    scale_factor = 5
    points = np.argwhere(map_3d == 1)
    step = 1.0 / scale_factor
    fine_points = []
    
    for x, y, z in points:
        x_fine = np.linspace(x, x + step * (scale_factor - 1), scale_factor)
        y_fine = np.linspace(y, y + step * (scale_factor - 1), scale_factor)
        z_fine = np.linspace(z, z + step * (scale_factor - 1), scale_factor)
        grid_x, grid_y, grid_z = np.meshgrid(x_fine, y_fine, z_fine)
        mask = ((grid_x == grid_x.min()) | (grid_x == grid_x.max()) |
                (grid_y == grid_y.min()) | (grid_y == grid_y.max()) |
                (grid_z == grid_z.min()) | (grid_z == grid_z.max()))
        grid_x[mask] = 0
        grid_y[mask] = 0
        grid_z[mask] = 0
        fine_points.append(np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T)
    
    fine_points = np.vstack(fine_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fine_points.astype(np.float32))
    o3d.io.write_point_cloud(args.out, pcd, write_ascii=True)
    print(f"Point cloud saved as {args.out}, with {len(fine_points)} points.")

if __name__ == "__main__":
    main()