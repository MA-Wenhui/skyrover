import open3d as o3d
import argparse

# 设置命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Visualize PCD Point Cloud")
    parser.add_argument("pcd_filename", type=str, help="Path to the PCD file")
    return parser.parse_args()

def main():
    # 获取命令行传入的 PCD 文件名
    args = parse_args()
    pcd_filename = args.pcd_filename
    
    # 读取 PCD 点云文件
    pcd = o3d.io.read_point_cloud(pcd_filename)
    
    # 检查文件是否读取成功
    if not pcd.is_empty():
        # 显示点云
        o3d.visualization.draw_geometries([pcd], window_name="PCD Point Cloud",
                                          width=800, height=600, left=50, top=50)
    else:
        print(f"Error: The file '{pcd_filename}' could not be loaded or is empty.")

if __name__ == "__main__":
    main()
