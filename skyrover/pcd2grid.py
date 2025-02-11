
import numpy as np

def load_pcd(file_path):
    """Load point cloud from a PCD file."""
    points = []
    header = True
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if header:
                if line.startswith('DATA'):  # End of header section
                    header = False
                continue

            # Parse point data
            if line:
                values = line.split()
                if len(values) >= 3:
                    try:
                        points.append([float(values[0]), float(values[1]), float(values[2])])
                    except ValueError:
                        continue

    return np.array(points)

def generate_3d_grid(points, min_bounds, max_bounds, grid_size):
    # Calculate the center of the grid using integer division for safe indexing
    center_min = tuple(int(i + grid_size // 2) for i in min_bounds)
    center_max = tuple(int(i - grid_size // 2) for i in max_bounds)

    # Ensure grid dimensions are integers
    grid_dimensions = (
        int(center_max[0] - center_min[0] + 1),
        int(center_max[1] - center_min[1] + 1),
        int(center_max[2] - center_min[2] + 1)
    )

    # Initialize a 3D grid with zeros
    grid = np.zeros(grid_dimensions, dtype=np.uint8)

    # Populate the grid based on point positions
    for point in points:
        x, y, z = point
        # Calculate the grid index, ensuring it's within bounds
        index_x = int(np.floor(x + grid_size / 2 ))- center_min[0]
        index_y = int(np.floor(y + grid_size / 2)) - center_min[1]
        index_z = int(np.floor(z + grid_size / 2)) - center_min[2]

        if (0 <= index_x < grid_dimensions[0] and 
            0 <= index_y < grid_dimensions[1] and 
            0 <= index_z < grid_dimensions[2]):
            grid[index_x, index_y, index_z] = 1  # Mark occupied cells

    return grid
