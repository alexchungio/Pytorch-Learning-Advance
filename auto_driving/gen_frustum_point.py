import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_frustum_points(camera_intrinsic, image_width=3840, image_height=2160,
                            max_depth=50, frustum_shape=None):
    """
    Generate camera frustum point cloud

    Args:
        camera_intrinsic: 3x3 camera intrinsic matrix
        image_width: Image width
        image_height: Image height
        max_depth: Maximum depth in meters
        frustum_shape: Shape of frustum grid

    Returns:
        points_3d: Nx3 3D point cloud array (in camera coordinate system)
    """
    num_width, num_height, num_depth = frustum_shape

    # generate pixel coordinate grid
    u = np.linspace(0, image_width - 1, num_width)
    v = np.linspace(0, image_height - 1, num_height)

    U, V = np.meshgrid(u, v)

    # generate depth values
    depth = np.linspace(0, max_depth - 1, num_depth)

    U_flatten = U.reshape(-1, 1)
    V_flatten = V.reshape(-1, 1)
    ones_flatten = np.ones_like(U_flatten)
    uv_one = np.concatenate((U_flatten, V_flatten, ones_flatten), axis=1)

    # back-project all pixels to normalized camera coordinate
    camera_intrinsic_inv = np.linalg.inv(camera_intrinsic)
    normalize_coord = uv_one @ camera_intrinsic_inv.T

    # scale normalized coord by depth
    all_points = normalize_coord[np.newaxis, :, :] * depth[:, np.newaxis, np.newaxis]

    points_3d = all_points.reshape(-1, 3)

    return points_3d


def visualize_frustum(points_3d, title="Camera Frustum Point Cloud"):
    """
    Visualize frustum point cloud

    Args:
        points_3d: Nx3 3D point cloud array
        title: Plot title
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Sample points for faster visualization
    if len(points_3d) > 100000:
        indices = np.random.choice(len(points_3d), 100000, replace=False)
        points_sampled = points_3d[indices]
    else:
        points_sampled = points_3d

    # Plot point cloud
    scatter = ax.scatter(points_sampled[:, 0],
                         points_sampled[:, 1],
                         points_sampled[:, 2],
                         c=points_sampled[:, 2],  # Color by Z value (depth)
                         cmap='viridis',
                         s=0.5,
                         alpha=0.6)

    # Set axis labels
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m) - Depth', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Depth (m)', fontsize=12)

    # Set equal aspect ratio for axes
    max_range = np.array([
        points_sampled[:, 0].max() - points_sampled[:, 0].min(),
        points_sampled[:, 1].max() - points_sampled[:, 1].min(),
        points_sampled[:, 2].max() - points_sampled[:, 2].min()
    ]).max() / 2.0

    mid_x = (points_sampled[:, 0].max() + points_sampled[:, 0].min()) * 0.5
    mid_y = (points_sampled[:, 1].max() + points_sampled[:, 1].min()) * 0.5
    mid_z = (points_sampled[:, 2].max() + points_sampled[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"Total points: {len(points_3d)}")
    print(f"X range: [{points_3d[:, 0].min():.2f}, {points_3d[:, 0].max():.2f}] meters")
    print(f"Y range: [{points_3d[:, 1].min():.2f}, {points_3d[:, 1].max():.2f}] meters")
    print(f"Z range (depth): [{points_3d[:, 2].min():.2f}, {points_3d[:, 2].max():.2f}] meters")


if __name__ == "__main__":
    # Front camera parameters
    cam_front = {
        "intra": [
            [1903.23710447121, 0.0, 1920.43709757232],
            [0.0, 1903.68240318844, 1084.34668921649],
            [0.0, 0.0, 1.0]
        ],
        "D": [-0.0303987666733171, -0.00386911425981525, 0.0, 0.0],
        "extra": [
            [-0.0009262916649869974, -0.9999617901002602, 0.008692544117497168, -0.0036524469925175314],
            [0.009675465947979188, -0.008701102930628682, -0.9999153344991167, 1.2981115648081694],
            [0.9999527625555323, -0.0008421088254290461, 0.009683156008516217, -1.9478534255216424],
            [0.0, 0.0, 0.0, 1.0]
        ]
    }

    image_shape = (3840, 2160)  # width, height
    max_depth = 25  # 0~25 m

    # frustum shape
    frustum_shape = (480, 270, 25)  # width, height, depth

    # Convert to numpy array
    camera_intrinsic = np.array(cam_front["intra"])
    print("Camera intrinsic matrix:")
    print(camera_intrinsic)

    # Generate frustum point cloud
    print("Generating frustum point cloud...")
    points_3d = generate_frustum_points(
        camera_intrinsic,
        image_width=image_shape[0],
        image_height=image_shape[1],
        max_depth=max_depth,  # 0 ~ 25m
        frustum_shape=frustum_shape,

    )
    print(f"Point cloud generation complete! Total {len(points_3d)} points\n")

    # Visualize
    print("Visualizing...")
    visualize_frustum(points_3d, title=f"Front Camera Frustum ({image_shape[0]}x{image_shape[1]}, 0-{max_depth}m)")
