
import os
import numpy as np
import pickle
import zarr
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import torch
import cv2
from tqdm import tqdm
import time

# Import point cloud processing functions
try:
    from convert_point_data import d2p, color_one_point_cloud, sample_one_point_cloud
except ImportError:
    print("Warning: convert_point_data.py not found. Some functionality may be limited.")
    
    def d2p(depth_img):
        raise NotImplementedError("d2p function not available")
    
    def color_one_point_cloud(point_cloud, color_image, num_points=1024, use_cuda=False):
        raise NotImplementedError("color_one_point_cloud function not available")
        
    def sample_one_point_cloud(point_cloud, color_image, num_points=1024, use_cuda=False):
        raise NotImplementedError("sample_one_point_cloud function not available")

def load_pkl_data(pkl_path):
    """Load data from PKL file"""
    print(f"Loading PKL file: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_zarr_data(zarr_path, point_cloud_key='point_cloud'):
    """Load point cloud data from ZARR file"""
    print(f"Loading ZARR file: {zarr_path}")
    root = zarr.open(zarr_path, mode='r')
    # support keys under 'data' group or top-level
    container = None
    if 'data' in root:
        container = root['data']
    else:
        container = root

    if point_cloud_key in container:
        point_cloud = container[point_cloud_key][:]
        
        # Print debug information
        print(f"Point cloud shape: {point_cloud.shape}")
        print(f"Point cloud dtype: {point_cloud.dtype}")

        # special-case name used previously
        if point_cloud_key == 'seg point cloud':
            if point_cloud.shape[-1] == 7:
                point_cloud = point_cloud[..., :6]
            elif point_cloud.shape[-1] == 4:
                point_cloud = point_cloud[..., :3]

        point_cloud = np.ascontiguousarray(point_cloud, dtype=np.float32)
        return point_cloud
    else:
        raise ValueError(f"Key '{point_cloud_key}' not found in ZARR file {zarr_path}")

def flip_point_cloud_coords(point_cloud):
    """Flip Y and Z coordinates (equivalent to 180° rotation around X axis)"""
    coords = np.copy(point_cloud[:, :3])
    # Flip Y and Z - equivalent to 180 degree rotation around X
    coords[:, 1] = -coords[:, 1]  # Flip Y
    coords[:, 2] = -coords[:, 2]  # Flip Z
    
    # Create new point cloud with flipped coordinates
    flipped_cloud = point_cloud.copy()
    flipped_cloud[:, :3] = coords
    return flipped_cloud

def apply_transform(point_cloud, scale=1.0, rotate_x=0, rotate_y=0, rotate_z=0, translate=[0, 0, 0]):
    """
    Apply scale, rotation, and translation transformations to the point cloud
    
    Args:
        point_cloud: Input point cloud (N, 3+) or (N, 6+)
        scale: Scale factor
        rotate_x, rotate_y, rotate_z: Rotation angles around X/Y/Z axes (degrees)
        translate: Translation vector [tx, ty, tz]
    
    Returns:
        transformed_cloud: Transformed point cloud
    """
    transformed_cloud = point_cloud.copy()
    coords = transformed_cloud[:, :3].copy()
    
    # 1. Scale
    coords = coords * scale
    
    # 2. Rotation (Using rotation matrices)
    # Convert angles to radians
    rx = np.radians(rotate_x)
    ry = np.radians(rotate_y)
    rz = np.radians(rotate_z)
    
    # Rotate around X axis
    if rotate_x != 0:
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        coords = coords @ Rx.T
    
    # Rotate around Y axis
    if rotate_y != 0:
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        coords = coords @ Ry.T
    
    # Rotate around Z axis
    if rotate_z != 0:
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        coords = coords @ Rz.T
    
    # 3. Translation
    coords = coords + np.array(translate)
    
    # Update coordinates
    transformed_cloud[:, :3] = coords
    
    return transformed_cloud

def visualize_point_cloud_open3d(point_cloud, window_name="Point Cloud Visualization", flip_axes=False,
                                 scale=1.0, rotate_x=0, rotate_y=0, rotate_z=0, translate=[0, 0, 0]):
    """Visualize point cloud using Open3D"""
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    
    # Print debug info
    print(f"Visualizing point cloud with shape {point_cloud.shape}")
    
    # Process 3D point cloud array (e.g. 660x750x6 shape)
    if len(point_cloud.shape) == 3 and point_cloud.shape[2] in [3, 6]:
        h, w, c = point_cloud.shape
        print(f"Reshaping organized point cloud from {point_cloud.shape} to ({h*w}, {c})")
        point_cloud = point_cloud.reshape(-1, c)
    # Handle batch dimension
    elif len(point_cloud.shape) == 3 and point_cloud.shape[0] == 1:
        # Handle batch dimension
        point_cloud = point_cloud[0]
    
    # Apply coordinate transformation if needed
    if flip_axes:
        point_cloud = flip_point_cloud_coords(point_cloud)
    
    # Apply custom transformations
    if scale != 1.0 or rotate_x != 0 or rotate_y != 0 or rotate_z != 0 or translate != [0, 0, 0]:
        point_cloud = apply_transform(point_cloud, scale, rotate_x, rotate_y, rotate_z, translate)
    
    # Ensure data is contiguous and float32
    coords = np.ascontiguousarray(point_cloud[:, :3], dtype=np.float32)
    
    # Set coordinates
    pcd.points = o3d.utility.Vector3dVector(coords)
    
    # If color information is available (RGB format, range 0-255)
    if point_cloud.shape[1] >= 6:
        colors = np.ascontiguousarray(point_cloud[:, 3:6] / 255.0, dtype=np.float32)  # Normalize to [0,1]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize
    o3d.visualization.draw_geometries([pcd], window_name=window_name)

def visualize_point_cloud_matplotlib(point_cloud, title="Point Cloud", flip_axes=False):
    """Visualize point cloud using Matplotlib"""
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Process 3D point cloud array (e.g. 660x750x6 shape)
        if len(point_cloud.shape) == 3 and point_cloud.shape[2] in [3, 6]:
            h, w, c = point_cloud.shape
            print(f"Reshaping organized point cloud from {point_cloud.shape} to ({h*w}, {c})")
            point_cloud = point_cloud.reshape(-1, c)
        # Handle batch dimension
        elif len(point_cloud.shape) == 3 and point_cloud.shape[0] == 1:
            point_cloud = point_cloud[0]
        
        # Apply coordinate transformation if needed
        if flip_axes:
            point_cloud = flip_point_cloud_coords(point_cloud)
        
        # Ensure data is contiguous float32
        point_cloud = np.ascontiguousarray(point_cloud, dtype=np.float32)
        
        # Downsample to speed up rendering
        max_points = 50000  # Maximum point count to display
        if len(point_cloud) > max_points:
            print(f"Downsampling from {len(point_cloud)} to {max_points} points for faster rendering")
            idx = np.random.choice(len(point_cloud), max_points, replace=False)
            point_cloud = point_cloud[idx]
        
        # Extract coordinates
        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2]
        
        # If color information is available
        if point_cloud.shape[1] >= 6:
            colors = point_cloud[:, 3:6] / 255.0  # Normalize to [0,1]
            colors = np.clip(colors, 0, 1)  # Ensure valid range
        else:
            colors = 'blue'
        
        # Plot point cloud
        ax.scatter(x, y, z, c=colors, marker='.', s=1)
        
        # Set axis labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # Display figure
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)  # Small pause to render the plot
        
        return fig
    except Exception as e:
        print(f"Error in matplotlib visualization: {e}")
        fig = plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', color='red')
        plt.title("Error in visualization")
        plt.show(block=False)
        plt.pause(0.1)
        return fig

def browse_point_clouds_matplotlib(point_clouds, flip_axes=False):
    """Browse point clouds using Matplotlib (with working key navigation)"""
    if len(point_clouds) == 0:
        print("No point clouds to display")
        return
    
    current_idx = 0
    num_clouds = len(point_clouds)
    
    # Buffer for jump to step functionality
    input_buffer = ""
    # Flag to indicate user wants to exit
    exit_requested = False
    
    print(f"\nKey Guide:")
    print(f"  'A' - Previous point cloud")
    print(f"  'D' - Next point cloud")
    print(f"  'Q' - Exit")
    print(f"  'Esc' - Clear input (if typing) or Exit (if no input)")
    print(f"  '0-9' + 'J' - Jump to specific step (enter numbers then press J)")
    print(f"  Example: Type '245' then 'J' to jump to step 245\n")
    
    # Display the first point cloud
    fig = visualize_point_cloud_matplotlib(point_clouds[current_idx], 
                                          f"Point Cloud {current_idx+1}/{num_clouds}",
                                          flip_axes=flip_axes)
    
    # Set up keyboard event handling
    def on_key(event):
        nonlocal current_idx, input_buffer, exit_requested
        
        # Check if key is a number (for jump functionality)
        if event.key in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            input_buffer += event.key
            print(f"Input: {input_buffer}", end='\r')
            return True
            
        # Jump to specific step
        if event.key == 'j' and input_buffer:
            try:
                step = int(input_buffer)
                if 0 <= step < num_clouds:
                    current_idx = step
                    print(f"Jumping to step {step}")
                    plt.close(fig)
                else:
                    print(f"Step {step} out of range (0-{num_clouds-1})")
                input_buffer = ""  # Clear buffer
            except ValueError:
                print(f"Invalid step number: {input_buffer}")
                input_buffer = ""  # Clear buffer
            return True
        
        # Clear input buffer on escape or exit if no input
        if event.key == 'escape':
            if input_buffer:
                input_buffer = ""
                print("Input cleared           ", end='\r')
            else:
                print("Exiting visualization")
                exit_requested = True
                plt.close(fig)
                return False
            return True
        
        if event.key == 'a':
            current_idx = (current_idx - 1) % num_clouds
            plt.close(fig)
        elif event.key == 'd':
            current_idx = (current_idx + 1) % num_clouds
            plt.close(fig)
        elif event.key == 'q':
            print("Exiting visualization")
            exit_requested = True
            plt.close(fig)
            return False
        
        return True
    
    # Connect the event handler
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Main loop
    while plt.fignum_exists(fig.number) and not exit_requested:
        plt.pause(0.1)
        
        # If the window was closed
        if not plt.fignum_exists(fig.number) and not exit_requested:
            # Check if we need to open a new window (navigation)
            if 0 <= current_idx < num_clouds:
                print(f"Displaying point cloud {current_idx+1}/{num_clouds}")
                fig = visualize_point_cloud_matplotlib(point_clouds[current_idx], 
                                                     f"Point Cloud {current_idx+1}/{num_clouds}",
                                                     flip_axes=flip_axes)
                fig.canvas.mpl_connect('key_press_event', on_key)
            else:
                break
    
    print("Visualization ended")

def create_custom_open3d_window(point_clouds, flip_axes=False, scale=1.0, 
                                rotate_x=0, rotate_y=0, rotate_z=0, translate=[0, 0, 0]):
    """Create a custom Open3D window with keyboard navigation"""
    if len(point_clouds) == 0:
        print("No point clouds to display")
        return
    
    current_idx = 0
    num_clouds = len(point_clouds)
    
    # Buffer for jump to step functionality
    input_buffer = ""
    
    # Create a custom visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    
    # Function to update point cloud
    def update_point_cloud(idx):
        vis.clear_geometries()
        
        point_cloud = point_clouds[idx]
        
        # Process 3D point cloud array
        if len(point_cloud.shape) == 3 and point_cloud.shape[2] in [3, 6]:
            h, w, c = point_cloud.shape
            print(f"Reshaping organized point cloud from {point_cloud.shape} to ({h*w}, {c})")
            point_cloud = point_cloud.reshape(-1, c)
        elif len(point_cloud.shape) == 3 and point_cloud.shape[0] == 1:
            point_cloud = point_cloud[0]
        
        # Apply coordinate transformation if needed
        if flip_axes:
            point_cloud = flip_point_cloud_coords(point_cloud)
        
        # Apply custom transformations
        if scale != 1.0 or rotate_x != 0 or rotate_y != 0 or rotate_z != 0 or translate != [0, 0, 0]:
            point_cloud = apply_transform(point_cloud, scale, rotate_x, rotate_y, rotate_z, translate)
        
        # Ensure point cloud is properly formatted for Open3D
        try:
            # Ensure data is contiguous and float32
            coords = np.ascontiguousarray(point_cloud[:, :3], dtype=np.float32)
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coords)
            
            if point_cloud.shape[1] >= 6:
                colors = np.ascontiguousarray(point_cloud[:, 3:6] / 255.0, dtype=np.float32)
                pcd.colors = o3d.utility.Vector3dVector(colors)
            
            vis.add_geometry(pcd)
            vis.update_renderer()
            
            # Update window title (not directly supported in Open3D)
            print(f"Displaying point cloud {idx+1}/{num_clouds}")
        except Exception as e:
            print(f"Error updating point cloud: {e}")
            print(f"Point cloud first few rows:\n{point_cloud[:5]}")
            
            # Fall back to simplified point cloud if there's an error
            try:
                # Last resort: simpler method
                flat_pc = point_cloud.reshape(-1, point_cloud.shape[-1]) if len(point_cloud.shape) > 2 else point_cloud
                simple_coords = np.asarray(flat_pc[:100, :3], dtype=np.float32)
                simple_pcd = o3d.geometry.PointCloud()
                simple_pcd.points = o3d.utility.Vector3dVector(simple_coords)
                vis.add_geometry(simple_pcd)
                vis.update_renderer()
                print("Displaying simplified point cloud (first 100 points only)")
            except Exception as e2:
                print(f"Failed to display even simplified point cloud: {e2}")
        
    # Key callbacks
    def prev_cloud(vis):
        nonlocal current_idx
        current_idx = (current_idx - 1) % num_clouds
        update_point_cloud(current_idx)
        return False
    
    def next_cloud(vis):
        nonlocal current_idx
        current_idx = (current_idx + 1) % num_clouds
        update_point_cloud(current_idx)
        return False
    
    # Number key callbacks (0-9)
    def handle_number_key(vis, key):
        nonlocal input_buffer
        # Convert ASCII code to number string
        num_str = chr(key)
        input_buffer += num_str
        print(f"Input: {input_buffer}", end='\r')
        return False
    
    # Jump to step callback
    def jump_to_step(vis):
        nonlocal current_idx, input_buffer
        if input_buffer:
            try:
                step = int(input_buffer)
                if 0 <= step < num_clouds:
                    current_idx = step
                    print(f"\nJumping to step {step}")
                    update_point_cloud(current_idx)
                else:
                    print(f"\nStep {step} out of range (0-{num_clouds-1})")
                input_buffer = ""  # Clear buffer
            except ValueError:
                print(f"\nInvalid step number: {input_buffer}")
                input_buffer = ""  # Clear buffer
        return False
    
    # Clear input buffer
    def clear_input(vis):
        nonlocal input_buffer
        if input_buffer:
            input_buffer = ""
            print("Input cleared           ", end='\r')
        else:
            # If no input buffer, ESC key closes window
            vis.close()
        return False
    
    # Register key callbacks
    vis.register_key_callback(65, prev_cloud)  # 'A' key
    vis.register_key_callback(68, next_cloud)  # 'D' key
    vis.register_key_callback(74, jump_to_step)  # 'J' key
    vis.register_key_callback(27, clear_input)  # ESC key
    vis.register_key_callback(81, lambda vis: vis.close())  # 'Q' key to exit
    
    # Register number keys (0-9)
    for i in range(10):
        vis.register_key_callback(48 + i, lambda vis, key=48+i: handle_number_key(vis, key))
    
    print(f"\nKey Guide:")
    print(f"  'A' - Previous point cloud")
    print(f"  'D' - Next point cloud")
    print(f"  'Q' - Exit")
    print(f"  'Esc' - Clear input (if typing) or Exit (if no input)")
    print(f"  'Any numbers' + 'J' - Jump to specific step (type numbers then press J)")
    print(f"  Example: Type '245' then 'J' to jump to step 245\n")
    
    # Display first point cloud
    update_point_cloud(current_idx)
    
    # Run the visualization
    vis.run()
    vis.destroy_window()

def browse_point_clouds(point_clouds, visualize_func=visualize_point_cloud_open3d, flip_axes=False,
                       scale=1.0, rotate_x=0, rotate_y=0, rotate_z=0, translate=[0, 0, 0]):
    """Interactive interface for browsing multiple point clouds"""
    # For zarr files, point_clouds might be a 3D array instead of a list
    if isinstance(point_clouds, np.ndarray) and len(point_clouds.shape) == 3:
        print(f"Converting 3D array to list of {point_clouds.shape[0]} point clouds")
        # Convert to list of point clouds
        point_clouds_list = [point_clouds[i] for i in range(point_clouds.shape[0])]
        point_clouds = point_clouds_list
    
    # Data compatibility check
    if isinstance(point_clouds, list) and len(point_clouds) > 0:
        # Check formatting of first point cloud
        first_pc = point_clouds[0]
        print(f"First point cloud - type: {type(first_pc)}, shape: {first_pc.shape}, dtype: {first_pc.dtype}")
        
        # Ensure all point clouds are float32
        for i in range(len(point_clouds)):
            if point_clouds[i].dtype != np.float32:
                print(f"Converting point cloud {i} to float32")
                point_clouds[i] = np.ascontiguousarray(point_clouds[i], dtype=np.float32)
    
    # Check type and choose appropriate browser
    if isinstance(point_clouds, list):
        num_clouds = len(point_clouds)
        if num_clouds == 0:
            print("No point clouds to display")
            return
        
        # Use the custom Open3D window for better interaction
        if visualize_func == visualize_point_cloud_open3d:
            create_custom_open3d_window(point_clouds, flip_axes=flip_axes, 
                                       scale=scale, rotate_x=rotate_x, 
                                       rotate_y=rotate_y, rotate_z=rotate_z, 
                                       translate=translate)
        else:
            browse_point_clouds_matplotlib(point_clouds, flip_axes=flip_axes)
    else:
        # Single point cloud case
        try:
            # Try to directly visualize
            visualize_func(point_clouds, flip_axes=flip_axes, scale=scale,
                         rotate_x=rotate_x, rotate_y=rotate_y, 
                         rotate_z=rotate_z, translate=translate)
        except Exception as e:
            print(f"Error visualizing point cloud: {e}")
            print("Attempting to convert data format...")
            
            # Try to convert to a compatible format
            if isinstance(point_clouds, np.ndarray):
                # If it's a single point cloud
                if len(point_clouds.shape) == 2:
                    point_clouds = np.ascontiguousarray(point_clouds, dtype=np.float32)
                    visualize_func(point_clouds, flip_axes=flip_axes, scale=scale,
                                 rotate_x=rotate_x, rotate_y=rotate_y, 
                                 rotate_z=rotate_z, translate=translate)
                # If it's a batch of point clouds
                elif len(point_clouds.shape) == 3:
                    print(f"Visualizing first point cloud from batch of {point_clouds.shape[0]}")
                    cloud = np.ascontiguousarray(point_clouds[0], dtype=np.float32)
                    visualize_func(cloud, flip_axes=flip_axes, scale=scale,
                                 rotate_x=rotate_x, rotate_y=rotate_y, 
                                 rotate_z=rotate_z, translate=translate)
                else:
                    print(f"Unsupported point cloud shape: {point_clouds.shape}")
            else:
                print(f"Unsupported point cloud type: {type(point_clouds)}")

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Visualization Tool')
    parser.add_argument('file_path', help='Path to PKL or ZARR file')
    parser.add_argument('--vis', choices=['o3d', 'mpl'], default='o3d', 
                        help='Visualization method: o3d (Open3D) or mpl (Matplotlib)')
    parser.add_argument('--num_points', type=int, default=1024, 
                        help='Number of points to sample (applies only with --sample)')
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Use CUDA for point cloud processing')
    parser.add_argument('--step', type=int, default=None,
                        help='Specific timestep to display (default: show all)')
    parser.add_argument('--flip', action='store_true',
                        help='Flip point cloud coordinates (rotate 180° around X axis)')
    parser.add_argument('--sample', action='store_true',
                        help='For PKL files: Sample points (default: use original full point cloud)')
    parser.add_argument('--cloud_type', type=str, default='point_cloud',
                        help='Point cloud dictionary key name to read (e.g. "segment_10" or "seg_point_cloud").')
    
    # Transformation arguments
    parser.add_argument('--scale', type=float, default=0.2,
                        help='Scale factor for point cloud (default: 1.0)')
    parser.add_argument('--rotate_x', type=float, default=180,
                        help='Rotation angle around X axis in degrees (default: 0)')
    parser.add_argument('--rotate_y', type=float, default=0,
                        help='Rotation angle around Y axis in degrees (default: 0)')
    parser.add_argument('--rotate_z', type=float, default=275,
                        help='Rotation angle around Z axis in degrees (default: 0)')
    parser.add_argument('--translate', type=float, nargs=3, default=[0, 0, 0],
                        metavar=('TX', 'TY', 'TZ'),
                        help='Translation vector [tx, ty, tz] (default: [0, 0, 0])')
    
    args = parser.parse_args()
    
    # Choose visualization method
    if args.vis == 'o3d':
        vis_func = visualize_point_cloud_open3d
    else:
        vis_func = visualize_point_cloud_matplotlib
    
    # treat cloud_type as the actual dictionary key name
    cloud_key = args.cloud_type
    print(f"Selected cloud key: '{cloud_key}'")

    # Load data based on file type
    if args.file_path.endswith('.pkl'):
        data = load_pkl_data(args.file_path)

        # If the requested key exists and is non-empty, use it
        if cloud_key in data:
            candidate = data[cloud_key]
            # Accept list/ndarray-like containers
            if isinstance(candidate, (list, tuple, np.ndarray)) and len(candidate) > 0:
                point_clouds = candidate
                print(f"Read {len(point_clouds)} point clouds from PKL file using key '{cloud_key}'")
            else:
                raise ValueError(f"Key '{cloud_key}' found in PKL but is empty or not a list/ndarray. Type: {type(candidate)}")
        else:
            available = [k for k,v in data.items() if isinstance(v, (list, tuple, np.ndarray))]
            raise ValueError(f"Key '{cloud_key}' not found in PKL. Available array/list keys: {available}")

        # If point_clouds is not provided, user might have depth/color and want generation (kept for compatibility)
        if ('point_cloud' not in data) and ('depth' in data and 'color' in data) and (not isinstance(point_clouds, (list, np.ndarray))):
            print("Generating point clouds from depth and color images...")
            if 'depth' not in data or 'color' not in data:
                raise ValueError("Depth or color data missing in PKL file")
            
            point_clouds = []
            for i in tqdm(range(len(data['depth']))):
                if args.step is not None and i != args.step:
                    continue
                    
                try:
                    # Convert depth image to point cloud
                    point_cloud = d2p(data['depth'][i])
                    
                    if args.sample:
                        processed_pc = sample_one_point_cloud(
                            point_cloud, 
                            data['color'][i], 
                            num_points=args.num_points, 
                            use_cuda=args.use_cuda
                        )
                    else:
                        processed_pc = color_one_point_cloud(
                            point_cloud,
                            data['color'][i],
                            use_cuda=args.use_cuda
                        )
                    
                    point_clouds.append(processed_pc)
                except Exception as e:
                    print(f"Error processing point cloud {i}: {e}")
    
    elif args.file_path.endswith('.zarr'):
        # Load point cloud directly from ZARR file using the specified key
        try:
            point_clouds = load_zarr_data(args.file_path, cloud_key)
            print(f"Read point clouds from ZARR key '{cloud_key}'")
        except ValueError as e:
            root = zarr.open(args.file_path, mode='r')
            data_keys = list(root['data'].keys()) if 'data' in root else list(root.keys())
            raise ValueError(f"{e}\nAvailable keys in ZARR: {data_keys}")
        
        # If specific step is specified
        if args.step is not None:
            if 0 <= args.step < len(point_clouds):
                point_clouds = point_clouds[args.step:args.step+1]  # Keep as array but with one element
                print(f"Selected point cloud at step {args.step}")
            else:
                raise ValueError(f"Specified step {args.step} out of range (0-{len(point_clouds)-1})")
    else:
        raise ValueError("Unsupported file type, please use .pkl or .zarr files")
    
    # Visualize point cloud
    if isinstance(point_clouds, list) and len(point_clouds) == 1:
        vis_func(point_clouds[0], flip_axes=args.flip, scale=args.scale,
                rotate_x=args.rotate_x, rotate_y=args.rotate_y, 
                rotate_z=args.rotate_z, translate=args.translate)
    else:
        browse_point_clouds(point_clouds, vis_func, flip_axes=args.flip,
                          scale=args.scale, rotate_x=args.rotate_x,
                          rotate_y=args.rotate_y, rotate_z=args.rotate_z,
                          translate=args.translate)

if __name__ == "__main__":
    main()
    
    '''
    # Example usage:
    # View point cloud in normal coordinate system
    python view_point2.py path/to/your/file.zarr --vis o3d --cloud_type fps --flip
    
    # View flipped point cloud rotated 180 degrees around X-axis
    python view_point2.py path/to/your/file.zarr --vis o3d --flip --cloud_type fps
    
    # View original full point cloud generated from depth and color images (PKL file)
    python view_point2.py path/to/your/file.pkl --vis o3d 
    
    # View sampled point cloud (PKL file)
    python view_point2.py path/to/your/file.pkl --vis o3d --sample --num_points 2048 
    
    # Use Matplotlib to view flipped point cloud at specific timestep
    python view_point2.py path/to/your/file.zarr --vis mpl --step 10 --flip 
    '''