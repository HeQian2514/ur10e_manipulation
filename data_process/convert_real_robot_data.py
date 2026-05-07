import os
import cv2
import zarr
import pickle
import tqdm
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import torchvision
from termcolor import cprint
import re
import time

import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import torchvision
import socket
import pickle
import scipy.spatial.transform as st



def decimal_places(x):
    s = str(x)
    if '.' in s:
        return len(s.split('.')[-1].rstrip('0'))
    return 0

if __name__ == "__main__":
    from convert_point_data import d2p, sample_one_point_cloud
    from clear_data import clean_demo_data
    from fastsam2_point_final_sasa import dw_fps_sampling
    from convert_zarr2excel import zarr_to_excel
else:
    from data_process.convert_point_data import d2p, sample_one_point_cloud
    from data_process.clear_data import clean_demo_data
    
    from data_process.convert_zarr2excel import zarr_to_excel



def state2smooth_rotation_vector(rot_vec):
    """
    Convert rotation vector via Euler angles to a smooth rotation vector to avoid jumps
    
    Args:
    rot_vec: numpy array of shape (6,) or (n,6) containing position and rotation vectors
    
    Returns:
    Smoothed pose vector
    """
    result = np.array(rot_vec).copy()
    
    # Only process rotation part (3:6)
    if rot_vec.ndim == 1:
        # Single pose vector
        r = st.Rotation.from_rotvec(rot_vec[3:6])
        # Convert to Euler angles (ZYX order)
        euler = r.as_euler('zyx')
        euler[2] += np.pi  # Avoid singularities
        # Normalize to [-π, π]
        euler_norm = ((euler + np.pi) % (2 * np.pi)) - np.pi
        # Convert back to rotation vector
        result[3:6] = st.Rotation.from_euler('zyx', euler_norm).as_rotvec()
    else:
        # Batch pose vectors
        r = st.Rotation.from_rotvec(rot_vec[:, 3:6])
        euler = r.as_euler('zyx')
        euler_norm = ((euler + np.pi) % (2 * np.pi)) - np.pi
        result[:, 3:6] = st.Rotation.from_euler('zyx', euler_norm).as_rotvec()
    
    return result

def smooth_rotation_vector2state(rot_vec):
    """
    Convert rotation vector via Euler angles to a smooth rotation vector to avoid jumps
    
    Args:
    rot_vec: numpy array of shape (6,) or (n,6) containing position and rotation vectors
    
    Returns:
    Smoothed pose vector
    """
    result = np.array(rot_vec).copy()
    
    # Only process rotation part (3:6)
    if rot_vec.ndim == 1:
        # Single pose vector
        r = st.Rotation.from_rotvec(rot_vec[3:6])
        # Convert to Euler angles (ZYX order)
        euler = r.as_euler('zyx')
        euler[2] -= np.pi  # Avoid singularities
        # Normalize to [-π, π]
        euler_norm = ((euler + np.pi) % (2 * np.pi)) - np.pi
        # Convert back to rotation vector
        result[3:6] = st.Rotation.from_euler('zyx', euler_norm).as_rotvec()
    else:
        # Batch pose vectors
        r = st.Rotation.from_rotvec(rot_vec[:, 3:6])
        euler = r.as_euler('zyx')
        euler[:, 2] -= np.pi  # Avoid singularities
        euler_norm = ((euler + np.pi) % (2 * np.pi)) - np.pi
        result[:, 3:6] = st.Rotation.from_euler('zyx', euler_norm).as_rotvec()
    
    return result

def smooth_positions(positions, window_size=3):
    """
    Apply moving average filter to position data
    
    Args:
        positions: position data (N, 3) or (N, 6)
        window_size: sliding window size
    
    Returns:
        smoothed_positions: smoothed positions
    """
    if len(positions) < window_size:
        return positions
    
    smoothed = np.copy(positions)
    for i in range(len(positions)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(positions), i + window_size // 2 + 1)
        smoothed[i] = np.mean(positions[start_idx:end_idx], axis=0)
    
    return smoothed

def preproces_image(image):
    img_size = 84
    
    image = image.astype(np.float32)
    image = torch.from_numpy(image).cuda()
    image = image.permute(2, 0, 1) # HxWx4 -> 4xHxW
    image = torchvision.transforms.functional.resize(image, (img_size, img_size))
    image = image.permute(1, 2, 0) # 4xHxW -> HxWx4
    image = image.cpu().numpy()
    return image

if __name__ == "__main__":
    USE_ROTATION = True 
    DELTA_ACTION = False
    DELTA_ROTATION = False
    UNIT_SCALE = 1000.0  # Convert meters to millimeters
    SKIP = 6  # Only save data every SKIP frames
    crop_size = 'large' # middle small large
    PC = True  # Whether to save point cloud data
    ST = 3  # SKIP_THRESHOLD
    ROTATION_ST = 0.1  # Rotation skip frame threshold (radians)
    pc_number = 1024
    pose_threshold = 1e-4

    task_name = 'pot_loading' # 'stacking_cubes' 'objects_collection' 'cups_shifting' 'water_pouring' 'drawer_loading'
    expert_data_paths = [
        'data/expert_demos/{}'.format(task_name),
    ]

    print('clear data path:', expert_data_paths)
    print('Clearing pkl data begins...')
    for expert_data_path in expert_data_paths:    
        # Clean data, remove consecutive frames with identical pose and gripper. For data integrity, it's best to clean only once. If already cleaned, comment out this for loop
        if os.path.exists(expert_data_path):
            print('Cleaning data in path:', expert_data_path)
            clean_demo_data(expert_data_path, pose_threshold=pose_threshold)
    print('Clearing pkl data ends.')
    
    if DELTA_ACTION:
        if DELTA_ROTATION:
            info = f'_SCL{int(UNIT_SCALE)}_SKP{SKIP}_ST{int(ST)}_100RST{int(100*ROTATION_ST)}_DX_DR_{int(PC)*pc_number}_PT{decimal_places(pose_threshold)}_{crop_size}'
        else:
            info = f'_SCL{int(UNIT_SCALE)}_SKP{SKIP}_ST{int(ST)}_100RST{int(100*ROTATION_ST)}_DX_{int(PC)*pc_number}_PT{decimal_places(pose_threshold)}_{crop_size}'
    else:
        if DELTA_ROTATION:
            info = f'_SCL{int(UNIT_SCALE)}_SKP{SKIP}_ST{int(ST)}_100RST{int(100*ROTATION_ST)}_DR_{int(PC)*pc_number}_PT{decimal_places(pose_threshold)}_{crop_size}'
        else:
            info = f'_SCL{int(UNIT_SCALE)}_SKP{SKIP}_ST{int(ST)}_100RST{int(100*ROTATION_ST)}_{int(PC)*pc_number}_PT{decimal_places(pose_threshold)}_{crop_size}'


    cprint('Converting pkl data to zarr format...', 'red')
    cprint('Delta action: {}'.format(DELTA_ACTION), 'red')
    demo_dirs = []
    for expert_data_path in expert_data_paths:
        if os.path.exists(expert_data_path):
            path_demos = [os.path.join(expert_data_path, f) 
                         for f in sorted(os.listdir(expert_data_path)) 
                         if f.endswith('.pkl')]
            demo_dirs.extend(path_demos)
            cprint(f'Found {len(path_demos)} demos in: {expert_data_path}', 'cyan')
        else:
            cprint(f'Warning: Path does not exist: {expert_data_path}', 'yellow')
    cprint(f'Total demos found: {len(demo_dirs)}', 'cyan')
    if len(demo_dirs) == 0:
        cprint('Error: No demo files found!', 'red')
        exit(1)

    demo_learn = demo_dirs
    demo_test = demo_dirs[-1:]
    # demo_val = demo_dirs[:10]
    demo_val = demo_dirs[:1]

    save_base_path = next((path for path in expert_data_paths if os.path.exists(path)), expert_data_paths[0])
    
    for name, demo_set in zip(['learn', 'test', 'val'], [demo_learn, demo_test, demo_val]):

        id = f'{name}.zarr'
        save_data_info = os.path.join(save_base_path, task_name + info)
        save_data_path = os.path.join(save_data_info, id)
        cprint('Save data path: {}'.format(save_data_path), 'red')

        if os.path.exists(save_data_path):
            cprint('Data already exists at {}'.format(save_data_path), 'red')
            cprint("If you want to overwrite, delete the existing directory first.", "red")
            cprint("Do you want to overwrite? (y/n)", "red")
            user_input = 'y'
            if user_input == 'y':
                cprint('Overwriting {}'.format(save_data_path), 'red')
                os.system('rm -rf {}'.format(save_data_path))
            else:
                cprint('Exiting', 'red')
                exit()
        os.makedirs(save_data_path, exist_ok=True)

        # storage
        total_count = 0
        img_arrays = []
        fps_point_cloud_arrays = []
        seg_point_cloud_arrays = []
        depth_arrays = []
        state_arrays = []
        action_arrays = []
        if DELTA_ACTION:
            action_array = []
        episode_ends_arrays = []

        for demo_idx, demo_dir in enumerate(demo_set):
            seg_reset = True
            save_mask_visualizations = f'/home/hq/PROJECT/FlowPolicy/real_world/check_pkg/DW_FPS_IMAGE/{task_name}/{name}_demo_{demo_idx}'
            dir_name = os.path.dirname(demo_dir)

            cprint('Processing {}'.format(demo_dir), 'green')
            with open(demo_dir, 'rb') as f:
                demo = pickle.load(f)

            demo['gripper'] = [np.array(d/100.) for d in demo['gripper']]  # normalize to [0, 1]


            if False:
                # ===== Process first frame =====
                cprint(f'[{demo_idx + 1}/{len(demo_set)}] Processing first frame', 'cyan')
                try:
                    first_frame_depth = demo['depth'][0]
                    first_frame_color = demo['color'][0]
                    point_cloud = d2p(first_frame_depth)
                    
                    seg_obs_pointcloud = dw_fps_sampling(
                        point_cloud, 
                        first_frame_color, 
                        total_sample_points=pc_number, 
                        crop_scale=crop_size, 
                        reset=True,
                        iou_conf=[[0.5, 0.1], [0.5, 0.1]],
                        save_mask_visualizations=f'/home/hq/PROJECT/FlowPolicy/real_world/check_pkg/DW_FPS_IMAGE/{task_name}/first_frame'
                    )
                    cprint(f'First frame done: shape {seg_obs_pointcloud.shape}', 'cyan')
                except Exception as e:
                    cprint(f'Error processing first frame: {e}', 'red')

            else:

                # Smooth rotation vectors in pose to avoid jumps
                for i in range(0, len(demo['pose'])):
                    demo['pose'][i] = state2smooth_rotation_vector(demo['pose'][i])
                    
                demo_length = len(demo['depth'])

                demo['fps_point_cloud'] = []
                demo['seg_point_cloud'] = []

                # for step_idx in tqdm.tqdm(range(demo_length), desc=f'Processing {dir_name} point cloud'):
                    
                #     point_cloud = d2p(demo['depth'][step_idx])
                #     fps_sample_point_cloud = sample_one_point_cloud(point_cloud, demo['color'][step_idx], num_points=1024, use_cuda=True)
                #     demo['fps_point_cloud'].append(fps_sample_point_cloud)

                #     seg_sample_point_cloud = dw_fps_sampling(point_cloud, demo['color'][step_idx], total_sample_points=1024)
                #     demo['seg_point_cloud'].append(seg_sample_point_cloud)

                #     if 'color' not in demo or 'depth' not in demo:
                #         raise ValueError("Demo data must contain 'color' and 'depth' keys")

                # Smooth the entire trajectory before saving
                # demo['pose'] = smooth_positions(demo['pose'], window_size=5)
                
                skip_idx = []
                last_velocity = np.zeros(3)  # Initialize velocity of previous frame
                last_rotation = np.zeros(3)  # Initialize rotation vector of previous frame

                for step_idx in tqdm.tqdm(range(demo_length)):

                    scaled_action = demo['pose'][step_idx].copy()
                    scaled_action[:3] = scaled_action[:3] * UNIT_SCALE  # scale position
                    scaled_action[:3] = np.round(scaled_action[:3], 0)
                    if USE_ROTATION:
                        scaled_action[3:6] = np.round(scaled_action[3:6], 3)  # Keep 3 decimal places                
                
                    if step_idx > 0:
                        delta = np.abs(scaled_action[:3] - state_arrays[-1][:3])
                        
                        # Calculate current velocity (position change)
                        current_velocity = scaled_action[:3] - state_arrays[-1][:3]

                        # ===== New: calculate rotation change =====
                        if USE_ROTATION:
                            current_rotation = scaled_action[3:6]
                            last_state_rotation = state_arrays[-1][3:6]
                            # Calculate rotation vector difference
                            rotation_delta = np.linalg.norm(current_rotation - last_state_rotation)
                        else:
                            rotation_delta = 0.0

                        # Check if gripper state changed
                        gripper_changed = demo['gripper'][step_idx] != demo['gripper'][step_idx - 1]
                        
                        # Check for change from stationary to moving
                        # If velocity in any direction was 0 and now isn't, detect start
                        start_moving = np.any((np.abs(last_velocity) < 0.01) & (np.abs(current_velocity) >= 0.01))
                        
                        # Check if velocity direction is reversed
                        # When both velocities are non-zero, if signs are opposite, direction changed
                        velocity_reversed = np.any(
                            (np.abs(last_velocity) >= 0.01) &  # Previous frame was moving
                            (np.abs(current_velocity) >= 0.01) &  # Current frame is moving
                            (np.sign(last_velocity) != np.sign(current_velocity))  # Directions are opposite
                        )

                        rotation_changed = rotation_delta >= ROTATION_ST if USE_ROTATION else False
                        
                    else:
                        delta = np.array([ST, ST, ST])
                        current_velocity = np.zeros(3)
                        gripper_changed = False
                        start_moving = False
                        velocity_reversed = False
                        rotation_changed = False
                        rotation_delta = 0.0

                    # Skip frame condition judgment
                    if len(skip_idx) < SKIP and \
                        step_idx != demo_length - 1 and \
                        np.all(delta < ST) and \
                        not gripper_changed and \
                        not start_moving and \
                        not velocity_reversed and \
                        not rotation_changed:  # New: rotation cannot exceed threshold
                        
                        skip_idx.append(step_idx)
                        continue


                    else:
                        # Save current frame
                        scaled_action = scaled_action[:3] if not USE_ROTATION else scaled_action
                        robot_state = np.append(scaled_action, demo['gripper'][step_idx])
                        
                        total_count += 1
                        obs_image = demo['color'][step_idx]
                        obs_depth = demo['depth'][step_idx]

                        if False:
                            img = demo['color'][step_idx]
                            if img.dtype != np.uint8:
                                img = (img * 255).clip(0, 255).astype(np.uint8)
                            img_path = '/home/hq/PROJECT/FlowPolicy/real_world/data/debug_data/test_pc_images.jpg'
                            cv2.imwrite(img_path, img)
                            print(f"Sampled point cloud corresponding image saved to: {img_path}")

                        if PC:
                            point_cloud = d2p(demo['depth'][step_idx])
                            fps_obs_pointcloud = sample_one_point_cloud(point_cloud, demo['color'][step_idx], num_points=pc_number, use_cuda=True, crop_scale=crop_size)

                            if False:
                                from data_process.fastsam2_point_final_sasa import dw_fps_sampling
                                seg_obs_pointcloud = dw_fps_sampling(point_cloud, demo['color'][step_idx], total_sample_points=pc_number, crop_scale=crop_size, reset =seg_reset,
                                                                    iou_conf=[[0.5, 0.1], [0.5, 0.1]],
                                                                    save_mask_visualizations=save_mask_visualizations)
                            seg_reset = False
                        else:
                            fps_obs_pointcloud = np.zeros((pc_number, 3), dtype=np.float32)
                            seg_obs_pointcloud = np.zeros((pc_number, 3), dtype=np.float32)
                        
                        obs_image = preproces_image(obs_image)
                        obs_depth = preproces_image(np.expand_dims(obs_depth, axis=-1)).squeeze(-1)

                        if DELTA_ACTION:
                            if step_idx == 0:
                                last_delta_action = np.zeros_like(robot_state)
                                last_delta_action[-1] = robot_state[-1]  # gripper command
                            else:
                                last_delta_action = robot_state - state_arrays[-1]
                                if not DELTA_ROTATION and USE_ROTATION:
                                    last_delta_action[3:6] = state_arrays[-1][3:6]
                                last_delta_action[-1] = robot_state[-1]  # gripper command
                            action_array.append(last_delta_action)
                        else:
                            action = robot_state
                            action_arrays.append(action)
                    
                        img_arrays.append(obs_image)
                        fps_point_cloud_arrays.append(fps_obs_pointcloud)
                        seg_point_cloud_arrays.append(seg_obs_pointcloud)
                        depth_arrays.append(obs_depth)
                        state_arrays.append(robot_state)
                        
                        # Update previous frame velocity
                        last_velocity = current_velocity.copy()
                        if USE_ROTATION:
                            last_rotation = scaled_action[3:6].copy()
                        
                        skip_idx = []

                if DELTA_ACTION:
                    # Except gripper state, all action differences move forward by one
                    action_array.append(action_array[-1])
                    action_array = action_array[1:]  # remove the first extra action
                    action_array[-1][-1] = action_array[-2][-1]

                    action_arrays.extend(action_array)
                    action_array = []

                episode_ends_arrays.append(total_count)

        # create zarr file
        zarr_root = zarr.group(save_data_path)
        zarr_data = zarr_root.create_group('data')
        zarr_meta = zarr_root.create_group('meta')

        img_arrays = np.stack(img_arrays, axis=0)
        if img_arrays.shape[1] == 3: # make channel last
            img_arrays = np.transpose(img_arrays, (0,2,3,1))
        fps_point_cloud_arrays = np.stack(fps_point_cloud_arrays, axis=0)
        seg_point_cloud_arrays = np.stack(seg_point_cloud_arrays, axis=0)
        depth_arrays = np.stack(depth_arrays, axis=0)
        action_arrays = np.stack(action_arrays, axis=0)
        state_arrays = np.stack(state_arrays, axis=0)
        episode_ends_arrays = np.array(episode_ends_arrays)

        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
        img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
        fps_point_cloud_chunk_size = (100, fps_point_cloud_arrays.shape[1], fps_point_cloud_arrays.shape[2])
        seg_point_cloud_chunk_size = (100, seg_point_cloud_arrays.shape[1], seg_point_cloud_arrays.shape[2])
        depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
        if len(action_arrays.shape) == 2:
            action_chunk_size = (100, action_arrays.shape[1])
        elif len(action_arrays.shape) == 3:
            action_chunk_size = (100, action_arrays.shape[1], action_arrays.shape[2])
        else:
            raise NotImplementedError
        zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('fps_point_cloud', data=fps_point_cloud_arrays, chunks=fps_point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('seg_point_cloud', data=seg_point_cloud_arrays, chunks=seg_point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('state', data=state_arrays, chunks=(100, state_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
        zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, chunks=(100,), dtype='int64', overwrite=True, compressor=compressor)

        # print shape
        cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
        cprint(f'fps_point_cloud shape: {fps_point_cloud_arrays.shape}, range: [{np.min(fps_point_cloud_arrays)}, {np.max(fps_point_cloud_arrays)}]', 'green')
        cprint(f'fps_point_cloud range: x:[{np.min(fps_point_cloud_arrays[:,:,0])}, {np.max(fps_point_cloud_arrays[:,:,0])}], \n \
            y:[{np.min(fps_point_cloud_arrays[:,:,1])}, {np.max(fps_point_cloud_arrays[:,:,1])}], \n \
                z:[{np.min(fps_point_cloud_arrays[:,:,2])}, {np.max(fps_point_cloud_arrays[:,:,2])}]', 'green')
        cprint(f'seg_point_cloud shape: {seg_point_cloud_arrays.shape}, range: [{np.min(seg_point_cloud_arrays)}, {np.max(seg_point_cloud_arrays)}]', 'green')
        cprint(f'seg_point_cloud range: x:[{np.min(seg_point_cloud_arrays[:,:,0])}, {np.max(seg_point_cloud_arrays[:,:,0])}], \n \
            y:[{np.min(seg_point_cloud_arrays[:,:,1])}, {np.max(seg_point_cloud_arrays[:,:,1])}], \n \
                z:[{np.min(seg_point_cloud_arrays[:,:,2])}, {np.max(seg_point_cloud_arrays[:,:,2])}]', 'green')
        cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
        cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
        if USE_ROTATION:
            cprint(f'action range: x:[{np.min(action_arrays[:,0])}, {np.max(action_arrays[:,0])}], \n \
                y:[{np.min(action_arrays[:,1]):.3f}, {np.max(action_arrays[:,1]):.3f}], \n \
                    z:[{np.min(action_arrays[:,2]):.3f}, {np.max(action_arrays[:,2]):.3f}], \n \
                    rx:[{np.min(action_arrays[:,3]):.3f}, {np.max(action_arrays[:,3]):.3f}], \n \
                    ry:[{np.min(action_arrays[:,4]):.3f}, {np.max(action_arrays[:,4]):.3f}], \n \
                    rz:[{np.min(action_arrays[:,5]):.3f}, {np.max(action_arrays[:,5]):.3f}], \n \
                    gripper:[{np.min(action_arrays[:,-1])}, {np.max(action_arrays[:,-1])}]', 'green')
        else:
            cprint(f'action range: x:[{np.min(action_arrays[:,0])}, {np.max(action_arrays[:,0])}], \n \
                y:[{np.min(action_arrays[:,1]):.3f}, {np.max(action_arrays[:,1]):.3f}], \n \
                    z:[{np.min(action_arrays[:,2]):.3f}, {np.max(action_arrays[:,2]):.3f}], \n \
                    gripper:[{np.min(action_arrays[:,-1])}, {np.max(action_arrays[:,-1])}]', 'green')
        cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
        cprint(f'episode_ends shape: {episode_ends_arrays.shape}, range: [{np.min(episode_ends_arrays)}, {np.max(episode_ends_arrays)}]', 'green')
        cprint(f'total_count: {total_count}', 'green')
        cprint(f'Saved zarr file to {save_data_path}', 'green')

        zarr_to_excel(save_data_path)


