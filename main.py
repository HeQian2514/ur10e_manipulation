import sys
import os
import pickle
import time
import click
from multiprocessing.managers import SharedMemoryManager
from data_process.convert_point_data import d2p, sample_one_point_cloud
from data_process.convert_real_robot_data import smooth_rotation_vector2state, state2smooth_rotation_vector

from common.real_env import RealEnv
import torch
import numpy as np
import cv2
from omegaconf import DictConfig, OmegaConf
import hydra
import gc
from pynput import keyboard
import dill
import pathlib
import random
from common.cv2_util import get_image_transform
from common.precise_sleep import precise_wait
import scipy.spatial.transform as st
from common.pytorch_util import dict_apply
import psutil
# import clip

from common.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
# from train import TrainFlowPolicyWorkspace

if False:
    from spacemouse_shared_memory import Spacemouse
else:
    from common.gamepad_shared_memory import F710SpaceMouse as Spacemouse

q_pressed = False
c_pressed = False

def on_press(key):
    global q_pressed, c_pressed
    try:
        if key.char == 'q':
            print('Exiting program')
            q_pressed = True
        elif key.char == 'r':
            print('Switching mode')
            c_pressed = True
    except AttributeError:
        pass

def on_release(key):
    global q_pressed, c_pressed
    try:
        if key.char == 'q':
            q_pressed = False
        elif key.char == 'r':
            c_pressed = False
    except AttributeError:
        pass

# import multiprocessing
# multiprocessing.set_start_method('spawn', force=True)

OmegaConf.register_new_resolver("eval", eval)

def save_obd(save_data, data_dir):
    data_file = data_dir
    if not os.path.exists(data_file):
        os.makedirs(data_file)

    existing_files = [f for f in os.listdir(data_file) if f.startswith("demo_") and f.endswith(".pkl")]
    file_count = len(existing_files)

    # Generate new file name
    file_name = f"demo_{file_count}.pkl"
    data_file = os.path.join(data_file, file_name)
    
    # Save data
    with open(data_file, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Data saved to: {data_file}")

VELOCITY = 0.0015  # m per control loop

@hydra.main(config_path="config", config_name="real_robot")
def main(cfg: DictConfig):
    # train_cfg = hydra.compose(config_name="dp3_1_512")
    # train_runner = TrainFlowPolicyWorkspace(train_cfg)

    global q_pressed, c_pressed

    ## prepare for control
    OmegaConf.resolve(cfg)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    print("Keyboard listener started")

    ## parameters from cfg

    robot_ip = cfg.robot_ip
    frequency = cfg.frequency
    output = cfg.output
    vis_camera_idx = cfg.vis_camera_idx
    command_latency = cfg.command_latency

    load_ckpt_path = cfg.get('load_checkpoint_from', False)
    if cfg.policy_name == 'fp':
        policy = hydra.utils.instantiate(cfg.fp_policy)
    elif cfg.policy_name == 'dp3':
        policy = hydra.utils.instantiate(cfg.dp3_policy)
    elif cfg.policy_name == 'simple_dp3':
        policy = hydra.utils.instantiate(cfg.simple_dp3_policy)
    elif cfg.policy_name == 'focalpolicy':
        policy = hydra.utils.instantiate(cfg.focal_policy)
    # you can add more policies here
    else:
        raise 'policy error!'
    
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if load_ckpt_path:
        path = pathlib.Path(load_ckpt_path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')

    seed = cfg.model_training_seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ## set up control loop
    dt = 1 / frequency

    obs_resolution = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps

    button_a_state = False
    previous_button_a_state = False
    button_x_state = False
    previous_button_x_state = False

    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager) as sm, RealEnv(
            output_dir=output, 
            robot_ip=robot_ip, 
            frequency=frequency,
            n_obs_steps=n_obs_steps,
            obs_image_resolution=obs_resolution,
            obs_float32=True,
            enable_multi_cam_vis=True,
            record_raw_video=True,
            # number of threads per camera view for video recording (H.264)
            thread_per_video=3,
            # video recording quality, lower is better (but slower).
            video_crf=21,
            shm_manager=shm_manager) as env:
            cv2.setNumThreads(1)    # disable OpenCV multithreading

            # realsense exposure
            env.realsense.set_exposure(exposure=120, gain=0)
            # realsense white balance
            env.realsense.set_white_balance(white_balance=5900)

            print("Waiting for realsense")
            time.sleep(2.0)
            save_data = None
            print('Starting control loop!!!')
            while True:
                # ++++++++++ human control loop ++++++++++
                print("Human in control!")
                state = env.get_robot_state()
                target_pose = state['ActualTCPPose']
                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt 

                    # pump obs
                    obs = env.get_obs()
                    
                    # save obs
                    if button_a_state:  # save
                        if not previous_button_a_state: # begin save
                            save_data = {'color': [],'depth': [],'pose': [], 'gripper':[], 'timestamp': [],}
                            previous_button_a_state = True
                            print("Saving data...")
                        save_data['color'].append(obs[f'camera_{vis_camera_idx}_color'][-1])
                        save_data['depth'].append(obs[f'camera_{vis_camera_idx}_depth'][-1])
                        save_data['pose'].append(obs[f'robot_eef_pose'][-1])
                        save_data['gripper'].append(obs[f'robot_gripper_position'][-1])
                        save_data['timestamp'].append(time.time())
                        if iter_idx % 30 == 0:
                            print(f"Obs latency {time.time() - obs[f'timestamp'][-1]}")
                        
                    elif not button_a_state and previous_button_a_state: # end save
                        save_obd(save_data, cfg.data_dir)
                        previous_button_a_state = False
                        save_data = None

                        for _ in range(3):
                            gc.collect()
                        mem_after = psutil.virtual_memory().percent
                        available_gb = psutil.virtual_memory().available / (1024**3)
                        
                        print(f"{mem_after:.1f}%, Available: {available_gb:.2f}GB")

                    if q_pressed:
                        env.end_episode()
                        exit(0)
                    elif c_pressed:
                        c_pressed = False  # Reset state to avoid multiple triggers
                        break

                    precise_wait(t_sample)
                    # get teleop command
                    motion_state, button_ay, gripper_cmd = sm.get_motion_state_transformed()
                    button_a_state = button_ay[0]
                    button_y_state = button_ay[1]
                    button_x_state = button_ay[2]
                    button_l_state = button_ay[3]
                    button_r_state = button_ay[4]

                    # print(sm_state)
                    # dpos = motion_state[:3] * (env.max_pos_speed / frequency)
                    dpos = np.where(
                        np.abs(motion_state[:3]) > 0.1,
                        np.sign(motion_state[:3]) * VELOCITY,
                        0
                    )
                    # print(f'dpos: {dpos}')
                    drot_xyz = motion_state[3:] * (env.max_rot_speed / frequency)
  
                    drot = st.Rotation.from_euler('xyz', drot_xyz)
                    target_pose[:3] += dpos
                    target_pose[3:] = (drot * st.Rotation.from_rotvec(target_pose[3:])).as_rotvec()

                    if button_y_state:
                        print("Resetting pose")
                        target_pose = np.array(cfg.init_target_pose.copy())
                        gripper_cmd = 100  # open
                        t_command_target += 200 * dt 
                    elif button_x_state != previous_button_x_state:
                        print('spacific pose z!')
                        target_pose[2] -= 0.005  # z down
                        t_command_target += 100 * dt
                        previous_button_x_state = button_x_state
                    elif button_l_state:
                        print('spacific pose rxyz 1!')
                        target_pose[3:] = np.array([1.383, -1.529, 1.109])  # rxyz to 0
                        t_command_target += 120 * dt
                    elif button_r_state:
                        print('spacific pose rxyz 2!')
                        # target_pose[3:] = np.array([0.0, 0.0, 1.57])  # rxyz to [0,0,1.57]
                        # t_command_target += 100 * dt

                    # execute teleop command
                    env.exec_actions(
                        actions=[target_pose], 
                        gripper_cmds=[gripper_cmd],
                        timestamps=[t_command_target-time.monotonic()+time.time()]) 
                    
                    if iter_idx % 30 == 0:
                        formatted_pose = [f"{x:.3f}" for x in target_pose]
                        form_motion_state = [f"{x:.1f}" for x in motion_state]
                        print(f'target_pose: [{", ".join(formatted_pose)}], motion_state: {", ".join(form_motion_state)}, gripper_cmd: {gripper_cmd}')
                        # print(f'obs_pos: ',obs[f'robot_eef_pose'][-1])
                        # print(f'robot_state: ', env.get_robot_state()['ActualTCPPose'])
                        result = state2smooth_rotation_vector(target_pose)
                        if save_data == None:
                            print(f'Smoothed pose: {result}, saving none data ')
                        else:
                             print(f'Smoothed pose: {result}')
                        # result2 = smooth_rotation_vector2state(result)
                        # print(f'Smoothed pose2: {result2}')

                    precise_wait(t_cycle_end)
                    iter_idx += 1

                # ========== policy control loop ==============
                try:
                    if load_ckpt_path:
                        value = payload['state_dicts']['model']   
                        policy.load_state_dict(value)
                        # policy.normalizer.load_state_dict(payload['state_dicts']['normalizer'])
                        print(f"Loaded policy checkpoint from {load_ckpt_path}")
                        missing, unexpected = policy.load_state_dict(value, strict=False)
                        print("Missing keys:", missing)
                        print("Unexpected keys:", unexpected)
                    # train_runner.load_checkpoint(path='/home/hq/PROJECT/FlowPolicy/real_world/data/outputs/slide_block_big_1-dp3_1_512-SCL1000_SKP10_ST1_DX_PC1_1024_PT4_1_seed0/checkpoints/epoch=5900-test_mean_score=-0.000.ckpt')
                    # policy = train_runner.model
                    # policy.set_normalizer(train_runner.model.normalizer)
                    # if train_cfg.training.use_ema:
                    #     policy = train_runner.ema_model
                    # for param in policy.parameters():
                    #     param.data.copy_(torch.randn_like(param.data))
                    
                    policy.eval()
                    policy.cuda()
                    print("Warming up policy inference")
                    
                    eval_t_start = time.time() 
                    t_start = time.monotonic() 
                    env.start_episode(eval_t_start)
                    dt = 0.5 # s
                    frame_latency = 1/30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Policy Contral Started!")   
                    iter_idx = 0

                    steps_per_inference = cfg.n_action_steps # 6 steps per inference
                    action_offset = 1     # 0.0 seconds offset for action timestamps
                    action_buffer = []  # Buffer list for storing segmented actions
                    seg_reset = True

                    while True:
                        # get obs
                        motion_state, button_ay, gripper_cmd = sm.get_motion_state_transformed()
                        button_y_state = button_ay[1]

                        if button_y_state:
                            print("Resetting pose")
                            this_target_poses = [np.array(cfg.init_target_pose.copy())]
                            this_target_gripper_cmds = [100]  # 0 close 100 open
                            action_timestamps = [200 * 1 / frequency + time.time()]
                            action_buffer = []
                            delta_time = 0.1
                            seg_reset = True
                            
                        else:
                            if len(action_buffer) > 0:
                                print(f"=== Executing buffered action segment {len(action_buffer)} remaining ===")
                                
                                # Take the first action segment from the buffer
                                buffered_action = action_buffer.pop(0)
                                this_target_poses = buffered_action['poses']
                                this_target_gripper_cmds = buffered_action['gripper_cmds']
                                
                            else:                         
                                obs = env.get_obs()
                                obs_timestamps = obs['timestamp']
                                obs_delay = time.time() - obs_timestamps[-1]
                                print(f'Obs latency {obs_delay}')

                                # run inference                      
                                with torch.no_grad():
                                    s = time.time()
                                    curr_pos = np.stack(obs['robot_eef_pose'])
                                    print('obs pos:', curr_pos)
                                    if not cfg.use_rotation:
                                        init_pos = curr_pos[-1:].copy()
                                        rotation_pos = curr_pos[:, 3:]
                                        pos = curr_pos[:, :3]      # 
                                        state = pos.copy()*1000  # m to mm
                                    else:
                                        pos = curr_pos.copy()
                                        state_pos = curr_pos[:, :3].copy()*1000  # m to mm
                                        state_rot = state2smooth_rotation_vector(pos)
                                        state = np.concatenate([state_pos, state_rot[:, 3:]], axis=-1)
                                        
                                    gripper = np.stack(obs[f'robot_gripper_position']/100) 
                                    gripper = gripper.reshape(-1, 1)
                                    point_cloud = d2p(obs[f'camera_{vis_camera_idx}_depth'])
                                    sample_point_clouds = []
                                    if cfg.sample_method == 'fps':
                                        for i in range(len(point_cloud)):
                                            sample_point_cloud = sample_one_point_cloud(point_cloud[i], obs[f'camera_{vis_camera_idx}_color'][i], num_points=2048, use_cuda=True, crop_scale=cfg.crop_size)
                                            sample_point_clouds.append(sample_point_cloud)
                                    elif cfg.sample_method == 'segment':
                                        # from data.fastsam2_point_final_sasa import dw_fps_sampling
                                        # for i in range(len(point_cloud)):
                                        #     sample_point_cloud = dw_fps_sampling(
                                        #         point_cloud[i], obs[f'camera_{vis_camera_idx}_color'][i], total_sample_points=1024, crop_scale=cfg.crop_size, reset =seg_reset,
                                        #         iou_conf=[[0.5, 0.1], [0.5, 0.1]],
                                        #         save_mask_visualizations='/home/hq/PROJECT/FlowPolicy/real_world/check_pkg/DW_FPS_IMAGE/real_eval')
                                        #     seg_reset = False
                                        #     sample_point_clouds.append(sample_point_cloud)
                                        pass
                                            
                                    zero_point_clouds = [np.zeros_like(pc) for pc in sample_point_clouds]
                                    # sample_point_clouds = zero_point_clouds
                                    
                                    obs_dict_np = {'point_cloud': np.stack(sample_point_clouds), # n, 1024, 6
                                                    'agent_pos': np.concatenate([state, gripper], axis=-1)}  # n, 6+1


                                    if False:
                                        with open("/home/hq/PROJECT/FlowPolicy/real_world/data/debug_data/test_policy_para.txt", "w") as f:
                                            for name, param in policy.named_parameters():
                                                f.write(f"{name}: {param.data.cpu().numpy()}\n")

                                        segmented_data = {
                                            'fps_point_cloud': obs_dict_np['point_cloud']
                                        }
                                        with open(cfg.pc_save_path, 'wb') as f:
                                            pickle.dump(segmented_data, f)
                                        print(f"Segmented weighted sampled point cloud saved to: {cfg.pc_save_path}")

                                        img = obs[f'camera_{vis_camera_idx}_color'][-i-1]
                                        # If it is a 0-1 float, convert to 0-255 uint8
                                        if img.dtype != np.uint8:
                                            img = (img * 255).clip(0, 255).astype(np.uint8)
                                        img_path = os.path.join(cfg.image_save_path, f"real_pc_images.jpg")
                                        cv2.imwrite(img_path, img)
                                        print(f"Sampled point cloud corresponding image saved to: {img_path}")

                                    obs_dict = dict_apply(obs_dict_np, 
                                        lambda x: torch.from_numpy(x).unsqueeze(0).to(device))  # change to tensor

                                    result = policy.predict_action(obs_dict)

                                    
                                    # this action starts from the first obs step
                                    action = result['action'][0].detach().to('cpu').numpy()
                                    inference_latency = time.time() - s
                                    print('Inference latency:', inference_latency)

                                # calculate timing
                                t_cycle_end = t_start + iter_idx * dt 

                                # convert action to env action
                                if not cfg.use_rotation:
                                    action = np.concatenate([action[:, :3], rotation_pos[-1:].repeat(len(action), axis=0), action[:, 3:4]], axis=-1)
                                assert action.shape[-1] == 7
                                this_target_poses = action[:cfg.n_action_steps, :6]  # [h, 6]
                                this_target_gripper_cmds = action[:cfg.n_action_steps, 6]*100  # [h, 1]

                                # deal with action timestamps

                                this_target_poses = np.array(this_target_poses)
                                this_target_poses[:, :3] = this_target_poses[:, :3] / cfg.unit_scale
                                this_target_gripper_cmds = np.clip(this_target_gripper_cmds, 0, 100)
                                this_target_gripper_cmds[this_target_gripper_cmds > 10] = 100
                                this_target_gripper_cmds[this_target_gripper_cmds <= 10] = 0
                                # action_timestamps = action_timestamps[is_new]
                                
                                if cfg.delta_action:
                                    # this_target_poses = np.round(this_target_poses, 4)
                                    factor = 10 ** 4
                                    this_target_poses = np.trunc(this_target_poses * factor) / factor
                                    if cfg.use_rotation:
                                        if cfg.delta_rotation:
                                            this_target_poses = np.cumsum(this_target_poses, axis=0)
                                            this_target_poses = this_target_poses + state2smooth_rotation_vector(pos[-1])
                                        else:
                                            this_target_poses[:, :3] = np.cumsum(this_target_poses[:, :3], axis=0)
                                            this_target_poses[:, :3] = this_target_poses[:, :3] + pos[-1][:3]
                                    else:
                                        this_target_poses[:, :3] = np.cumsum(this_target_poses[:, :3], axis=0)
                                        this_target_poses[:, :3] = this_target_poses[:, :3] + pos[-1][:3]
                                if not cfg.use_rotation:
                                    this_target_poses[:, 3:] = this_target_poses[:, 3:] * 0.0
                                    this_target_poses[:, 3:] = this_target_poses[:, 3:] + state2smooth_rotation_vector(init_pos[-1])[3:]

                                # clip target poses
                                formatted_poses = [f"[{', '.join([f'{x:.3f}' for x in pose])}]" for pose in this_target_poses]
                                for i in range (len(this_target_poses)):
                                    print(f'target_poses: {formatted_poses[i]}, gripper_cmd: {this_target_gripper_cmds[i]:.1f}')
                                
                                # this_target_poses[:, :5] = np.clip(
                                #     this_target_poses[:, :5],   # x, y, z, rx, ry
                                #     [-0.4, -0.85, 0.225, -0.7, 0.1], # Reachable workspace under initial pose
                                #     [0.2, -0.4, 0.642, 0.6, 0.85])
                                # this_target_poses[:, :3] = np.clip(
                                #     this_target_poses[:, :3],   # x, y, z, rx, ry
                                #     [-0.4, -0.85, 0.225], # Reachable workspace under initial pose
                                #     [0.2, -0.4, 0.642])
                                this_target_poses[:, :3] = np.clip(
                                    this_target_poses[:, :3],   # x, y, z, rx, ry
                                    [-0.4, -0.87, 0.225], # Reachable workspace under initial pose
                                    [0.2, -0.4, 0.642])
                                # print(f'Clipped target poses: {this_target_poses}')
                                this_target_poses = smooth_rotation_vector2state(this_target_poses)
                                # print(f'Convert target poses: {this_target_poses}')

                                gripper_changes = np.where(np.diff(this_target_gripper_cmds) != 0)[0] + 1  # Indices of changing points

                                if len(gripper_changes) > 0:  # Changes exist
                                    print(f"Gripper changes detected at steps: {gripper_changes}")
                                    
                                    # Split points: [0, change1, change2, ..., len]
                                    split_points = np.concatenate([[0], gripper_changes, [len(this_target_gripper_cmds)]])
                                    
                                    # Direct use of first segment
                                    first_seg_end = split_points[1]
                                    seg_poses = this_target_poses[:first_seg_end].copy()
                                    seg_gripper_cmds = this_target_gripper_cmds[:first_seg_end].copy()

                                    if len(split_points) > 2:
                                        next_seg_start = split_points[1]
                                        next_first_pose = this_target_poses[next_seg_start].copy()  # Change to single element
                                        next_first_gripper = seg_gripper_cmds[-1]  # Change to scalar
                                        
                                        seg_poses = np.vstack([seg_poses, [next_first_pose]])  # Use vstack to append a row
                                        seg_gripper_cmds = np.append(seg_gripper_cmds, next_first_gripper)  # Use append to append an element

                                    print(f"=== Segment 1: {len(seg_poses)} steps, gripper={seg_gripper_cmds[0]:.0f} ===")
    
                                    for i in range(1, len(split_points) - 1):
                                        start_idx = split_points[i]
                                        end_idx = split_points[i + 1]
                                        
                                        seg_poses_buf = this_target_poses[start_idx:end_idx].copy()
                                        seg_gripper_cmds_buf = this_target_gripper_cmds[start_idx:end_idx].copy()
                                        
                                        # If not the last segment, add the first point of next segment (but keep current segment's gripper state)
                                        if i < len(split_points) - 2:
                                            next_seg_start = end_idx
                                            next_first_pose = this_target_poses[next_seg_start].copy()
                                            next_first_gripper = seg_gripper_cmds_buf[-1]  # Keep current segment's gripper state
                                            
                                            seg_poses_buf = np.vstack([seg_poses_buf, [next_first_pose]])
                                            seg_gripper_cmds_buf = np.append(seg_gripper_cmds_buf, next_first_gripper)

                                        action_buffer.append({
                                            'poses': seg_poses_buf,
                                            'gripper_cmds': seg_gripper_cmds_buf
                                        })
                                        print(f"=== Segment {i+1}: {len(seg_poses_buf)} steps, gripper={seg_gripper_cmds_buf[0]:.0f} (buffered) ===")
    
                                    # Update current action to execute as first segment
                                    this_target_poses = seg_poses
                                    this_target_gripper_cmds = seg_gripper_cmds

                            curr_time = time.time()
                            action_timestamps = (np.arange(len(this_target_poses), dtype=np.float64) + 1
                                                )* dt + curr_time
                            
                            delta_time = action_timestamps[-1] - curr_time + action_offset
                            t_cycle_end = t_cycle_end + delta_time

                            # this_target_poses = this_target_poses[-1:]
                            # this_target_gripper_cmds = this_target_gripper_cmds[-1:]
                            # action_timestamps = action_timestamps[-1:]

                            formatted_poses = [f"[{', '.join([f'{x:.3f}' for x in pose])}]" for pose in this_target_poses]
                            print(f'current pose: {pos[-1]}')
                            for i in range (len(formatted_poses)):
                                print(f'clip_target_poses: {formatted_poses[i]}, gripper_cmd: {this_target_gripper_cmds[i]:.1f}, timestamp: {action_timestamps[i]:.3f}')
                        
                        # execute actions
                        env.exec_actions(
                            actions=this_target_poses,
                            gripper_cmds=this_target_gripper_cmds,
                            timestamps=action_timestamps
                        )
                        # print(f"Submitted {len(this_target_poses)} steps of actions.")

                        # if q_pressed:
                            # env.end_episode()
                            # exit(0)

                        # auto termination

                        # wait for execution
                        # precise_wait(t_cycle_end)
                        time.sleep(delta_time)  # avoid busy wait
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                    # Clear model on GPU
                    del policy
                    torch.cuda.empty_cache()
                
                print("Stopped.")

if __name__ == '__main__':
    main()

    '''fuser -v /dev/nvidia*'''
    '''pgrep python | xargs kill -9'''

