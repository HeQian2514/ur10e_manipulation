from typing import Optional, Callable, Dict
import os
import enum
import time
import json
import numpy as np
import pyrealsense2 as rs
import multiprocessing as mp
import cv2
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
from common.timestamp_accumulator import get_accumulate_timestamp_idxs
from data_process.convert_point_data import sample_one_point_cloud
from shared_memory.shared_ndarray import SharedNDArray
from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from common.video_recorder import VideoRecorder

class Command(enum.Enum):
    SET_COLOR_OPTION = 0
    SET_DEPTH_OPTION = 1
    START_RECORDING = 2
    STOP_RECORDING = 3
    RESTART_PUT = 4

class SingleRealsense(mp.Process):
    MAX_PATH_LENGTH = 4096 # linux path has a limit of 4096 bytes

    def __init__(
            self, 
            shm_manager: SharedMemoryManager,
            serial_number,
            resolution=(720,1280),
            capture_fps=30,
            put_fps=None,
            put_downsample=True,
            record_fps=None,
            enable_color=True,
            enable_depth=True,
            enable_points=False,
            enable_infrared=False,
            get_max_k=30,
            advanced_mode_config=None,
            transform: Optional[Callable[[Dict], Dict]] = None,
            vis_transform: Optional[Callable[[Dict], Dict]] = None,
            recording_transform: Optional[Callable[[Dict], Dict]] = None,
            video_recorder: Optional[VideoRecorder] = None,
            verbose=False
        ):
        super().__init__()

        if put_fps is None:
            put_fps = capture_fps
        if record_fps is None:
            record_fps = capture_fps

        # create ring buffer
        resolution = tuple(resolution)
        shape = resolution[::-1]
        examples = dict()
        if enable_color:
            examples['color'] = np.empty(
                shape=shape+(3,), dtype=np.uint8)
        if enable_depth:
            examples['depth'] = np.empty(
                shape=shape, dtype=np.uint16)
        if enable_points:
            examples['point_cloud'] = np.empty(
                shape=(1024,6), dtype=np.float32)
        if enable_infrared:
            examples['infrared'] = np.empty(
                shape=shape, dtype=np.uint8)
        examples['camera_capture_timestamp'] = 0.0
        examples['camera_receive_timestamp'] = 0.0
        examples['timestamp'] = 0.0
        examples['step_idx'] = 0

        vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if vis_transform is None 
                else vis_transform(dict(examples)),
            get_max_k=1,
            get_time_budget=0.2,
            put_desired_frequency=capture_fps
        )

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None
                else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps
        )

        # create command queue
        examples = {
            'cmd': Command.SET_COLOR_OPTION.value,
            'option_enum': rs.option.exposure.value,
            'option_value': 0.0,
            'video_path': np.array('a'*self.MAX_PATH_LENGTH),
            'recording_start_time': 0.0,
            'put_start_time': 0.0
        }

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            buffer_size=128
        )

        # create shared array for intrinsics
        intrinsics_array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(7,),
                dtype=np.float64)
        intrinsics_array.get()[:] = 0

        # create video recorder
        if video_recorder is None:
            # realsense uses bgr24 pixel format
            # default thread_type to FRAEM
            # i.e. each frame uses one core
            # instead of all cores working on all frames.
            # this prevents CPU over-subpscription and
            # improves performance significantly
            video_recorder = VideoRecorder.create_h264(
                fps=record_fps, 
                codec='h264',
                input_pix_fmt='bgr24', 
                crf=18,
                thread_type='FRAME',
                thread_count=1)

        # copied variables
        self.serial_number = serial_number
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.record_fps = record_fps
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.enable_points = enable_points
        self.enable_infrared = enable_infrared
        self.advanced_mode_config = advanced_mode_config
        self.transform = transform
        self.vis_transform = vis_transform
        self.recording_transform = recording_transform
        self.video_recorder = video_recorder
        self.verbose = verbose
        self.put_start_time = None

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.vis_ring_buffer = vis_ring_buffer
        self.command_queue = command_queue
        self.intrinsics_array = intrinsics_array
    
    @staticmethod
    def get_connected_devices_serial():
        serials = list()
        for d in rs.context().devices:
            if d.get_info(rs.camera_info.name).lower() != 'platform camera':
                serial = d.get_info(rs.camera_info.serial_number)
                product_line = d.get_info(rs.camera_info.product_line)
                if product_line == 'D400':
                    # only works with D400 series
                    serials.append(serial)
        serials = sorted(serials)
        return serials

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        super().start()
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()
    
    def end_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)
    
    def get_vis(self, out=None):
        return self.vis_ring_buffer.get(out=out)
    
    # ========= user API ===========
    def set_color_option(self, option: rs.option, value: float):
        self.command_queue.put({
            'cmd': Command.SET_COLOR_OPTION.value,
            'option_enum': option.value,
            'option_value': value
        })
    
    def set_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """

        if exposure is None and gain is None:
            # auto exposure
            self.set_color_option(rs.option.enable_auto_exposure, 1.0)
        else:
            # manual exposure
            self.set_color_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_color_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_color_option(rs.option.gain, gain)
    
    def set_white_balance(self, white_balance=None):
        if white_balance is None:
            self.set_color_option(rs.option.enable_auto_white_balance, 1.0)
        else:
            self.set_color_option(rs.option.enable_auto_white_balance, 0.0)
            self.set_color_option(rs.option.white_balance, white_balance)

    def get_intrinsics(self):

        fx, fy, ppx, ppy = self.intrinsics_array.get()[:4]
        mat = np.eye(3)
        mat[0,0] = fx
        mat[1,1] = fy
        mat[0,2] = ppx
        mat[1,2] = ppy
        return mat

    def get_depth_scale(self):
        scale = self.intrinsics_array.get()[-1]
        return scale
    
    def start_recording(self, video_path: str, start_time: float=-1):
        assert self.enable_color

        path_len = len(video_path.encode('utf-8'))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError('video_path too long.')
        self.command_queue.put({
            'cmd': Command.START_RECORDING.value,
            'video_path': video_path,
            'recording_start_time': start_time
        })
        
    def stop_recording(self):
        self.command_queue.put({
            'cmd': Command.STOP_RECORDING.value
        })
    
    def restart_put(self, start_time):
        self.command_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': start_time
        })

    def _d2p(self, depth_img):
        """
        Convert depth image to point cloud.
        """
        
        # get intrinsics
        intrinsics = self.get_intrinsics()
        depth_scale = self.get_depth_scale()

        # create point cloud
        # st = time.time()
        xmap = np.arange(self.intrinsics_array.get()[5])
        ymap = np.arange(self.intrinsics_array.get()[4])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depth_img * depth_scale
        points_x = (xmap - intrinsics[0,2]) * points_z / intrinsics[0,0]
        points_y = (ymap - intrinsics[1,2]) * points_z / intrinsics[1,1]
        cloud = np.stack([points_x, points_y, points_z], axis=-1)
        # et = time.time()
        # print(f"Point cloud generation 1 took {et - st:.4f} seconds")
        
        return cloud

    def get_point_cloud(self, frameset, color_img, depth_img, use_cuda=False):
        """
        Generate point cloud with RGB data from color and depth images.
        Uses fastest native RealSense SDK implementation with farthest point sampling.
        
        Args:
            frameset: RealSense frameset containing aligned frames
            color_img: RGB color image as numpy array
            depth_img: Depth image as numpy array
            
        Returns:
            Sampled point cloud with shape (1024, 6) containing [x, y, z, r, g, b]
        """
        # Use RealSense SDK's built-in point cloud functionality (fastest method)
        pc = rs.pointcloud()
        depth_frame = frameset.get_depth_frame()
        
        # Calculate point cloud (without color mapping yet)
        points = pc.calculate(depth_frame)
        
        # Get vertices and texture coordinates
        vertices = np.asanyarray(points.get_vertices())
        
        # First, create array with just XYZ coordinates
        xyz_points = np.array([(v[0], v[1], v[2]) for v in vertices])
        
        # Filter invalid points (depth = 0 or NaN)
        valid_mask = ~np.isnan(xyz_points[:, 0]) & (xyz_points[:, 2] > 0)
        filtered_xyz = xyz_points[valid_mask]
        
        # Store indices of valid points to access texture coordinates later
        valid_indices = np.where(valid_mask)[0]
        
        # If we have enough points, perform farthest point sampling
        if len(filtered_xyz) > 0:
            try:
                import torch
                import pytorch3d.ops as torch3d_ops
                
                # Apply farthest point sampling on XYZ only
                xyz_tensor = torch.from_numpy(filtered_xyz)
                if use_cuda and torch.cuda.is_available():
                    xyz_tensor = xyz_tensor.cuda()
                
                # Sample points using FPS on position only
                K = [1024]
                _, indices = torch3d_ops.sample_farthest_points(
                    points=xyz_tensor.unsqueeze(0), K=K)
                indices = indices.squeeze(0).cpu().numpy()
                
                # Map back to indices in the original point cloud
                sampled_indices = valid_indices[indices]
                
                # Now get texture coordinates only for sampled points
                texcoords = np.asanyarray(points.get_texture_coordinates())
                
                # Map RGB colors to point cloud (needed for texture coordinates)
                pc.map_to(frameset.get_color_frame())
                
                # Get image dimensions
                h, w = color_img.shape[:2]
                
                # Create final point cloud with colors
                final_points = np.zeros((1024, 6), dtype=np.float32)
                
                # Fill in XYZ coordinates
                final_points[:, 0:3] = filtered_xyz[indices]
                
                # Fill in RGB colors only for sampled points
                for i, orig_idx in enumerate(sampled_indices):
                    tc = texcoords[orig_idx]
                    u, v = int(tc[0] * w), int(tc[1] * h)
                    if 0 <= u < w and 0 <= v < h:
                        final_points[i, 3:6] = color_img[v, u] / 255.0
                
                return final_points
                
            except ImportError:
                # Fallback if PyTorch3D not available - optimize this case too
                # Choose random indices first, then add colors
                if len(filtered_xyz) >= 1024:
                    idx = np.random.choice(len(filtered_xyz), 1024, replace=False)
                else:
                    idx = np.random.choice(len(filtered_xyz), 1024, replace=True)
                
                # Map RGB colors to point cloud (needed for texture coordinates)
                pc.map_to(frameset.get_color_frame())
                texcoords = np.asanyarray(points.get_texture_coordinates())
                
                # Get image dimensions
                h, w = color_img.shape[:2]
                
                # Create final point cloud with colors
                final_points = np.zeros((1024, 6), dtype=np.float32)
                
                # Fill in XYZ coordinates
                final_points[:, 0:3] = filtered_xyz[idx]
                
                # Fill in RGB colors only for sampled points
                sampled_indices = valid_indices[idx]
                for i, orig_idx in enumerate(sampled_indices):
                    tc = texcoords[orig_idx]
                    u, v = int(tc[0] * w), int(tc[1] * h)
                    if 0 <= u < w and 0 <= v < h:
                        final_points[i, 3:6] = color_img[v, u] / 255.0
                
                return final_points
        else:
            # If no valid points, return zero array
            return np.zeros((1024, 6), dtype=np.float32)

    def get_point_cloud0(self, frameset, color_img, depth_img):
        """
        Generate point cloud with RGB data from color and depth images.
        Uses fastest native RealSense SDK implementation with farthest point sampling.
        
        Args:
            frameset: RealSense frameset containing aligned frames
            color_img: RGB color image as numpy array
            depth_img: Depth image as numpy array
            
        Returns:
            Sampled point cloud with shape (1024, 6) containing [x, y, z, r, g, b]
        """
        # Use RealSense SDK's built-in point cloud functionality (fastest method)
        pc = rs.pointcloud()
        depth_frame = frameset.get_depth_frame()
        
        # Map RGB colors to point cloud
        pc.map_to(frameset.get_color_frame())
        
        # Calculate point cloud
        points = pc.calculate(depth_frame)
        
        # Get vertices and texture coordinates
        vertices = np.asanyarray(points.get_vertices())
        texcoords = np.asanyarray(points.get_texture_coordinates())
        
        # Get image dimensions
        h, w = color_img.shape[:2]
        
        # Initialize point cloud array with x,y,z,r,g,b
        cloud = np.zeros((len(vertices), 6), dtype=np.float32)
        
        # Fill x,y,z coordinates
        cloud[:, 0:3] = np.array([(v[0], v[1], v[2]) for v in vertices])
        
        # Fill r,g,b colors
        for i, tc in enumerate(texcoords):
            # Ensure texture coordinates are within valid range
            u, v = int(tc[0] * w), int(tc[1] * h)
            if 0 <= u < w and 0 <= v < h:
                cloud[i, 3:6] = color_img[v, u] / 255.0  # Normalize RGB to [0,1]
        
        # Filter invalid points (depth = 0 or NaN)
        valid_mask = ~np.isnan(cloud[:, 0]) & (cloud[:, 2] > 0)
        valid_points = cloud[valid_mask]
        
        # If we have enough points, perform farthest point sampling
        if len(valid_points) > 0:
            try:
                import torch
                import pytorch3d.ops as torch3d_ops
                
                # Apply farthest point sampling to get 1024 uniformly distributed points
                sampled_points, _ = self.farthest_point_sampling(valid_points, num_points=1024)
                
                # Ensure we have exactly 1024 points
                if len(sampled_points) < 1024:
                    # If not enough points, fill by repeating existing points
                    repeat_count = int(np.ceil(1024 / len(sampled_points)))
                    repeated_points = np.tile(sampled_points, (repeat_count, 1))
                    sampled_points = repeated_points[:1024]
                
                return sampled_points
                
            except ImportError:
                # If pytorch3d is not available, use random sampling
                if len(valid_points) >= 1024:
                    idx = np.random.choice(len(valid_points), 1024, replace=False)
                else:
                    idx = np.random.choice(len(valid_points), 1024, replace=True)
                return valid_points[idx]
        else:
            # If no valid points, return zero array
            return np.zeros((1024, 6), dtype=np.float32)
            
    def farthest_point_sampling(self, points, num_points=1024, use_cuda=False):
        """
        Perform farthest point sampling algorithm to select uniformly distributed points
        
        Args:
            points: Point cloud array with shape (N, 6), containing x,y,z,r,g,b
            num_points: Number of points to sample
            use_cuda: Whether to use GPU acceleration
            
        Returns:
            Sampled point cloud with shape (num_points, 6)
        """
        try:
            import torch
            import pytorch3d.ops as torch3d_ops
            
            K = [num_points]
            if use_cuda and torch.cuda.is_available():
                points = torch.from_numpy(points).cuda()
                sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
                sampled_points = sampled_points.squeeze(0)
                sampled_points = sampled_points.cpu().numpy()
            else:
                points = torch.from_numpy(points)
                sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
                sampled_points = sampled_points.squeeze(0)
                sampled_points = sampled_points.numpy()

            return sampled_points, indices
        except ImportError:
            print("Warning: PyTorch3D not installed, using random sampling instead of farthest point sampling")
            # Random sampling as fallback
            if len(points) >= num_points:
                idx = np.random.choice(len(points), num_points, replace=False)
            else:
                idx = np.random.choice(len(points), num_points, replace=True)
            return points[idx], idx
     
    # ========= interval API ===========
    def run(self):
        # limit threads
        # threadpool_limits(1)
        # cv2.setNumThreads(1)

        w, h = self.resolution
        fps = self.capture_fps
        align = rs.align(rs.stream.color)
        # Enable the streams from all the intel realsense devices
        rs_config = rs.config()
        if self.enable_color:
            rs_config.enable_stream(rs.stream.color, 
                w, h, rs.format.bgr8, fps)
        if self.enable_depth:
            rs_config.enable_stream(rs.stream.depth, 
                w, h, rs.format.z16, fps)
        if self.enable_infrared:
            rs_config.enable_stream(rs.stream.infrared,
                w, h, rs.format.y8, fps)
        
        try:
            rs_config.enable_device(self.serial_number)

            # start pipeline
            pipeline = rs.pipeline()
            pipeline_profile = pipeline.start(rs_config)

            # report global time
            # https://github.com/IntelRealSense/librealsense/pull/3909
            d = pipeline_profile.get_device().first_color_sensor()
            d.set_option(rs.option.global_time_enabled, 1)

            # setup advanced mode
            if self.advanced_mode_config is not None:
                json_text = json.dumps(self.advanced_mode_config)
                device = pipeline_profile.get_device()
                advanced_mode = rs.rs400_advanced_mode(device)
                advanced_mode.load_json(json_text)

            # get
            color_stream = pipeline_profile.get_stream(rs.stream.color)
            intr = color_stream.as_video_stream_profile().get_intrinsics()
            order = ['fx', 'fy', 'ppx', 'ppy', 'height', 'width']
            for i, name in enumerate(order):
                self.intrinsics_array.get()[i] = getattr(intr, name)

            if self.enable_depth:
                depth_sensor = pipeline_profile.get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()
                self.intrinsics_array.get()[-1] = depth_scale
            
            # one-time setup (intrinsics etc, ignore for now)
            if self.verbose:
                print(f'[SingleRealsense {self.serial_number}] Main loop started.')

            # put frequency regulation
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            iter_idx = 0
            t_start = time.time()
            while not self.stop_event.is_set():
                # wait for frames to come in
                frameset = pipeline.wait_for_frames()
                receive_time = time.time()
                # align frames to color
                frameset = align.process(frameset)

                # grab data
                data = dict()
                data['camera_receive_timestamp'] = receive_time
                # realsense report in ms
                data['camera_capture_timestamp'] = frameset.get_timestamp() / 1000
                if self.enable_color:
                    color_frame = frameset.get_color_frame()
                    data['color'] = np.asarray(color_frame.get_data())
                    t = color_frame.get_timestamp() / 1000
                    data['camera_capture_timestamp'] = t
                    # print('device', time.time() - t)
                    # print(color_frame.get_frame_timestamp_domain())
                if self.enable_depth:
                    data['depth'] = np.asarray(
                        frameset.get_depth_frame().get_data())
                if self.enable_points:
                    assert self.enable_depth and self.enable_color
                    point_cloud = self._d2p(data['depth'])
                    data['point_cloud'] = sample_one_point_cloud(point_cloud, data['color'])

                if self.enable_infrared:
                    data['infrared'] = np.asarray(
                        frameset.get_infrared_frame().get_data())
                
                # apply transform
                put_data = data
                if self.transform is not None:
                    put_data = self.transform(dict(data))

                if self.put_downsample:                
                    # put frequency regulation
                    local_idxs, global_idxs, put_idx \
                        = get_accumulate_timestamp_idxs(
                            timestamps=[receive_time],
                            start_time=put_start_time,
                            dt=1/self.put_fps,
                            # this is non in first iteration
                            # and then replaced with a concrete number
                            next_global_idx=put_idx,
                            # continue to pump frames even if not started.
                            # start_time is simply used to align timestamps.
                            allow_negative=True
                        )

                    for step_idx in global_idxs:
                        put_data['step_idx'] = step_idx
                        # put_data['timestamp'] = put_start_time + step_idx / self.put_fps
                        put_data['timestamp'] = receive_time
                        # print(step_idx, data['timestamp'])
                        self.ring_buffer.put(put_data, wait=False)
                else:
                    step_idx = int((receive_time - put_start_time) * self.put_fps)
                    put_data['step_idx'] = step_idx
                    put_data['timestamp'] = receive_time
                    self.ring_buffer.put(put_data, wait=False)

                # signal ready
                if iter_idx == 0:
                    self.ready_event.set()
                
                # put to vis
                vis_data = data
                if self.vis_transform == self.transform:
                    vis_data = put_data
                elif self.vis_transform is not None:
                    vis_data = self.vis_transform(dict(data))
                self.vis_ring_buffer.put(vis_data, wait=False)
                
                # record frame
                rec_data = data
                if self.recording_transform == self.transform:
                    rec_data = put_data
                elif self.recording_transform is not None:
                    rec_data = self.recording_transform(dict(data))

                if self.video_recorder.is_ready():
                    self.video_recorder.write_frame(rec_data['color'], 
                        frame_time=receive_time)

                # perf
                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / duration, 1)
                t_start = t_end
                if self.verbose:
                    print(f'[SingleRealsense {self.serial_number}] FPS {frequency}')

                # fetch command from queue
                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    if cmd == Command.SET_COLOR_OPTION.value:
                        sensor = pipeline_profile.get_device().first_color_sensor()
                        option = rs.option(command['option_enum'])
                        value = float(command['option_value'])
                        sensor.set_option(option, value)
                        # print('auto', sensor.get_option(rs.option.enable_auto_exposure))
                        # print('exposure', sensor.get_option(rs.option.exposure))
                        # print('gain', sensor.get_option(rs.option.gain))
                    elif cmd == Command.SET_DEPTH_OPTION.value:
                        sensor = pipeline_profile.get_device().first_depth_sensor()
                        option = rs.option(command['option_enum'])
                        value = float(command['option_value'])
                        sensor.set_option(option, value)
                    elif cmd == Command.START_RECORDING.value:
                        video_path = str(command['video_path'])
                        start_time = command['recording_start_time']
                        if start_time < 0:
                            start_time = None
                        self.video_recorder.start(video_path, start_time=start_time)
                    elif cmd == Command.STOP_RECORDING.value:
                        self.video_recorder.stop()
                        # stop need to flush all in-flight frames to disk, which might take longer than dt.
                        # soft-reset put to drop frames to prevent ring buffer overflow.
                        put_idx = None
                    elif cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = command['put_start_time']
                        # self.ring_buffer.clear()

                iter_idx += 1
        finally:
            self.video_recorder.stop()
            rs_config.disable_all_streams()
            self.ready_event.set()
        
        if self.verbose:
            print(f'[SingleRealsense {self.serial_number}] Exiting worker process.')
