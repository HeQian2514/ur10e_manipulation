import os
import logging
import enum
import time
import numpy as np

import multiprocessing as mp
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
from common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from common.precise_sleep import precise_wait

from shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

import sys
import select
import termios
import tty

from multiprocessing.managers import SharedMemoryManager

from common.gripper_controller import AG95TCPClient

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2

class RTDEInterpolationController(mp.Process):

    def __init__(self,
            shm_manager: SharedMemoryManager, 
            robot_ip, 
            robot_port=30004,
            frequency=125, 
            lookahead_time=0.1, 
            gain=300,
            max_pos_speed=0.25, # 5% of max speed
            max_rot_speed=0.16, # 5% of max speed
            launch_timeout=3,
            tcp_offset_pose=None,
            payload_mass=None,
            payload_cog=None,
            joints_init=None,
            joints_init_speed=1.05,
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=128,
            config_filename = "/home/hq/PROJECT/FlowPolicy/real_world/RTDE_Python_Client_Library/examples/control_loop_configuration.xml"
            ):
        super().__init__(name="RTDEPositionalController")

        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.config_filename = config_filename
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose

        self.gripper_client = AG95TCPClient(ip='192.168.1.29', port=8888)
        self.gripper_client.connect()
        self.gripper_client.initialize()
        self.gripper_state = np.array(100)  

        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'gripper_cmd': np.array(100),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        if receive_keys is None:
            example = {
                'ActualTCPPose': np.zeros((6,), dtype=np.float64),
                'ActualTCPSpeed': np.zeros((6,), dtype=np.float64),
                'GripperPosition': np.array(100),    # shape: ()
                'robot_receive_timestamp': time.time()
            }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[RTDEPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    # ========= command methods ============
    def servoL(self, pose, duration=0.1, gripper=None):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)
        if gripper is None:
            gripper = self.gripper_state

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'gripper_cmd': np.array(gripper),
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time, gripper=None):
        assert target_time > time.time()
        pose = np.array(pose)
        assert pose.shape == (6,)
        if gripper is None:
            gripper = self.gripper_state

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'gripper_cmd': np.array(gripper),  
            'target_time': target_time
        }
        self.input_queue.put(message)

    def read_state(self, state):
        robot_state = {
            'ActualTCPPose': state.actual_TCP_pose,
            'ActualTCPSpeed': state.actual_TCP_speed,
            'GripperPosition': self.gripper_state,
            'robot_receive_timestamp': time.time()
        }
        return robot_state

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= main loop in process ============
    def setp_to_list(self, sp):
        sp_list = []
        for i in range(0, 6):
            sp_list.append(sp.__dict__["input_double_register_%i" % i])
        return sp_list

    def list_to_setp(self, sp, list):
        for i in range(0, 6):
            sp.__dict__["input_double_register_%i" % i] = list[i]
        return sp

    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))
            
        conf = rtde_config.ConfigFile(self.config_filename)
        state_names, state_types = conf.get_recipe("state")
        setp_names, setp_types = conf.get_recipe("setp")
        watchdog_names, watchdog_types = conf.get_recipe("watchdog")

        con = rtde.RTDE(self.robot_ip, self.robot_port)
        con.connect()

        # setup recipes
        con.send_output_setup(state_names, state_types, frequency=self.frequency)
        setp = con.send_input_setup(setp_names, setp_types)
        watchdog = con.send_input_setup(watchdog_names, watchdog_types)

        # start data synchronization
        if not con.send_start():
            sys.exit()

        # servol mode
        watchdog.input_int_register_0 = 2

        try:
            state = con.receive()
            print("Waiting for playbot start...")
            while True:
                state = con.receive()
                if state.robot_status_bits == 3:  # robot is in RUNNING state
                    break

            curr_pose = state.actual_TCP_pose
            print(f"Current pose: {curr_pose}")
            self.list_to_setp(setp, curr_pose)
            con.send(setp)
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )    

            iter_idx = 0
            keep_running = True
            dt = 1. / self.frequency
            start_t = time.monotonic()
            while keep_running:
                t_cycle_end = start_t + iter_idx * dt

                t_now = time.monotonic()
                pose_command = pose_interp(t_now)
                self.list_to_setp(setp, pose_command)
                con.send(setp)
                con.send(watchdog)

                # update robot state
                state = con.receive()
                robot_state = self.read_state(state)
                self.ring_buffer.put(robot_state)

                # fetch command from input queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # process commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    if cmd == Command.STOP.value:
                        keep_running = False
                        break
                    elif cmd == Command.SERVOL.value:
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed
                        )
                        last_waypoint_time = t_insert
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time

                        if self.gripper_state != command['gripper_cmd']:
                            self.gripper_client.set_mode_params(
                                position=command['gripper_cmd'],
                                force=100
                            )
                            self.gripper_state = np.array(command['gripper_cmd'])

                    else:
                        keep_running = False
                        break
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1
                precise_wait(t_cycle_end)

        finally:
            con.send_pause()
            con.disconnect()
            self.ready_event.set()

            if self.verbose:
                print(f"[RTDEPositionalController] Controller process stopped.")
