from common.f710_gamepad import F710GamePad
from shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
import multiprocessing as mp
import numpy as np
import time

# API class for F710 gamepad as a SpaceMouse
class F710SpaceMouse(mp.Process):
    """
    A class to handle the F710 gamepad as a SpaceMouse.
    It uses the F710GamePad class to get input values.
    """
    
    def __init__(self,
                 shm_manager,
                 frequency=200,
                 dtype=np.float32,
                 n_buttons=2,
                 get_max_k=32):
        super().__init__()

        self.gamepad = F710GamePad()
            

        example = {
            # 3 translation, 3 rotation, 1 period
            'motion_event': np.zeros((6,), dtype=np.int64),
            'button_A_state': np.zeros((1,), dtype=bool),
            'button_Y_state': np.zeros((1,), dtype=bool),
            'button_X_state': np.zeros((1,), dtype=bool),
            'button_L': np.zeros((1,), dtype=bool),
            'button_R': np.zeros((1,), dtype=bool),
            'gripper_cmd': np.array(100),
            'receive_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager, 
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )
        
        self.gripper = False
        self.previous_button_state = False
        self.button_a = False
        self.previous_button_a_state = False
        self.button_y = False
        self.previous_button_y_state = False
        self.button_x = False
        self.previous_button_x_state = False
        self.button_l = False
        self.previous_button_l_state = False
        self.button_r = False
        self.previous_button_r_state = False
        self.max_value = 500
        self.deadzone = 0.05  # deadzone for joystick input

        # shared variables
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.ring_buffer = ring_buffer

        # copied variables
        self.frequency = frequency
        self.dtype = dtype
        self.n_buttons = n_buttons
    
    def GetInput(self, joyL=1, joyR=1, trigL=1, trigR=1, buttons=1, hat=1, joyL_max=100):
        return self.gamepad.GetInput(joyL=1, joyR=1, trigL=1, trigR=1, buttons=1, hat=1, joyL_max=100)
    
    def input_transform(self, input):
        '''[[0.0, 0.0], [0.0, 0.0], 0.0, 0.0, [0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]'''
        x, y, rx, ry = input[0][0], input[0][1], input[1][1], -input[1][0]

        z = 60*input[5][4] + (-60)*input[5][6] 
        rz = 120*input[5][5] + (-120)*input[5][7]
        button_state = np.zeros((self.n_buttons,), dtype=bool)
        button_state[0] = np.array(input[5][0], dtype=bool)
        button_state[1] = np.array(input[5][1], dtype=bool)

        current_x_state = bool(input[5][0])
        if current_x_state and not self.previous_button_x_state:
            self.button_x = not self.button_x
        self.previous_button_x_state = current_x_state

        current_l_state = input[4][0] == -1
        if current_l_state and not self.previous_button_l_state:
            self.button_l = not self.button_l
        self.previous_button_l_state = current_l_state

        current_r_state = input[4][0] == 1
        if current_r_state and not self.previous_button_r_state:
            self.button_r = not self.button_r
        self.previous_button_r_state = current_r_state

        current_a_state = bool(input[5][1])
        if current_a_state and not self.previous_button_a_state:
            self.button_a = not self.button_a
        self.previous_button_a_state = current_a_state

        current_y_state = bool(input[5][3])
        if current_y_state and not self.previous_button_y_state:
            self.button_y = not self.button_y
        self.previous_button_y_state = current_y_state

        current_button_state = bool(input[5][2])
        if current_button_state and not self.previous_button_state:
            self.gripper = not self.gripper
        self.previous_button_state = current_button_state

        motion_event = np.array([x, y, z, rx, ry, rz], dtype=np.int64)

        return motion_event, button_state, np.array(self.button_a), np.array(self.button_y), np.array(self.button_x), np.array(self.button_l), np.array(self.button_r)

    #========== start stop API ===========

    def start(self, wait=True):
        super().start()
        if wait:
            self.ready_event.wait()
    
    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.join()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ======= get state APIs ==========

    def get_motion_state(self):
        state = self.ring_buffer.get()
        state = np.array(state['motion_event'][:6], 
            dtype=self.dtype) / self.max_value
        is_dead = (-self.deadzone < state) & (state < self.deadzone)
        state[is_dead] = 0
        return state
     
    def get_motion_state_transformed(self):
        state = self.get_motion_state()
        tf_state = np.zeros_like(state)
        tf_state[:3] = state[:3]
        tf_state[3:] = state[3:]

        all_state = self.ring_buffer.get()
        gripper_cmd = all_state['gripper_cmd']

        return tf_state, [all_state['button_A_state'], all_state['button_Y_state'], all_state['button_X_state'], all_state['button_L'], all_state['button_R']], gripper_cmd

    def get_button_state(self):
        state = self.ring_buffer.get()
        return state['button_A_state']
    
    def is_button_pressed(self, button_id):
        return self.get_button_state()[button_id]
    
    # ========= main loop ==========
    def run(self):
        try:

            motion_event = np.zeros((6,), dtype=np.int64)
            button_state = np.zeros((self.n_buttons,), dtype=bool)
            # send one message immediately so client can start reading
            self.ring_buffer.put({
                'motion_event': motion_event,
                'receive_timestamp': time.time()
            })
            self.ready_event.set()  

            while not self.stop_event.is_set():
                # get input from gamepad
                inputs = self.GetInput(joyL=1, joyR=1, trigL=1, trigR=1, buttons=1, hat=1, joyL_max=100)

                # print("Inputs received:", inputs)     # Debugging line to check inputs

                # update motion event
                motion_event, button_b, button_a, button_y, button_x, button_l, button_r = self.input_transform(inputs)

                self.ring_buffer.put({
                    'motion_event': motion_event,
                    'button_A_state': button_a,
                    'button_Y_state': button_y,
                    'button_X_state': button_x,
                    'button_L': button_l,
                    'button_R': button_r,
                    'gripper_cmd': self.gripper * np.array(100),
                    'receive_timestamp': time.time()
                })

        finally:
            self.ready_event.clear()
            print("F710SpaceMouse process stopped.")

if __name__ == '__main__':
    from multiprocessing.managers import SharedMemoryManager

    # Example usage
    with SharedMemoryManager() as shm_manager:

        with F710SpaceMouse(shm_manager) as space_mouse:    # with automatically calls __enter__ and __exit__

                while True:
                    sm_state = space_mouse.get_motion_state_transformed()
                    print("SpaceMouse state:", sm_state)
            