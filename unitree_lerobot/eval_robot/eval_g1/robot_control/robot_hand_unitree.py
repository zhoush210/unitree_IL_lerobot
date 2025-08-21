# for dex3-1
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_                               # idl
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
# for gripper
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_                           # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

import numpy as np
from enum import IntEnum
import time
import os
import sys
import threading
from multiprocessing import Process, shared_memory, Array, Lock

from unitree_lerobot.utils.weighted_moving_filter import WeightedMovingFilter


unitree_tip_indices = [4, 9, 14] # [thumb, index, middle] in OpenXR
Dex3_Num_Motors = 7
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"


class Dex3_1_Controller:
    def __init__(self, right_hand_state_array=None, left_hand_state_array=None, dual_hand_action_array=None, fps=100.0, Unit_Test=False):
        self.fps = fps
        self.Unit_Test = Unit_Test

        # 初始化手爪状态和动作数组
        self.left_hand_state_array, self.right_hand_state_array = right_hand_state_array,left_hand_state_array
        self.left_hand_action_array = dual_hand_action_array  # 左手动作
        self.right_hand_action_array = dual_hand_action_array  # 右手动作

        # 初始化手爪控制器的发布器和订阅器
        self.LeftHandCmb_publisher = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
        self.RightHandCmb_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(kTopicDex3LeftState, HandState_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicDex3RightState, HandState_)
        self.RightHandState_subscriber.Init()

        # 启动线程接收手爪状态
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        # 等待手爪状态初始化完成
        while True:
            if any(self.left_hand_state_array) and any(self.right_hand_state_array):
                break
            time.sleep(0.01)
            print("[Dex3_1_Controller] Waiting for DDS data...")

        # 初始化控制消息（只做一次）
        self._initialize_control_messages()

        print("Dex3_1_Controller initialization complete!")

    def _initialize_control_messages(self):
        """初始化控制消息，只调用一次"""
        q = 0.0
        dq = 0.0
        tau = 0.0
        
        kp = 1.5
        kd = 0.2
        
        # kp = 0.0
        # kd = 0.0

        # initialize dex3-1's left hand cmd msg
        self.left_msg  = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Left_JointIndex:
            ris_mode = self._RIS_Mode(id = id, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.left_msg.motor_cmd[id].mode = motor_mode
            self.left_msg.motor_cmd[id].q    = q
            self.left_msg.motor_cmd[id].dq   = dq
            self.left_msg.motor_cmd[id].tau  = tau
            self.left_msg.motor_cmd[id].kp   = 0.0
            self.left_msg.motor_cmd[id].kd   = 0.0

        # initialize dex3-1's right hand cmd msg
        self.right_msg = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Right_JointIndex:
            ris_mode = self._RIS_Mode(id = id, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.right_msg.motor_cmd[id].mode = motor_mode  
            self.right_msg.motor_cmd[id].q    = q
            self.right_msg.motor_cmd[id].dq   = dq
            self.right_msg.motor_cmd[id].tau  = tau
            self.right_msg.motor_cmd[id].kp   = kp
            self.right_msg.motor_cmd[id].kd   = kd

    def _subscribe_hand_state(self):
        while True:
            left_hand_msg  = self.LeftHandState_subscriber.Read()
            right_hand_msg = self.RightHandState_subscriber.Read()
            if left_hand_msg is not None and right_hand_msg is not None:
                # Update left hand state
                for idx, id in enumerate(Dex3_1_Left_JointIndex):
                    self.left_hand_state_array[idx] = left_hand_msg.motor_state[id].q
                # Update right hand state
                for idx, id in enumerate(Dex3_1_Right_JointIndex):
                    self.right_hand_state_array[idx] = right_hand_msg.motor_state[id].q
            time.sleep(0.002)
    
    class _RIS_Mode:
        def __init__(self, id=0, status=0x01, timeout=0):
            self.motor_mode = 0
            self.id = id & 0x0F  # 4 bits for id
            self.status = status & 0x07  # 3 bits for status
            self.timeout = timeout & 0x01  # 1 bit for timeout

        def _mode_to_uint8(self):
            self.motor_mode |= (self.id & 0x0F)
            self.motor_mode |= (self.status & 0x07) << 4
            self.motor_mode |= (self.timeout & 0x01) << 7
            return self.motor_mode

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """set current left, right hand motor state target q"""
        for idx, id in enumerate(Dex3_1_Left_JointIndex):
            self.left_msg.motor_cmd[id].q = left_q_target[idx]
        for idx, id in enumerate(Dex3_1_Right_JointIndex):
            self.right_msg.motor_cmd[id].q = right_q_target[idx]

        self.LeftHandCmb_publisher.Write(self.left_msg)
        self.RightHandCmb_publisher.Write(self.right_msg)

class Dex3_1_Left_JointIndex(IntEnum):
    kLeftHandThumb0 = 0
    kLeftHandThumb1 = 1
    kLeftHandThumb2 = 2
    kLeftHandMiddle0 = 3
    kLeftHandMiddle1 = 4
    kLeftHandIndex0 = 5
    kLeftHandIndex1 = 6

class Dex3_1_Right_JointIndex(IntEnum):
    kRightHandThumb0 = 0
    kRightHandThumb1 = 1
    kRightHandThumb2 = 2
    kRightHandIndex0 = 3
    kRightHandIndex1 = 4
    kRightHandMiddle0 = 5
    kRightHandMiddle1 = 6


unitree_gripper_indices = [4, 9] # [thumb, index]
Gripper_Num_Motors = 2
kTopicGripperCommand = "rt/unitree_actuator/cmd"
kTopicGripperState = "rt/unitree_actuator/state"

class Gripper_Controller:
    def __init__(self, left_hand_array, right_hand_array, dual_gripper_data_lock = None, dual_gripper_state_out = None, dual_gripper_action_out = None, 
                       filter = True, fps = 200.0, Unit_Test = False):
        """
        [note] A *_array type parameter requires using a multiprocessing Array, because it needs to be passed to the internal child process

        left_hand_array: [input] Left hand skeleton data (required from XR device) to control_thread

        right_hand_array: [input] Right hand skeleton data (required from XR device) to control_thread

        dual_gripper_data_lock: Data synchronization lock for dual_gripper_state_array and dual_gripper_action_array

        dual_gripper_state: [output] Return left(1), right(1) gripper motor state

        dual_gripper_action: [output] Return left(1), right(1) gripper motor action

        fps: Control frequency

        Unit_Test: Whether to enable unit testing
        """

        print("Initialize Gripper_Controller...")

        self.fps = fps
        self.Unit_Test = Unit_Test
        if filter:
            self.smooth_filter = WeightedMovingFilter(np.array([0.5, 0.3, 0.2]), Gripper_Num_Motors)
        else:
            self.smooth_filter = None

        if self.Unit_Test:
            ChannelFactoryInitialize(0)
 
        # initialize handcmd publisher and handstate subscriber
        self.GripperCmb_publisher = ChannelPublisher(kTopicGripperCommand, MotorCmds_)
        self.GripperCmb_publisher.Init()

        self.GripperState_subscriber = ChannelSubscriber(kTopicGripperState, MotorStates_)
        self.GripperState_subscriber.Init()

        self.dual_gripper_state = [0.0] * len(Gripper_JointIndex)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_gripper_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while True:
            if any(state != 0.0 for state in self.dual_gripper_state):
                break
            time.sleep(0.01)
            print("[Gripper_Controller] Waiting to subscribe dds...")

        self.gripper_control_thread = threading.Thread(target=self.control_thread, args=(left_hand_array, right_hand_array, self.dual_gripper_state,
                                                                                         dual_gripper_data_lock, dual_gripper_state_out, dual_gripper_action_out))
        self.gripper_control_thread.daemon = True
        self.gripper_control_thread.start()

        print("Initialize Gripper_Controller OK!\n")

    def _subscribe_gripper_state(self):
        while True:
            gripper_msg  = self.GripperState_subscriber.Read()
            if gripper_msg is not None:
                for idx, id in enumerate(Gripper_JointIndex):
                    self.dual_gripper_state[idx] = gripper_msg.states[id].q
            time.sleep(0.002)
    
    def ctrl_dual_gripper(self, gripper_q_target):
        """set current left, right gripper motor state target q"""
        for idx, id in enumerate(Gripper_JointIndex):
            self.gripper_msg.cmds[id].q = gripper_q_target[idx]

        self.GripperCmb_publisher.Write(self.gripper_msg)
        # print("gripper ctrl publish ok.")
    
    def control_thread(self, left_hand_array, right_hand_array, dual_gripper_state_in, dual_hand_data_lock = None, 
                             dual_gripper_state_out = None, dual_gripper_action_out = None):
        self.running = True

        DELTA_GRIPPER_CMD = 0.18         # The motor rotates 5.4 radians, the clamping jaw slide open 9 cm, so 0.6 rad <==> 1 cm, 0.18 rad <==> 3 mm
        THUMB_INDEX_DISTANCE_MIN = 0.05  # Assuming a minimum Euclidean distance is 5 cm between thumb and index.
        THUMB_INDEX_DISTANCE_MAX = 0.07  # Assuming a maximum Euclidean distance is 9 cm between thumb and index.
        LEFT_MAPPED_MIN  = 0.0           # The minimum initial motor position when the gripper closes at startup.
        RIGHT_MAPPED_MIN = 0.0           # The minimum initial motor position when the gripper closes at startup.
        # The maximum initial motor position when the gripper closes before calibration (with the rail stroke calculated as 0.6 cm/rad * 9 rad = 5.4 cm).
        LEFT_MAPPED_MAX = LEFT_MAPPED_MIN + 5.40 
        RIGHT_MAPPED_MAX = RIGHT_MAPPED_MIN + 5.40
        left_target_action  = (LEFT_MAPPED_MAX - LEFT_MAPPED_MIN) / 2.0
        right_target_action = (RIGHT_MAPPED_MAX - RIGHT_MAPPED_MIN) / 2.0

        dq = 0.0
        tau = 0.0
        kp = 5.00
        kd = 0.05
        # initialize gripper cmd msg
        self.gripper_msg  = MotorCmds_()
        self.gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(len(Gripper_JointIndex))]
        for id in Gripper_JointIndex:
            self.gripper_msg.cmds[id].dq  = dq
            self.gripper_msg.cmds[id].tau = tau
            self.gripper_msg.cmds[id].kp  = kp
            self.gripper_msg.cmds[id].kd  = kd

        try:
            while self.running:
                start_time = time.time()

                left_target_action  = np.array(left_hand_array[:]).copy()
                right_target_action = np.array(right_hand_array[:]).copy()

                # get current dual gripper motor state
                dual_gripper_state = np.array(dual_gripper_state_in[:])

                # clip dual gripper action to avoid overflow
                left_actual_action  = np.clip(left_target_action,  dual_gripper_state[1] - DELTA_GRIPPER_CMD, dual_gripper_state[1] + DELTA_GRIPPER_CMD) 
                right_actual_action = np.clip(right_target_action, dual_gripper_state[0] - DELTA_GRIPPER_CMD, dual_gripper_state[0] + DELTA_GRIPPER_CMD)

                dual_gripper_action = np.array([right_actual_action, left_actual_action])

                if self.smooth_filter:
                    self.smooth_filter.add_data(dual_gripper_action)
                    dual_gripper_action = self.smooth_filter.filtered_data

                if dual_gripper_state_out and dual_gripper_action_out:
                    with dual_hand_data_lock:
                        dual_gripper_state_out[:] = dual_gripper_state - np.array([RIGHT_MAPPED_MIN, LEFT_MAPPED_MIN])
                        dual_gripper_action_out[:] = dual_gripper_action - np.array([RIGHT_MAPPED_MIN, LEFT_MAPPED_MIN])
                
                # print(f"LEFT: euclidean:{left_euclidean_distance:.4f} \tstate:{dual_gripper_state_out[1]:.4f}\
                #       \ttarget_action:{right_target_action - RIGHT_MAPPED_MIN:.4f} \tactual_action:{dual_gripper_action_out[1]:.4f}")
                # print(f"RIGHT:euclidean:{right_euclidean_distance:.4f} \tstate:{dual_gripper_state_out[0]:.4f}\
                #       \ttarget_action:{left_target_action - LEFT_MAPPED_MIN:.4f} \tactual_action:{dual_gripper_action_out[0]:.4f}")

                self.ctrl_dual_gripper(dual_gripper_action)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            print("Gripper_Controller has been closed.")

class Gripper_JointIndex(IntEnum):
    kLeftGripper = 0
    kRightGripper = 1

