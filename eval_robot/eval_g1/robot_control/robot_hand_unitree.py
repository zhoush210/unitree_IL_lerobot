from unitree_dds_wrapper.robots.trihand.trihand_pub_cmd import UnitreeTrihand as trihand_pub
from unitree_dds_wrapper.robots.trihand.trihand_sub_state import UnitreeTrihand as trihand_sub

import numpy as np
import time
from multiprocessing import Array

import threading

class Dex3_1_Controller:
    def __init__(self, fps = 100.0):
        self.dex3_pub = trihand_pub()

        kp = np.full(7, 1.5)
        kd = np.full(7, 0.2)
        q = np.full(7,0.0)
        dq = np.full(7,0.0)
        tau = np.full(7,0.0)
        self.dex3_pub.left_hand.kp = kp
        self.dex3_pub.left_hand.kd = kd
        self.dex3_pub.left_hand.q = q 
        self.dex3_pub.left_hand.dq = dq
        self.dex3_pub.left_hand.tau = tau

        self.dex3_pub.right_hand.kp = kp
        self.dex3_pub.right_hand.kd = kd
        self.dex3_pub.right_hand.q = q
        self.dex3_pub.right_hand.dq = dq
        self.dex3_pub.right_hand.tau = tau
        self.lr_hand_state_array = Array('d', 14, lock=True)

        self.sub_state = trihand_sub()
        self.sub_state.wait_for_connection()

        self.subscribe_state_thread = threading.Thread(target=self.subscribe_state)
        self.subscribe_state_thread.start()

        self.running = True
        self.fps = fps

        print("UnitreeDex3 Controller init ok.")

    def subscribe_state(self):
        while True:
            lq,rq= self.sub_state.sub()
            self.lr_hand_state_array[:] = np.concatenate((lq,rq))
            time.sleep(0.002)
            

    def ctrl(self, left_angles, right_angles):
        """set current left, right hand motor state target q"""
        self.dex3_pub.left_hand.q  = left_angles
        self.dex3_pub.right_hand.q  = right_angles
        self.dex3_pub.pub()
        # print("hand pub ok")

    def get_current_dual_hand_q(self):
        """return current left, right hand motor state q"""
        temp_lrq = self.lr_hand_state_array[:].copy()
        return temp_lrq
    