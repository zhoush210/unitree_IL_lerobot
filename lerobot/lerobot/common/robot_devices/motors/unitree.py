import time
import math
import sys
import numpy as np
import json
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc
sys.path.append('./z1_sdk/lib')
import unitree_arm_interface as uai
from unitree_dds_wrapper.publisher import Publisher
from unitree_dds_wrapper.subscription import Subscription
from unitree_dds_wrapper.idl import std_msgs
from unitree_dl_utils.robots.aloha.aloha import MotorStates

np.set_printoptions(precision=3, suppress=True)

dt = 1 / 250.
HAS_GRIPPER = True
V_MAX = 0.5
nq = 7 if HAS_GRIPPER else 6
GPRPPER_TAU_MAX = 5.
GRIPPER_KP = 20. 
DELTA_GRIPPER_CMD = GPRPPER_TAU_MAX / GRIPPER_KP / 25.6 
DELTA_GRIPPER_CMD = DELTA_GRIPPER_CMD * 20  

class LowCmd(Subscription):
    def __init__(self):
        super().__init__(message=std_msgs.msg.dds_.String_, topic="rt/z1/cmd")
        self.q = np.zeros(nq)
    
    def post_communication(self):
        data = json.loads(self.msg.data)
        new_q = np.array(data["q"])
        assert len(new_q) == nq
        self.q = new_q

class LowState(Publisher):
    def __init__(self):
        super().__init__(message=std_msgs.msg.dds_.String_, topic="rt/z1/state")
        self.data = {
            "q": [],
            "qd": [],
            "endPose": []
        }

    def pre_communication(self): 
        assert len(self.data["q"]) == nq
        assert len(self.data["qd"]) == nq
        assert len(self.data["endPose"]) == 6
        self.msg.data = json.dumps(self.data)

def get_group_sync_key(data_name, motor_names):
    group_key = f"{data_name}_" + "_".join(motor_names)
    return group_key

def get_log_name(var_name, fn_name, data_name, motor_names):
    motor_names = "test"
    group_key = get_group_sync_key(data_name, motor_names)
    log_name = f"{var_name}_{fn_name}_{group_key}"
    return log_name

class Z1:

    def __init__(
        self,
    ):
        self.z1 =  uai.ArmInterface(hasGripper=False)
        self.z1.setFsmLowcmd()
        self.z1model:uai.Z1Model = self.z1._ctrlComp.armModel

        print(f"==============================================================================")
        print(f"old kp:\t{self.z1.lowcmd.kp}  \nkd:\t{self.z1.lowcmd.kd}\n")
        kp = np.array([4,   8,   4,   4,   4,  4])
        kd = np.array([400, 800, 400, 400, 400, 400])
        self.z1.lowcmd.setControlGain(kp, kd)
        self.z1.lowcmd.setGripperGain(4, 100)            
        self.z1.sendRecv()                               
        print(f"kp new:\t{self.z1.lowcmd.kp}  \nkd new:\t{self.z1.lowcmd.kd}\n")
        print(f"==============================================================================\n\n")

        print(f"start running")

        # real_freq = args.frequency
        self.dt_arm = 1/250.0                    
        self.dt_data = 1.0 / 30     
        self.interp_ratio = 4                    
        print(f"dt_arm:{self.dt_arm}  dt_data:{self.dt_data}  interp_ratio:{self.interp_ratio}")

        self.z1_qd = np.zeros(6, dtype=np.float32) 
        self.z1_qdd = np.zeros(6, dtype=np.float32) 
        self.z1_ftip = np.zeros(6, dtype=np.float32) 
        self.q_prev = np.concatenate([self.z1.lowstate.getQ(), [self.z1.lowstate.getGripperQ()]], axis=0)

        self.motors = []
        self.logs = {}

    def connect(self):
        self.is_connected = True

    def reconnect(self):
        self.is_connected = True

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    def read(self, data_name, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"Z1() is not connected. You need to run `motors_bus.connect()`."
            )

        start_time = time.perf_counter()

        values = []
        value = self.z1.lowstate.getQ().tolist()   + [self.z1.lowstate.getGripperQ()]
        values.append(value)

        values = np.array(values)

        # log the number of seconds it took to read the data from the motors
        delta_ts_name = get_log_name("delta_timestamp_s", "read", data_name, motor_names)
        self.logs[delta_ts_name] = time.perf_counter() - start_time

        # log the utc time at which the data was received
        ts_utc_name = get_log_name("timestamp_utc", "read", data_name, motor_names)
        self.logs[ts_utc_name] = capture_timestamp_utc()

        return values

    def joint_poses_interp(self, poses, ratio=1.0):
        steps = poses.shape[0]
        joints =  poses.shape[1]
        steps_new = math.ceil(steps*ratio)
        xp = np.linspace(0, steps, steps) 
        x = np.linspace(0, steps, steps_new) 

        poses_new = None
        for j in range(joints):
            fp = poses[:, j]
            y = np.interp(x, xp, fp)
            poses_new = y.reshape(steps_new,1) if j==0 else np.hstack((poses_new, y.reshape(steps_new,1)))

        return poses_new


    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"Z1() is not connected. You need to run `motors_bus.connect()`."
            )

        start_time = time.perf_counter()
        # q_target = values[:6]
        q_target = values
        q_target[[3,4]] = q_target[[4,3]]
        q_target[-1] = -q_target[-1]

        q_interped = np.stack([self.q_prev, q_target], axis=0) 
        q_interped = self.joint_poses_interp(q_interped, ratio=self.interp_ratio)  
        for q in q_interped:
            time_arm = time.time()
            self.z1.q = q[0:6]
            self.z1.qd = self.z1_qd
            self.z1.tau = self.z1model.inverseDynamics(self.z1.q, self.z1.qd, self.z1_qdd, self.z1_ftip) 
            self.z1.gripperQ = np.clip(q[6], self.z1.lowstate.getGripperQ() - DELTA_GRIPPER_CMD, self.z1.lowstate.getGripperQ() + DELTA_GRIPPER_CMD) 
            self.z1.setArmCmd(self.z1.q, self.z1.qd, self.z1.tau)
            self.z1.setGripperCmd(self.z1.gripperQ, self.z1.gripperQd, self.z1.gripperTau)
            self.z1.sendRecv() 
            time.sleep(max(0, self.dt_arm/2.0 - (time.time() - time_arm)))  
        self.q_prev = q_target
                
        # log the number of seconds it took to write the data to the motors
        delta_ts_name = get_log_name("delta_timestamp_s", "write", data_name, motor_names)
        self.logs[delta_ts_name] = time.perf_counter() - start_time

        # TODO(rcadene): should we log the time before sending the write command?
        # log the utc time when the write has been completed
        ts_utc_name = get_log_name("timestamp_utc", "write", data_name, motor_names)
        self.logs[ts_utc_name] = capture_timestamp_utc()

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"Z1() is not connected. Try running `motors_bus.connect()` first."
            )
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()


class  Aloha_mini:

    def __init__(
        self,
    ):
        self.d1 = MotorStates("aloha_mini", window_size=20) 
        self.d1.wait_for_connection()
        
        self.motors = []
        self.is_connected = False
        self.logs = {}

    def connect(self):
        self.is_connected = True

    def reconnect(self):
        self.is_connected = True

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    def read(self, data_name, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"Aloha_mini() is not connected. You need to run `motors_bus.connect()`."
            )
        start_time = time.perf_counter()

        values = []
        value = self.d1.q
        values.append(value)

        values = np.array(values)

        # log the number of seconds it took to read the data from the motors
        delta_ts_name = get_log_name("delta_timestamp_s", "read", data_name, motor_names)
        self.logs[delta_ts_name] = time.perf_counter() - start_time

        # log the utc time at which the data was received
        ts_utc_name = get_log_name("timestamp_utc", "read", data_name, motor_names)
        self.logs[ts_utc_name] = capture_timestamp_utc()

        return values

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"Aloha_mini() is not connected. Try running `motors_bus.connect()` first."
            )

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
