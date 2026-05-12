import pybullet as p
import time
import pybullet_data
from urdf_models import models_data
from .robot import Panda, UR5Robotiq85, UR5Robotiq140
from .utilities import Camera
import numpy as np
import os
import yaml
from scipy.spatial.transform import Rotation as R
import numpy as np
import copy
import math

class SceneBase:

    SIMULATION_STEP_DELAY = 1 / 240.0

    def __init__(self):
        # load YCB models
        self.ycbmodels = models_data.model_lib()
        self.ycbNameList = self.ycbmodels.model_name_list
        self.flags = p.URDF_USE_INERTIA_FROM_FILE
        self.parent_path = os.path.dirname(os.path.abspath(__file__))
        self.grand_path = os.path.dirname(self.parent_path)

        # define environment
        self.physicsClient = p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # load plane and table
        self.planeID = p.loadURDF("plane.urdf")
        self.tableID = p.loadURDF(self.parent_path + "/assets/urdf/objects/table.urdf", [0.6, 0, 0], p.getQuaternionFromEuler([0, 0, np.pi / 2]))

        # load robot
        self.robot = UR5Robotiq85((0, 0, 0.63), (0, 0, 0))
        self.robot.load()
        self.robot.step_simulation = self.step_simulation
        
        # load camera
        # self.camera = Camera((1, 1, 1),
        #                      (0, 0, 0),
        #                      (0, 0, 1),
        #                      0.1, 5, (320, 320), 40)
        self.interval = 600
        
    def render(self, interval = None):
        if interval is None:
            interval = self.interval
        for _ in range(interval):
            p.stepSimulation()
            time.sleep(self.SIMULATION_STEP_DELAY)

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def get_grasp_yaw_from_object_orientation(self, quat):
        """
        计算物体局部 x 轴与世界坐标系 x 轴之间的夹角（弧度）
        输入：
            quat: 物体的四元数 [x, y, z, w]
        输出：
            yaw: float，夹角弧度值，范围 (-π/2, π/2]
        """
        rot_mat = R.from_quat(quat).as_matrix()
        local_x_axis = rot_mat[:, 0]  # 第0列就是局部 x 轴在世界坐标系下的方向

        # 投影到 XY 平面
        proj_x, proj_y = local_x_axis[0], local_x_axis[1]
        yaw = math.atan2(proj_y, proj_x)  # 与世界 x 轴的夹角
        # 确保范围在 (-π/2, π/2]
        if yaw <= -np.pi / 2:
            yaw += np.pi
        elif yaw > np.pi / 2:
            yaw -= np.pi
        return yaw
    
    # Primitive Action in Simulation
    def LookandMoveTo(self, x: str, model_id, index=0):
        '''
            x is the component of model
            如果model只有一个link, 直接获取刚体的位姿
        '''
        pose_modidy = {0: 0, 1: 0.03, 2: -0.03}
        pos, orn = None, None
        num_joints = p.getNumJoints(model_id)
        if num_joints == 0:
            pos, orn = p.getBasePositionAndOrientation(model_id)
        else:
            for joint_index in range(num_joints):
                joint_info = p.getJointInfo(model_id, joint_index)
                current_link_name = joint_info[12].decode("utf-8")  # 索引12是link名称
                if current_link_name == x:
                    # 获取link状态
                    link_state = p.getLinkState(model_id, joint_index)
                    pos, orn = link_state[0], link_state[1]  # 位置和方向
                    break
        if pos is None or orn is None:
            raise ValueError(f"Link {x} not found in model {model_id}")
        tmp_pos = list(pos)
        tmp_pos[1] += pose_modidy[index]
        pos = tuple(tmp_pos)
        # 计算yaw
        yaw = self.get_grasp_yaw_from_object_orientation(orn)

        euler = p.getEulerFromQuaternion(orn)
        target_link_pose = np.array(pos + euler)
        # 获取当前末端的位置
        current_ee_pose = self.robot.get_ee_pose()
        target_ee_pose = copy.copy(current_ee_pose)
        # 高度不变的条件下平移到目标点的上方，并转动yaw，对齐转角
        target_ee_pose[:2] = target_link_pose[:2]
        target_ee_pose[2] -= 0.05
        target_joints_state = self.robot.arm_IK(target_ee_pose)
        target_joints_state[-1] -= yaw

        self.robot.move_ee(target_joints_state, 'joint')
        self.render()

        return target_link_pose
    
    def Grab(self):
        self.robot.move_gripper(0.01)
        self.render()

    def Release(self):
        self.robot.open_gripper()
        self.render()

    def Twist(self):
        '''
            clockwise twist the object
        '''
        current_joint_states = self.robot.get_arm_joint_states()
        target_joint_states = copy.copy(current_joint_states)
        target_joint_states[-1] += np.pi / 2
        self.robot.move_ee(target_joint_states, 'joint')
        self.render()

    def Rotate(self):
        '''
            counterclockwise rotate the object
        '''
        current_joint_states = self.robot.get_arm_joint_states()
        target_joint_states = copy.copy(current_joint_states)
        target_joint_states[-1] -= np.pi / 2
        self.robot.move_ee(target_joint_states, 'joint')
        self.render()

    def Press(self):
        '''
            press the object
        '''
        self.robot.close_gripper()
        current_ee_pose = self.robot.get_ee_pose()
        target_ee_pose = copy.copy(current_ee_pose)
        target_ee_pose[2] -= 0.03
        self.robot.move_ee(target_ee_pose, 'end')
        self.render()
        target_ee_pose[2] += 0.03
        self.robot.move_ee(target_ee_pose, 'end')
        self.robot.open_gripper()
        self.render()

        

    def Pull(self):
        '''
            pull the object
        '''
        current_ee_pose = self.robot.get_ee_pose()
        target_ee_pose = copy.copy(current_ee_pose)
        count = 20
        steps = 0.24 / 20
        for i in range(count):
            target_ee_pose[0] -= steps
            self.robot.move_ee(target_ee_pose, 'end')
            self.render(self.interval // count)
        # target_ee_pose[0] -= 0.22
        # self.robot.move_ee(target_ee_pose, 'end')
        # self.render()

    def Push(self):
        '''
            push the object
        '''
        current_ee_pose = self.robot.get_ee_pose()
        target_ee_pose = copy.copy(current_ee_pose)
        count = 20
        steps = 0.24 / 20
        for i in range(count):
            target_ee_pose[0] += steps
            self.robot.move_ee(target_ee_pose, 'end')
            self.render(self.interval // count)
        # target_ee_pose[0] += 0.22
        # self.robot.move_ee(target_ee_pose, 'end')
        # self.render()

    def Pour(self):
        '''
            pour the container
        '''

    def MoveDown(self, DownHeight):
        '''
            DownHeight is the height robot need to move down
        '''
        current_ee_pose = self.robot.get_ee_pose()
        target_ee_pose = copy.copy(current_ee_pose)
        target_ee_pose[2] -= DownHeight
        self.robot.move_ee(target_ee_pose, 'end')
        self.render()

    def Hold(self):
        '''
            hold the object
        '''
        self.robot.MoveToDefaultPose()
        self.render()
    
    def TwistandRotate(self):
        '''
            stir the container
        '''
        for t in range(3):
            self.Twist()
            self.render()
            self.Rotate()
            self.render()

    def LiftUp(self, UpHeight):
        '''
            lift up
        '''
        current_ee_pose = self.robot.get_ee_pose()
        target_ee_pose = copy.copy(current_ee_pose)
        target_ee_pose[2] += UpHeight
        self.robot.move_ee(target_ee_pose, 'end')
        self.render()

    def Endl(self):
        '''
            return to the default position
        '''
        self.robot.MoveToDefaultPose()
        self.render()
    
    def get_robot_current_pose(self):
        return self.robot.get_ee_pose()

    def reset_robot(self):
        '''
            重置机械臂
        '''
        self.robot.reset()
        self.robot.move_ee([0.5, 0.0, 1.13, 0, np.pi/2, 0], 'end')
        self.robot.open_gripper()
        self.render()

class NormalScene(SceneBase):
    def __init__(self):
        super().__init__()
        self.config = self.load_config(self.grand_path + '/Scene.yaml')
        self.models = self.load_models()
    
    def load_models(self):
        models = {}
        for model_name, model_config in self.config['NormalScene'].items():
            model_position = model_config['Pose']['position']
            model_orientation = model_config['Pose']['orientation']
            if 'Drawer' in model_name:
                model_path = model_config['Path']
                model_id = p.loadURDF(model_path, model_position, p.getQuaternionFromEuler(model_orientation), useFixedBase=True)
                models[model_name] = model_id
            elif 'Building_Blocks' in model_name:
                model_path = model_config['Path']
                model_id = p.loadURDF(model_path, model_position, p.getQuaternionFromEuler(model_orientation))
                models[model_name] = model_id
            else:
                ycbmodelIndex = model_config['index']
                model_path = self.ycbmodels[self.ycbNameList[ycbmodelIndex]]
                model_id = p.loadURDF(model_path, model_position, p.getQuaternionFromEuler(model_orientation), flags=self.flags)
                models[model_name] = model_id
            p.changeDynamics(model_id, 0,
                     lateralFriction=2.0,
                     rollingFriction=0.1,
                     spinningFriction=0.1,
                     contactProcessingThreshold=0)
        return models
 

if __name__ == "__main__":
    scene = SceneBase()
    scene.robot.reset()
    while True:
        scene.robot.move_ee([0.5, 0.0, 1.13, 0, np.pi/2, np.pi/2], 'end')
        scene.robot.open_gripper()
        for _ in range(120):
            scene.step_simulation()
        time.sleep(scene.SIMULATION_STEP_DELAY)
