import pybullet as p
import pybullet_data
import time
import numpy as np
from urdf_models import models_data

models = models_data.model_lib()
flags = p.URDF_USE_INERTIA_FROM_FILE


physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF("plane.urdf")
table_id = p.loadURDF("table/table.urdf", [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))
# tray_id = p.loadURDF("tray/tray.urdf", [0.5, 0.9, 0.6], p.getQuaternionFromEuler([0, 0, 0]))
drawer_id = p.loadURDF("./assets/urdf/objects/drawer.urdf", [0.7, 0, 1.0], p.getQuaternionFromEuler([0, 0, - np.pi / 2]))
# mug_id = p.loadURDF("/home/chpppy/Desktop/ycb-tools/models/ycb/013_apple/apple.urdf", [0.7, 0.5, 1.0], p.getQuaternionFromEuler([0, 0, 0]))
ug_id = p.loadURDF(models["plastic_apple"], [0.7, 0.5, 1.0], flags=flags)
while True:
    p.stepSimulation()
    time.sleep(1./60.)
p.disconnect()