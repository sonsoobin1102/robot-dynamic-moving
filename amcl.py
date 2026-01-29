import numpy as np

class SimulatedAMCL:
    def __init__(self):
        self.x_noise = 0.005 
        self.y_noise = 0.005
        self.yaw_noise = 0.005

    def get_estimated_pose(self, true_x):
        ex = true_x[0] + np.random.normal(0, self.x_noise)
        ey = true_x[1] + np.random.normal(0, self.y_noise)
        eyaw = true_x[2] + np.random.normal(0, self.yaw_noise)
        return np.array([ex, ey, eyaw, true_x[3], true_x[4]])
