from threading import Thread
import numpy as np
from panda_mujoco import ControlRobot

class DemoControl(ControlRobot):
    """
    DemoControl extends the ControlRobot class to implement a specific task demonstration 
    for two Panda robots in a MuJoCo simulation. It defines a sequence of actions such as 
    moving, gripping, and holding objects to showcase the capabilities of the robot controllers.
    """
    def __init__(self):
        ControlRobot.__init__(self)

    def step(self):
        """
        Executes a sequence of predefined tasks for the two robots, 'panda1' and 'panda2'. 
        Each robot moves to specific positions, manipulates objects using the gripper, 
        and returns to its initial position.
        
        The task involves:
        1. Moving 'panda1' to pick up 'box1', placing it at a target position, and returning.
        2. Moving 'panda2' to pick up 'box2', placing it at a target position, and returning.
        """
        y_pos_box1 = self.data.body("box1").xpos.copy()
        y_pos_0_1 = self.data.body("hand1").xpos.copy()
        y_quat_0_1 = self.data.body("hand1").xquat.copy()
        y_quat_d_1 = np.array([0,0,1,0])    
        y_pos_box2 = self.data.body("box2").xpos.copy()
        y_pos_0_2 = self.data.body("hand2").xpos.copy()
        y_quat_0_2 = self.data.body("hand2").xquat.copy()
        y_quat_d_2 = np.array([0,0,1,0])          
        self.hold("panda1", 0.5)
        self.gripper("panda1", open=True)
        self.move_to("panda1", np.array([[y_pos_box1[0]],[y_pos_box1[1]],[y_pos_0_1[-1]]]),y_quat_d_1,1)
        self.move_to("panda1", np.array([[y_pos_box1[0]],[y_pos_box1[1]],[y_pos_box1[-1]+0.12]]),y_quat_d_1,2)
        self.hold("panda1", 0.5)
        self.gripper("panda1", open=False)
        self.move_to("panda1", y_pos_0_1.reshape((3,1)), y_quat_d_1, 5)
        self.move_to("panda1", np.array([[0.05],[0],[y_pos_0_1[-1]]]),y_quat_d_1,5)
        self.move_to("panda1", np.array([[0],[0],[y_pos_box1[-1]+0.12]]),y_quat_d_1,2)
        self.gripper("panda1", open=True)
        self.move_to("panda1", np.array([[0.05],[0],[y_pos_0_1[-1]]]),y_quat_d_1,5)
        self.move_to("panda1", y_pos_0_1.reshape((3,1)), y_quat_d_1, 1)

        y_pos_box1 = self.data.body("box1").xpos.copy()
       
        self.gripper("panda2", open=True)
        self.move_to("panda2", np.array([[y_pos_box2[0]],[y_pos_box2[1]],[y_pos_0_2[-1]]]),y_quat_d_2,2)
        self.move_to("panda2", np.array([[y_pos_box2[0]],[y_pos_box2[1]],[y_pos_box2[-1]+0.12]]),y_quat_d_2,2)
        self.hold("panda2", 0.5)
        self.gripper("panda2", open=False)
        self.hold("panda2", 0.5)
        self.move_to("panda2", y_pos_0_2.reshape((3,1)), y_quat_d_2, 5)
        self.move_to("panda2", np.array([[-0.05],[0],[y_pos_0_2[-1]]]),y_quat_d_2,5)
        self.move_to("panda2", np.array([[0],[0],[y_pos_box2[-1]+0.20]]),y_quat_d_2,2)
        self.gripper("panda2", open=True)
        self.move_to("panda2", np.array([[-0.05],[0],[y_pos_0_2[-1]]]),y_quat_d_2,5)
        self.move_to("panda2", y_pos_0_2.reshape((3,1)), y_quat_d_2, 1)

    def start(self):
        """
        Starts the demo by running the task sequence in a separate thread (for 'step' function)
        while simultaneously rendering the simulation environment.
        """
        step_thread = Thread(target=self.step)
        step_thread.start()
        self.render()

if __name__ == "__main__":
    controller = DemoControl()
    controller.start()