from threading import Thread
import numpy as np
from panda_mujoco import ControlRobot
from utils.utils import RRT

class RRTControl(ControlRobot, RRT):
    """
    A class that combines functionalities of ControlRobot and RRT (Rapidly-Exploring Random Tree)
    to perform path planning and robot control.

    Attributes:
        Inherits attributes and methods from ControlRobot and RRT classes.
    """
    def __init__(self):
        ControlRobot.__init__(self)
        RRT.__init__(self)

    def step(self):
        """
        Performs a single operational step for both robots ('panda1' and 'panda2').
        The function includes logic for path planning, object manipulation, and returning to a home position.
        """
        # Extract positional and orientation data for the robots and objects in the environment.
        y_pos_box1 = self.data.body("box1").xpos.copy()
        y_pos_0_1 = self.data.body("hand1").xpos.copy()
        y_quat_0_1 = self.data.body("hand1").xquat.copy()
        y_quat_d_1 = np.array([0,0,1,0])  # Desired quaternion orientation for robot 1
        y_pos_box2 = self.data.body("box2").xpos.copy()
        y_pos_0_2 = self.data.body("hand2").xpos.copy()
        y_quat_0_2 = self.data.body("hand2").xquat.copy()
        y_quat_d_2 = np.array([0,0,1,0])  # Desired quaternion orientation for robot 2
        duration = 2  # Duration for movement execution
        # Iterate over each robot to perform tasks.
        for robot in ['panda1', 'panda2']:
            path1 = None
            path2 = None  
            target_occupied = False
            # Set robot-specific variables.
            if robot == 'panda1':
                y_pos_0 = y_pos_0_1
                y_pos_box = y_pos_box1
                y_quat_d = y_quat_d_1
            else:
                y_pos_0 = y_pos_0_2
                y_pos_box = y_pos_box2
                y_quat_d = y_quat_d_2
            # State-based logic for the robot's operation.
            if self.robot_states[robot] == 'still':
                self.robot_states[robot] = "plan_path_to_obj"
                print(f"{robot} planning its path to box{robot[-1]}...")
                # Plan a path to the object (box).
                path1 = RRT.path_plan(self, y_pos_0, np.array([y_pos_box[0],y_pos_box[1],y_pos_box[-1]+0.20]), robot)
                if path1:
                    print("Path found:", path1)
                    self.robot_states[robot] = "path_planned_to_obj"
                    # Add an endpoint slightly above the box.
                    end_point = np.array([y_pos_box[0], y_pos_box[1], y_pos_box[-1]+0.12])
                    path1.append(end_point)
                else:
                    print(f"No path found to box{robot[-1]}")
            if self.robot_states[robot] == "path_planned_to_obj":
                self.gripper(robot, open=True)
                self.follow_path(robot, path1, y_quat_d, duration)
                self.robot_states[robot] = "obj_reached"
            if self.robot_states[robot] == "obj_reached":
                self.hold(robot, 0.5)
                self.gripper(robot, open=False)
                print(f"{robot} got box{robot[-1]}.")
                self.hold(robot, 0.5)
                self.robot_states[robot] = "obj_caught"
            if self.robot_states[robot] == "obj_caught":
                # Return to the starting position with the object.
                reversed_path1 = path1[::-1]
                self.follow_path(robot, reversed_path1, y_quat_d, duration+2)
                print(f"{robot} moved to home position with box{robot[-1]}.")
                self.robot_states[robot] = "plan_path_to_target"
            if self.robot_states[robot] == "plan_path_to_target":
                print(f"{robot} planning its path to target...")
                # Plan a path to the target location.
                path2 = RRT.path_plan(self, y_pos_0, np.array([0,0,y_pos_box[-1]+0.25]), robot)
                if path2:
                    print("Path found:", path2)
                    self.robot_states[robot] = "path_planned_to_target"
                    # Adjust the endpoint based on target occupancy.
                    if target_occupied:
                        end_point = np.array([0,0,y_pos_box[-1]+0.12])
                    else:
                        end_point = np.array([0,0,y_pos_box[-1]+0.20])
                    path2.append(end_point)
                else:
                    print("No path found to target")
            if self.robot_states[robot] == "path_planned_to_target":
                self.follow_path(robot, path2, y_quat_d, duration+2)
                self.robot_states[robot] = "target_reached"
            if self.robot_states[robot] == "target_reached":
                self.gripper(robot, open=True)
                print(f"{robot} released box{robot[-1]}.")
                self.robot_states[robot] = "obj_released"
                target_occupied = True
            if self.robot_states[robot] == "obj_released":
                # Return to the home position.
                reversed_path2 = path2[::-1]
                self.follow_path(robot, reversed_path2, y_quat_d, duration)
                print(f"{robot} moved to home position.")
                self.robot_states[robot] = "still"
    
    def follow_path(self, robot, path, y_quat_d, duration):
        """
        Moves the robot along a given path, ensuring it maintains a specific orientation.
        Args:
            robot (str): The robot identifier ('panda1' or 'panda2').
            path (list): List of waypoints to follow.
            y_quat_d (np.array): Desired orientation (quaternion) for the robot.
            duration (float): Time duration for each movement step.
        """
        for _ in range (len(path)):
            self.move_to(robot, np.array([[path[_][0]],[path[_][1]],[path[_][-1]]]), y_quat_d, duration)
    
    def start(self):
        """
        Starts the operation by running the task sequence in a separate thread
        while simultaneously rendering the simulation environment.
        """
        step_thread = Thread(target=self.step)
        step_thread.start()
        self.render()

if __name__ == "__main__":
    controller = RRTControl()
    controller.start()