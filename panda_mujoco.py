import time
from threading import Thread
import glfw
import mujoco
import numpy as np
from OpenGL.GL import glReadPixels, GL_RGB, GL_UNSIGNED_BYTE
import imageio
from utils.utils import Kinematics

class ControlRobot(Kinematics):
    """
    This class primarily provides a framework to:
        1. Set up the simulation environment for the Panda robots.
        2. Control the robot's movements programmatically.
        3. Render the environment for visualization and interaction.
    """
    def __init__(self):
        """
        Initialize the ControlRobot class by setting up the simulation model, 
        rendering configurations, camera, and scene.
        """
        Kinematics.__init__(self)
        self.height, self.width = 480, 640  # Rendering window resolution.
        self.fps = 30  # Rendering framerate.
        self.model = mujoco.MjModel.from_xml_path("simulation_env/world(box).xml")
        self.data = mujoco.MjData(self.model)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.run = True
        self.robot_states = {
            "panda1": "still",
            "panda2": "still"
        }
        # checking for keyframes
        for key in range(self.model.nkey):
            mujoco.mj_resetDataKeyframe(self.model, self.data, key)
            mujoco.mj_forward(self.model, self.data)

    def gripper(self, robot, open=True):
        """
        Control the gripper of the robot.
        Args:
            robot (str): Identifier for the robot (e.g., 'panda1' or 'panda2').
            open (bool): If True, opens the gripper; otherwise, closes it.
        """
        for i in range(1000):
            if open:
                self.data.actuator(f"actuator8_{robot[-1]}").ctrl = 255
            else:
                self.data.actuator(f"actuator8_{robot[-1]}").ctrl = 0
            mujoco.mj_step(self.model, self.data) # Advance the simulation
            time.sleep(1e-3)
        print(f"gripper opened for {robot}" if open else f"gripper closed for {robot}")
    
    def move_to(self, robot, y_pos_d, y_quat_d, length):
        """
        Move the robot's end-effector to a desired position and orientation.
        Args:
            robot (str): Identifier for the robot (e.g., 'panda1' or 'panda2').
            y_pos_d (ndarray): Desired position of the end-effector (3x1 array).
            y_quat_d (ndarray): Desired orientation quaternion of the end-effector (4x1 array).
            length (float): Duration of the movement in seconds.
        """
        num_step = 0
        step_lenght = 1e-3
        while self.run:
            # get the current posture of endeffector
            y_pos_0 = self.data.body(f"hand{robot[-1]}").xpos.copy().reshape((3,1))
            y_quat_0 = self.data.body(f"hand{robot[-1]}").xquat.copy()
            # calculate the desired velocity and angular velocity
            y_pos_dot_d = (y_pos_d - y_pos_0).reshape(3,1)/(length - num_step*step_lenght)
            y_rot_delta = np.zeros((3))
            dt = length - num_step*step_lenght
            # Mujoco function is not intuitive, it calculate the q_res in the equation q_d*q_res = q_0.
            # mujoco.mju_subQuat(y_rot_delta, y_quat_0, y_quat_d)
            y_rot_delta = Kinematics.subQuat(self, y_quat_0, y_quat_d)
            y_rot_dot_d = y_rot_delta.reshape(3,1)/dt

            y_dot_d = np.vstack((y_pos_dot_d,y_rot_dot_d))

            # get current configuration of arm
            q_0 = None
            if robot == "panda1":
                q_0 = self.data.qpos.copy()[0:9].reshape(9,1)
            if robot == "panda2":
                q_0 = self.data.qpos.copy()[9:18].reshape(9,1)

            # get current jacobian matrix
            jacp = np.zeros((3, self.model.nv)) # position
            jacr = np.zeros((3, self.model.nv)) # rotation
            bodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"hand{robot[-1]}")
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, bodyid)
            jac = None
            if robot == "panda1":
                jac = np.vstack((jacp[:,:9],jacr[:,:9]))
            if robot == "panda2":
                jac = np.vstack((jacp[:,9:18],jacr[:,9:18]))

            # calculate inverse kinematics as an optimization problem
            q_d = Kinematics.inv_kin(self, y_dot_d, q_0, jac)
            
            for i in range(1, 8):
                self.data.actuator(f'actuator{i}_{robot[-1]}').ctrl = q_d[i-1]
            mujoco.mj_step(self.model, self.data) # Advance the simulation
            time.sleep(1e-3)
            num_step = num_step + 1
            # Check if the motion is completed
            if np.linalg.norm(x = y_pos_d - y_pos_0, ord = 2) < 1e-3:
                print(f"{robot} motion to {y_pos_d.flatten()} completed")
                break
            elif num_step*step_lenght >= length:
                print(f"{robot} motion cannot be completed")
                break

    def hold(self, robot, length):
        """
        Hold the robot in its current position for a specified duration.
        Args:
            robot (str): Identifier for the robot (e.g., 'panda1' or 'panda2').
            length (float): Duration to hold the position in seconds.
        """
        # Get the current configuration of the robot
        q_0 = self.data.qpos.copy()
        num_step = 0
        step_lenght = 1e-3
        print(f"holding {robot}")
        while  self.run:
            # Set the control input
            for i in range(1, 9):
                if robot == "panda1":
                    self.data.actuator(f'actuator{i}_{robot[-1]}').ctrl = q_0[i-1]
                if robot == "panda2":
                    self.data.actuator(f'actuator{i}_{robot[-1]}').ctrl = q_0[i+8]
            mujoco.mj_step(self.model, self.data) # Advance the simulation
            time.sleep(1e-3)
            num_step = num_step + 1
            # Check if the holding time is completed
            if num_step*step_lenght > length:
                print(f"{robot} holding completed")
                break
    
    def render(self,output_path='rendered_video.mp4'):
        """
        Render the simulation using GLFW and allow user interaction via mouse.
        """
        # Initialize GLFW
        glfw.init() 
        # Create the window
        glfw.window_hint(glfw.SAMPLES, 8)
        window = glfw.create_window(self.width, self.height, "Control 1", None, None)
        glfw.make_context_current(window)
        # MuJoCo rendering context
        self.context = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        opt = mujoco.MjvOption()
        pert = mujoco.MjvPerturb()
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        writer = imageio.get_writer(output_path, fps=self.fps)
        # Variables for mouse interaction
        self.last_x = 0
        self.last_y = 0
        self.is_dragging = False
        # Mouse button callback
        def mouse_button_callback(window, button, action, mods):
            if button == glfw.MOUSE_BUTTON_LEFT:
                if action == glfw.PRESS:
                    self.is_dragging = True
                    self.last_x, self.last_y = glfw.get_cursor_pos(window)
                    print('Mouse pressed at:', self.last_x, self.last_y)
                elif action == glfw.RELEASE:
                    self.is_dragging = False
                    print('Mouse released')
        # Cursor position callback
        def cursor_position_callback(window, xpos, ypos):
            if self.is_dragging:
                dx = xpos - self.last_x
                dy = ypos - self.last_y
                self.last_x = xpos
                self.last_y = ypos
                # Rotate the camera based on mouse movement
                mujoco.mjv_moveCamera(
                    self.model,
                    mujoco.mjtMouse.mjMOUSE_ROTATE_H,  # Change to mjMOUSE_ZOOM or mjMOUSE_MOVE for other controls
                    dx / self.width,
                    dy / self.height,
                    self.scene,
                    self.cam
                )
        # Scroll callback for zooming
        def scroll_callback(window, xoffset, yoffset):
            mujoco.mjv_moveCamera(
                self.model,
                mujoco.mjtMouse.mjMOUSE_ZOOM,
                0,
                -0.05 * yoffset,  # Zoom sensitivity
                self.scene,
                self.cam
            )
        # Register GLFW callbacks
        glfw.set_mouse_button_callback(window, mouse_button_callback)
        glfw.set_cursor_pos_callback(window, cursor_position_callback)
        glfw.set_scroll_callback(window, scroll_callback)
        # Main rendering loop
        while not glfw.window_should_close(window):
            w, h = glfw.get_framebuffer_size(window)
            viewport.width = w
            viewport.height = h
            # Update the scene and render
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                opt,
                pert,
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.scene,
            )
            mujoco.mjr_render(viewport, self.scene, self.context)
            # Read pixels from the framebuffer
            buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE, buffer)
            # Flip the image vertically (OpenGL uses bottom-left origin)
            buffer = np.flip(buffer, axis=0)
            # Append the frame to the video writer
            writer.append_data(buffer)


            time.sleep(1.0 / self.fps)
            glfw.swap_buffers(window)
            glfw.poll_events()
        # Cleanup
        writer.close()
        self.run = False
        glfw.terminate() 
