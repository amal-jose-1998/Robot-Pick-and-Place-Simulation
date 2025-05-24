import numpy as np
import mujoco
import time

class Kinematics():
    """
    This class provides fundamental kinematics calculations, including inverse kinematics 
    and quaternion operations.
    """
    def __init__(self):
        self.W = np.eye(9)*1e1
        self.C = np.diag(np.array([1e1,1e1,1e1,1e1,1e1,1e1]))*1e1

    def for_kin(self):
        pass

    def inv_kin(self, y_dot_d, q_0, jac):
        """        
        Solve inverse kinematic problem using the following optimization problem:
        (min ||y_dot_d - j * q_dot_d)||_W + ||q_dot_d||_C)
        :param y_dot_d (numpy.ndarray): The desired end-effector velocity.
        :param q_0 (numpy.ndarray): The initial joint angles.
        :param jac (numpy.ndarray): The Jacobian matrix.
        :return numpy.ndarray: The desired joint velocities.
        """
        q_d = q_0 + np.linalg.inv(jac.T@self.C@jac+self.W)@jac.T@self.C@y_dot_d
        return q_d

    def subQuat(self, y_quat_0, y_quat_d):
        """
        Calculate the delta rotation between two quaternions.
        (y_quat_d = y_quat_delta * y_quat_0)
        :param y_quat_0 (numpy.ndarray): The initial quaternion, represented as a 4-element array.
        :param y_quat_d (numpy.ndarray): The target quaternion, represented as a 4-element array.
        :return numpy.ndarray: The rotation vector representing the delta rotation between the two quaternions.
        """
        # conjugate of y_quat_0
        y_quat_0_inv = y_quat_0.copy()
        for i in range(1,4):
            y_quat_0_inv[i] = -y_quat_0_inv[i]
        # calculate the delta rotation
        y_quat_delta = np.zeros((4))
        mujoco.mju_mulQuat(y_quat_delta, y_quat_d, y_quat_0_inv)
        # convert the delta rotation to ratation vector
        y_rot_delta = y_quat_delta[1:4] / np.linalg.norm(x = y_quat_delta[1:4], ord = 2)
        delta = 2 * np.arccos(y_quat_delta[0])
        y_rot_delta = y_rot_delta * delta
        return y_rot_delta

class CollisionDetection(Kinematics):
    """
    A class for collision detection in robotic environments using kinematics.
    It provides methods to check for collisions and control a simulation model.
    """
    def __init__(self):
        """
        Initializes the CollisionDetection class by setting thresholds, 
        loading a simulation model for background collision checking, and 
        initializing simulation data.
        """
        Kinematics.__init__(self)
        self.collision_threshold = 0.30  # Define a safe minimum distance
        self.num_points = 5 # No of points between the joints to trace the robot as a line structure
        # load another simulation model without rendering to check the collision creiteria in the background of the main simulation
        self.sim_model = mujoco.MjModel.from_xml_path("/home/amal/Desktop/ISW_HiWi/cable routing demo/2024-cable-routing-demo/simulation_env/world(box).xml")
        self.sim_data = mujoco.MjData(self.sim_model)
        self.run_sim = True
        # Initialize the simulation to a default state
        for key in range(self.sim_model.nkey):
            mujoco.mj_resetDataKeyframe(self.sim_model, self.sim_data, key)
            mujoco.mj_forward(self.sim_model, self.sim_data)

    def is_collision_free(self, new_pos, cur_pos, robot):
        """
        Checks if the motion to a new position is collision-free for the specified robot.
        Args:
            new_pos (np.array): The target position for the robot's motion.
            cur_pos (np.array): The current position of the robot.
            robot (str): The robot identifier (e.g., 'panda1' or 'panda2').
        Returns:
            bool: True if the motion is collision-free, False otherwise.
        """
        y_quat_d = np.array([0,0,1,0]) # Desired orientation
        response = self.move_sim_rob(robot, np.array([[new_pos[0]], [new_pos[1]], [new_pos[-1]]]), y_quat_d, 1)
        if response:
            robot1_points = self.get_robot_points("panda1")
            robot2_points = self.get_robot_points("panda2")
            # Check distances between points of both robots
            for point1 in robot1_points:
                for point2 in robot2_points:
                    if np.linalg.norm(point1 - point2) < self.collision_threshold:
                        return False  # Collision detected
            return True  # No collision detected
        else:
            # Reset to current position if motion fails
            response = self.move_sim_rob(robot, np.array([[cur_pos[0]], [cur_pos[1]], [cur_pos[-1]]]), y_quat_d, 1)
            return False
        
    def move_sim_rob(self, robot, y_pos_d, y_quat_d, length):
        """
        Simulates robot motion to a desired position and orientation in the background simulation.
        Args:
            robot (str): The robot identifier (e.g., 'panda1' or 'panda2').
            y_pos_d (np.array): Desired position.
            y_quat_d (np.array): Desired orientation (quaternion).
            length (float): Duration of the motion.
        Returns:
            bool: True if the motion was successfully completed, False otherwise.
        """
        num_step = 0
        step_lenght = 1e-3
        while self.run_sim:
            # Current state of the robot in the simulation
            y_pos_0 = self.sim_data.body(f"hand{robot[-1]}").xpos.copy().reshape((3,1))
            y_quat_0 = self.sim_data.body(f"hand{robot[-1]}").xquat.copy()
            # Compute target velocities
            y_pos_dot_d = (y_pos_d - y_pos_0).reshape(3,1)/(length - num_step*step_lenght)
            y_rot_delta = np.zeros((3))
            dt = length - num_step*step_lenght
            y_rot_delta = Kinematics.subQuat(self, y_quat_0, y_quat_d)
            y_rot_dot_d = y_rot_delta.reshape(3,1)/dt
            y_dot_d = np.vstack((y_pos_dot_d,y_rot_dot_d))
            q_0 = None
            # Get the current joint positions
            if robot == "panda1":
                q_0 = self.sim_data.qpos.copy()[0:9].reshape(9,1)
            if robot == "panda2":
                q_0 = self.sim_data.qpos.copy()[9:18].reshape(9,1)
            # Compute the Jacobian
            jacp = np.zeros((3, self.sim_model.nv)) # position
            jacr = np.zeros((3, self.sim_model.nv)) # rotation
            bodyid = mujoco.mj_name2id(self.sim_model, mujoco.mjtObj.mjOBJ_BODY, f"hand{robot[-1]}")
            mujoco.mj_jacBody(self.sim_model, self.sim_data, jacp, jacr, bodyid)
            jac = None
            if robot == "panda1":
                jac = np.vstack((jacp[:,:9],jacr[:,:9]))
            if robot == "panda2":
                jac = np.vstack((jacp[:,9:18],jacr[:,9:18]))
            # Perform inverse kinematics
            q_d = Kinematics.inv_kin(self, y_dot_d, q_0, jac)
            # Apply control inputs
            for i in range(1, 8):
                self.sim_data.actuator(f'actuator{i}_{robot[-1]}').ctrl = q_d[i-1]
            mujoco.mj_step(self.sim_model, self.sim_data) # Advance the simulation
            time.sleep(1e-3)
            num_step = num_step + 1
            # Check if the motion is completed
            if np.linalg.norm(x = y_pos_d - y_pos_0, ord = 2) < 1e-3:
                return True
            elif num_step*step_lenght >= length:
                return False
    
    def get_robot_points(self, robot_name):
        """
        Retrieves interpolated points along the robot's structure for collision checking.
        Args:
            robot_name (str): Name of the robot (e.g., 'panda1').
        Returns:
            list: A list of 3D points representing the robot's structure.
        """
        points = []
        site_positions = []
        # Get site positions for the robot
        for site_name in [f"panda_site{i}_{robot_name[-1]}" for i in range(0, 9)]:
            site_id = mujoco.mj_name2id(self.sim_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            site_position = self.sim_data.site(site_id).xpos
            site_positions.append(site_position)
        # Iterate through consecutive joints to interpolate points
        for i in range(len(site_positions) - 1):
            start = site_positions[i]
            end = site_positions[i + 1]
            if i == 0:
                points.append(start)
            # Interpolate points along the link
            for t in np.linspace(0, 1, self.num_points + 2)[1:-1]:
                interpolated_point = start * (1 - t) + end * t
                points.append(interpolated_point)    
            points.append(end)
        return points
    
class RRT(CollisionDetection):
    """
    Implements Rapidly-Exploring Random Tree (RRT) for path planning in a robotic environment.
    """
    def __init__(self):
        """
        Initializes the RRT class with environment bounds, maximum iterations, and step size.
        """
        CollisionDetection.__init__(self)
        self.bounds=[(-0.4, 0.4), (-0.2, 0.2), (0.25, 0.8)] # Workspace bounds
        self.max_iter=1000 # Maximum iterations for RRT
        self.step_size=0.25 # Step size for tree expansion
    
    def path_plan(self, xpos, goal, robot):
        """
        Plans a path from the current position to the goal using RRT.
        Args:
            xpos (np.array): Starting position.
            goal (np.array): Goal position.
            robot (str): The robot identifier (e.g., 'panda1').
        Returns:
            list: The planned path as a sequence of positions, or None if no path is found.
        """
        tree = [xpos]
        parent = {0: None} # maps each node's index to its parent node index.
        path_completion = False
        new_node = xpos
        path = []
        for _ in range(self.max_iter):
            if not path_completion: # compute the new node only if the path is not fully completed till the goal
                rand_point = np.array([np.random.uniform(low, high) for low, high in self.bounds]) # Random point in workspace
                # nearest node in the tree to the random point
                nearest_idx = np.argmin([np.linalg.norm(node - rand_point) for node in tree]) 
                nearest_node = tree[nearest_idx]
                # Expand towards the random point
                direction = rand_point - nearest_node
                direction = (direction / np.linalg.norm(direction)) * self.step_size
                new_node = nearest_node + direction            
            # Collision check
            if not self.is_collision_free(new_node, nearest_node, robot):
                continue  # If collision, do not add new nodes to the tree
            tree.append(new_node) # append the new node to the tree
            parent[len(tree) - 1] = nearest_idx# assign the parent
            # Check if goal is reached
            if np.linalg.norm(new_node - goal) < self.step_size:
                path_completion = True
                tree.append(goal)
                parent[len(tree) - 1] = len(tree) - 2
                current = len(tree) - 1 
                while current is not None:
                    path.append(tree[current])
                    current = parent[current]
                path.reverse()
                print("Iterations taken: ",_+1)
                return path # returns the path from start to the goal for both robots
        return None  # If path not found
    

if __name__ == "__main__":
    controller = RRT()
    start = np.array([0.2       , 0.15449948, 0.62450243])
    goal = np.array([0.3 , 0.  , 0.15])
    path = controller.path_plan(start, goal, 'panda1')
    print(path)