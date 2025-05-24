# **Robot Pick-and-Place Simulation with RRT**

This project implements a simulation environment for controlling two Panda robots using the MuJoCo physics engine. The robots are tasked with picking up and placing objects using **Rapidly-exploring Random Trees (RRT)** for path planning while avoiding collisions.



## **📌 Features**  

✅ **Dual-Robot Coordination** – Two Panda robots execute sequential pick-and-place operations.  
✅ **RRT Path Planning** – The robots autonomously find collision-free paths.  
✅ **Collision Detection** – A secondary simulation ensures safe motion.  
✅ **Realistic MuJoCo Simulation** – The motion is physics-driven and dynamic.  
✅ **Rendered Simulation Video** – The output is saved as an `.mp4` file.  


## **🌍 Simulation Environments**

To simplify getting started with robot control, I have created a simulation environment called world(box).xml. This environment features two Panda robots and two boxes. The objective is for each robot to pick up its respective box and place it at the origin (target), one after the other.

Additionally, another simulation environment named world(cable).xml is available. This setup is designed for performing cable pick-and-place operations.


## **🛠 Utilities**

The utils module contains essential classes for kinematics, collision detection, and RRT-based path planning. You can run the code to visualize how the algorithm generates different paths between a defined start and goal position.

The panda_mujoco.py module provides a class for simulating Panda robots within the MuJoCo framework.

## **🚀 Getting Started**

### **1️⃣ Install Dependencies**  

Clone the repository and create a virtual environment:  
```bash
git clone https://github.com/your-repository-name.git
cd your-repository-name
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
### **2️⃣ Run the Simulation**  
### Basic Pick-and-Place Demo (Hardcoded path)

```bash
python demo_pick_place.py
```
    
    RRT-Based Pick-and-Place Simulation
    The generated video will be saved as rendered_video.mp4.

## **🧠 How the Algorithm Works**
1️⃣ RRT Path Planning

The RRT algorithm generates a collision-free path between the start and goal:

✔️ Random sampling in the robot's workspace.
✔️ Connecting nodes to form a search tree.
✔️ Expanding towards the goal while ensuring collision-free movement.

2️⃣ Collision Detection

A background simulation verifies:

✔️ If a new pose causes a collision.
✔️ If a new pose is reachable by the robot.
✔️ If a new pose violates workspace constraints.

🚨 If a collision is detected, the pose is rejected, and an alternative path is explored.

3️⃣ Task Execution Sequence

1️⃣ Panda1 picks up box1, places it at the target (origin), and returns home.
2️⃣ Panda2 picks up box2 and places it on top of box1 at the origin.

The entire process is automated using the RRTControl class.
