import numpy as np
from physics_sim import PhysicsSim

class VerticalLiftTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 50., 0.]) 
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        #penalize movements in x,y,z axes when moving away from target_pos
        #reward movements in x,y,z axes when moving towards the target_pos
        
        #For eg: penalize proportionally based on distance from target (reward if target reached).
        x_target = self.target_pos[0]
        y_target = self.target_pos[1]
        z_target = self.target_pos[2]

        x_diff = abs(x_target - self.sim.pose[0])
        y_diff = abs(y_target - self.sim.pose[1])
        z_diff = abs(z_target - self.sim.pose[2])
        
        #reward within range (0-target) & punish otherwise
        if self.sim.pose[0] > x_target:
            x_reward = -1*x_diff
        elif self.sim.pose[0] < 0:
            x_reward = -1*x_diff
        else:
            x_reward = self.sim.pose[0]

        if self.sim.pose[1] > y_target:
            y_reward = -1*y_diff
        elif self.sim.pose[1] < 0:
            y_reward = -1*y_diff
        else:
            y_reward = self.sim.pose[1]

            
        if self.sim.pose[2] > z_target:
            z_reward = -1*z_diff
        elif self.sim.pose[2] < 0:
            z_reward = -1*z_diff
        else:
            z_reward = self.sim.pose[2]
            
        #x_reward = 1 if x_diff==0 else (1-np.tanh(x_diff))
        #y_reward = 1 if y_diff==0 else (1-np.tanh(y_diff))
        #z_reward = 1 if z_diff==0 else (1-np.tanh(z_diff))
        
        #reward = x_reward + 2*(y_reward) + z_reward
        #reward = np.tanh(x_reward) + 2*np.tanh(y_reward) + np.tanh(z_reward)
        #reward = np.tanh(x_reward  + 2*y_reward + z_reward)
        #reward = np.tanh(reward)        
        #reward = np.tanh(1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum()
        
        x_reward = np.tanh(1 - 0.0033*(x_diff))
        y_reward = np.tanh(1 - 0.0033*(y_diff)) 
        z_reward = np.tanh(1 - 0.0033*(z_diff))
    
        reward = x_reward + y_reward + z_reward
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
            
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
    

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state