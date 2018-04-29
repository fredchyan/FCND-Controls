"""
PID Controller

components:
    follow attitude commands
    gps commands and yaw
    waypoint following
"""
import numpy as np
from math import sin, cos, tan, sqrt
from frame_utils import euler2RM

DRONE_MASS_KG = 0.5
GRAVITY = -9.81
MOI = np.array([0.005, 0.005, 0.01])
MAX_THRUST = 10.0
MAX_TORQUE = 1.0

class NonlinearController(object):

    def __init__(self):
        """Initialize the controller object and control gains"""
        self.k_p_p = 20
        self.k_p_q = 20
        self.k_p_r = 5

        self.k_p_z = 4
        self.k_d_z = 1.5

        self.k_p_yaw = 4.5

        self.k_p_x = 6
        self.k_d_x = 4
        self.k_p_y = 6
        self.k_d_y = 4

        self.k_p_roll = 8
        self.k_p_pitch = 8

        return    

    def trajectory_control(self, position_trajectory, yaw_trajectory, time_trajectory, current_time):
        """Generate a commanded position, velocity and yaw based on the trajectory
        
        Args:
            position_trajectory: list of 3-element numpy arrays, NED positions
            yaw_trajectory: list yaw commands in radians
            time_trajectory: list of times (in seconds) that correspond to the position and yaw commands
            current_time: float corresponding to the current time in seconds
            
        Returns: tuple (commanded position, commanded velocity, commanded yaw)
                
        """

        ind_min = np.argmin(np.abs(np.array(time_trajectory) - current_time))
        time_ref = time_trajectory[ind_min]
        
        
        if current_time < time_ref:
            position0 = position_trajectory[ind_min - 1]
            position1 = position_trajectory[ind_min]
            
            time0 = time_trajectory[ind_min - 1]
            time1 = time_trajectory[ind_min]
            yaw_cmd = yaw_trajectory[ind_min - 1]
            
        else:
            yaw_cmd = yaw_trajectory[ind_min]
            if ind_min >= len(position_trajectory) - 1:
                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min]
                
                time0 = 0.0
                time1 = 1.0
            else:

                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min + 1]
                time0 = time_trajectory[ind_min]
                time1 = time_trajectory[ind_min + 1]
            
        position_cmd = (position1 - position0) * \
                        (current_time - time0) / (time1 - time0) + position0
        velocity_cmd = (position1 - position0) / (time1 - time0)
        
        
        return (position_cmd, velocity_cmd, yaw_cmd)
    
    def lateral_position_control(self, local_position_cmd, local_velocity_cmd, local_position, local_velocity,
                               acceleration_ff = np.array([0.0, 0.0])):
        """Generate horizontal acceleration commands for the vehicle in the local frame

        Args:
            local_position_cmd: desired 2D position in local frame [north, east]
            local_velocity_cmd: desired 2D velocity in local frame [north_velocity, east_velocity]
            local_position: vehicle position in the local frame [north, east]
            local_velocity: vehicle velocity in the local frame [north_velocity, east_velocity]
            acceleration_cmd: feedforward acceleration command
            
        Returns: desired vehicle 2D acceleration in the local frame [north, east]
        """
        x_actual, y_actual = local_position
        x_target, y_target = local_position_cmd
        x_dot_actual, y_dot_actual = local_velocity
        x_dot_target, y_dot_target = local_velocity_cmd
        x_dot_dot_target, y_dot_dot_target = acceleration_ff

        x_err = x_target - x_actual
        y_err = y_target - y_actual

        x_err_dot = x_dot_target - x_dot_actual
        y_err_dot = y_dot_target - y_dot_actual

        p_term_x = self.k_p_x * x_err
        p_term_y = self.k_p_y * y_err

        d_term_x = self.k_d_x * x_err_dot
        d_term_y = self.k_d_y * y_err_dot

        x_dot_dot_command = p_term_x + d_term_x + x_dot_dot_target
        y_dot_dot_command = p_term_y + d_term_y + y_dot_dot_target

        return np.array([x_dot_dot_command, y_dot_dot_command])
    
    def altitude_control(self, altitude_cmd, vertical_velocity_cmd, altitude, vertical_velocity, attitude, acceleration_ff=0.0):
        """Generate vertical acceleration (thrust) command

        Args:
            altitude_cmd: desired vertical position (+up)
            vertical_velocity_cmd: desired vertical velocity (+up)
            altitude: vehicle vertical position (+up)
            vertical_velocity: vehicle vertical velocity (+up)
            attitude: the vehicle's current attitude, 3 element numpy array (roll, pitch, yaw) in radians
            acceleration_ff: feedforward acceleration command (+up)
            
        Returns: thrust command for the vehicle (+up)
        """
        altitude_err = altitude_cmd - altitude

        altitude_dot_cmd = np.clip(self.k_p_z * (altitude_err) + vertical_velocity_cmd, -2.0, 5.0)

        
        u_1_bar = self.k_d_z * (altitude_dot_cmd -  vertical_velocity) + acceleration_ff

        rot_mat = euler2RM(attitude[0],attitude[1],attitude[2])
        b_z = rot_mat[2,2]
        c = (u_1_bar)/b_z
        thrust = np.clip(DRONE_MASS_KG * c, 0.0, MAX_THRUST)

        return thrust
        
    
    def roll_pitch_controller(self, acceleration_cmd, attitude, thrust_cmd):
        """ Generate the rollrate and pitchrate commands in the body frame
        
        Args:
            target_acceleration: 2-element numpy array (north_acceleration_cmd,east_acceleration_cmd) in m/s^2
            attitude: 3-element numpy array (roll, pitch, yaw) in radians
            thrust_cmd: vehicle thruts command in Newton
            
        Returns: 2-element numpy array, desired rollrate (p) and pitchrate (q) commands in radians/s
        """
        if not (thrust_cmd > 0.0):
            return np.array([0.0, 0.0])

        x_dot_dot_command, y_dot_dot_command = acceleration_cmd

        # thrust_cmd is positive up
        c = - thrust_cmd / DRONE_MASS_KG
        b_x_c = np.clip(x_dot_dot_command/c, -1.0, 1.0)
        b_y_c = np.clip(y_dot_dot_command/c, -1.0, 1.0)

        rot_mat = euler2RM(attitude[0],attitude[1],attitude[2])

        b_x = rot_mat[0,2]
        b_x_err = b_x_c - b_x
        b_x_p_term = self.k_p_roll * b_x_err

        b_y = rot_mat[1,2]
        b_y_err = b_y_c - b_y
        b_y_p_term = self.k_p_pitch * b_y_err

        b_x_commanded_dot = b_x_p_term
        b_y_commanded_dot = b_y_p_term

        rot_mat1=np.array([[rot_mat[1,0],-rot_mat[0,0]],[rot_mat[1,1],-rot_mat[0,1]]])/rot_mat[2,2]

        rot_rate = np.matmul(rot_mat1,np.array([b_x_commanded_dot,b_y_commanded_dot]).T)

        p_c = rot_rate[0]
        q_c = rot_rate[1]


        return np.array([p_c, q_c])
    
    def body_rate_control(self, body_rate_cmd, body_rate):
        """ Generate the roll, pitch, yaw moment commands in the body frame
        
        Args:
            body_rate_cmd: 3-element numpy array (p_cmd,q_cmd,r_cmd) in radians/second^2
            body_rate: 3-element numpy array (p,q,r) in radians/second^2
            
        Returns: 3-element numpy array, desired roll moment, pitch moment, and yaw moment commands in Newtons*meters
        """
        body_rate_err = body_rate_cmd - body_rate
        proportional_gain = np.array([self.k_p_p, self.k_p_q, self.k_p_r])
        u_body_rate_bar = proportional_gain * body_rate_err
        moment_bar = MOI * u_body_rate_bar
        moment_norm = np.linalg.norm(moment_bar)
        if moment_norm > MAX_TORQUE:
            unit_vector = moment_bar/moment_norm
            moment_bar = unit_vector * MAX_TORQUE
        return moment_bar
    
    def yaw_control(self, yaw_cmd, yaw):
        """ Generate the target yawrate
        
        Args:
            yaw_cmd: desired vehicle yaw in radians
            yaw: vehicle yaw in radians
        
        Returns: target yawrate in radians/sec
        """
        psi_cmd = np.mod(yaw_cmd, np.pi*2.0)
        psi_err = psi_cmd - yaw
        if psi_err > np.pi:
            psi_err = psi_err - 2.0*np.pi
        elif psi_err < -np.pi:
            psi_err = psi_err + 2.0*np.pi
        r_c = self.k_p_yaw * psi_err
        return r_c

