"""
Usage:
(robodiff)$ python demo_xarm.py --robot_ip <ip_of_xarm> --max_speed 100

Robot movement:
Move your SpaceMouse to move the robot EEF (3 spatial DoF only).
Press SpaceMouse left button once to reset to initial pose.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import sys
import numpy as np
import scipy.spatial.transform as st
from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)

# xArm API
sys.path.append('/home/USER/xArm-Python-SDK')
from xarm.wrapper import XArmAPI

@click.command()
@click.option('--robot_ip', '-ri', default="192.168.1.217", required=False, help="UR5's IP address e.g. 192.168.0.204")
@click.option('--init_joints', '-j', is_flag=False, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('--max_speed', '-ms', default=10, type=float, help="Max speed of the robot in mm/s.")
def main(robot_ip, init_joints, frequency, command_latency, max_speed):
    max_speed = max_speed * frequency
    dt = 1/frequency
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            Spacemouse(shm_manager=shm_manager) as sm:

            arm = XArmAPI(robot_ip, is_radian=True)
            cv2.setNumThreads(1)

            # initialize xarm
            arm.motion_enable(enable=True)
            arm.set_mode(1)
            arm.set_state(state=0)
            time.sleep(1)
            if init_joints:
                arm.reset(wait=True)

            arm.set_mode(1) # for set_servo_cartesian
            arm.set_state(0)

            
            time.sleep(1.0)
            print('Ready!')


            state = arm.get_position()
            target_pose = state[1]

            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
            while not stop:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # handle key presses
                press_events = key_counter.get_press_events()
                stage = key_counter[Key.space]

                precise_wait(t_sample)
                # get teleop command
                sm_state = sm.get_motion_state_transformed()
                # print(sm_state)
                dpos = sm_state[:3] * (max_speed / frequency)
                #set z_axis movement = 0
                dpos[2] = 0.0
                drot_xyz = sm_state[3:] * (max_speed / frequency)
                
                drot_xyz[:] = 0

                drot = st.Rotation.from_euler('xyz', drot_xyz)
                target_pose[:3] += dpos
                target_pose[3:] = (drot * st.Rotation.from_rotvec(
                    target_pose[3:])).as_rotvec()

                speed = np.linalg.norm(dpos) / dt
                
                # if sm.is_button_pressed(0):
                #     arm.set_mode(0)
                #     arm.set_state(0)
                #     time.sleep(0.1)
                #     # arm.reset(wait=True)
                #     arm.set_mode(5)
                #     arm.set_state(0)
                
                dvel = dpos

                if speed > 0.1:
                    dvel = dpos * (max_speed / speed)
                else:
                    dvel *= 0
                    # continue

                x_velocity, y_velocity, z_velocity = dvel
                # roll_velocity, pitch_velocity, yaw_velocity = drot_xyz
                # print(f'x_velocity: {x_velocity}, y_velocity: {y_velocity}, z_velocity: {z_velocity}')
                # print(f'roll_velocity: {roll_velocity}, pitch_velocity: {pitch_velocity}, yaw_velocity: {yaw_velocity}')
                print(f'Target_pose: {target_pose[:]}')


                # arm.vc_set_cartesian_velocity([x_velocity, y_velocity, z_velocity, roll_velocity, pitch_velocity, yaw_velocity])
                arm.set_servo_cartesian(mvpose=target_pose)
                precise_wait(t_cycle_end)
                iter_idx += 1

# %%
if __name__ == '__main__':
    main()