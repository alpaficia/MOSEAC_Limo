#!/usr/bin/env python3
import csv
import os
import rospy
from geometry_msgs.msg import Twist
import numpy as np
import math
import socket
from OptiTrackPython import NatNetClient
from threading import Lock
from pylimo import limo
from pynput import keyboard


def euler_from_quaternion(d):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = d[0]
    y = d[1]
    z = d[2]
    w = d[3]

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


class Optitrack(object):

    def __init__(self,
                 client_ip="192.168.0.119",  # change it to your LIMO IP address
                 server_ip="192.168.0.100",  # change it to your OPTITRACK PC IP address
                 multicast_address="239.255.42.99",  # change it to your multicast address
                 rigidbody_names2track=["limo0"]):  # change it to your LIMO name in OPTITRACK system
        if client_ip is None:
            hostname = socket.gethostname()
            client_ip = socket.gethostbyname(hostname)
        print(f"client ip: {client_ip}")
        print(f"server ip: {server_ip}")
        print(f"multicast address: {multicast_address}")
        print(f"body names to track: {rigidbody_names2track}")

        self.rigidbody_names2track = rigidbody_names2track
        self.lock_opti = Lock()
        self.optitrack_reading = {}
        # This will create a new NatNet client
        self.streamingClient = NatNetClient(client_ip,
                                            server_ip,
                                            multicast_address)

        # Configure the streaming client to call our rigid body handler on the emulator to send data out.
        self.streamingClient.rigidBodyListener = self.receive_rigid_body_frame

        # Start up the streaming client now that the callbacks are set up.
        # This will run perpetually, and operate on a separate thread.
        self.streamingClient.run()
        self.read_new_data()
        self.limo_position = np.full((2,), np.nan)

    def close(self):
        self.streamingClient.close()

    def read_new_data(self):
        for key in self.optitrack_reading:
            state_data = self.optitrack_reading[key][1]
            for ii in range(2):
                self.limo_position[ii] = np.array(state_data[ii], dtype=float)

    def receive_rigid_body_frame(self, timestamp, id, position, rotation, rigidBodyDescriptor):
        if rigidBodyDescriptor:
            for rbname in self.rigidbody_names2track:
                if rbname in rigidBodyDescriptor:
                    if id == rigidBodyDescriptor[rbname][0]:
                        # skips this message if still locked
                        if self.lock_opti.acquire(False):
                            try:
                                # rotation is a quaternion!
                                self.optitrack_reading[rbname] = [timestamp, position, rotation]
                            finally:
                                self.lock_opti.release()


class ActionRecorder:
    def __init__(self):
        rospy.init_node('action_recorder', anonymous=True)
        self.line_speed = 0.0
        self.yaw_speed = 0.0
        self.data_count = 0  # 初始化数据计数器
        self.max_data_count = 50000  # 设置目标数据量
        self.is_recording = False  # 用于控制是否记录数据到CSV文件
        self.last_control_time = rospy.Time.now()
        self.action = np.array([])
        self.optitracker = Optitrack()
        self.optitracker.read_new_data()
        self.limo_position = self.optitracker.limo_position
        self.limo = limo.LIMO()
        self.csv_file_name = '/home/agilex/Dong_SEAC/limo_pos.csv'
        self.init_csv_file()
        rospy.Subscriber("cmd_vel", Twist, self.cmd_vel_callback)
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()  # 开始监听键盘事件

    def init_csv_file(self):
        """初始化CSV文件，添加标题行（如果文件不存在）。"""
        if not os.path.exists(self.csv_file_name):
            with open(self.csv_file_name, mode='w', newline='') as file:
                writer = csv.writer(file)
                # 假设dataset_data_X和dataset_data_Y的维度和标题已知
                writer.writerow(
                    ['PosX', 'PosY', 'LinearSpeed', 'AngularSpeed', 'ControlDuration', 'LineSpeed', 'YawSpeed',
                     'NextPosX', 'NextPosY'])
                self.data_count = 0

    def append_to_csv(self, data_x, data_y):
        """将数据追加到CSV文件中。"""
        with open(self.csv_file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data_x.tolist() + data_y.tolist())
            self.data_count += 1

    def on_press(self, key):
        try:
            if key.char == 's':  # 按下"S"键
                self.is_recording = True
                self.last_control_time = rospy.Time.now()
                rospy.loginfo("Start recording the CSV file.")
            elif key.char == 't':  # 按下"T"键
                self.is_recording = False
                rospy.loginfo("Stop recording the CSV file.")
        except AttributeError:
            pass

    def cmd_vel_callback(self, msg):
        if msg.data:
            rospy.loginfo("I received the cmd data")
            self.line_speed = msg.linear.x
            self.yaw_speed = msg.angular.z
            current_time = rospy.Time.now()
            control_duration = (current_time - self.last_control_time).to_sec()
            self.last_control_time = current_time

            self.optitracker = Optitrack()
            self.optitracker.read_new_data()
            linear_speed = np.array(self.limo.GetLinearVelocity())
            angular_speed = np.array(self.limo.GetAngularVelocity())
            dataset_data_X = np.array([self.limo_position[0], self.limo_position[1], linear_speed, angular_speed,
                                       control_duration, self.line_speed, self.yaw_speed])
            rospy.loginfo("data_X record: ", dataset_data_X)
            limo_position_new = self.optitracker.limo_position
            dataset_data_Y = limo_position_new
            rospy.loginfo("data_Y record: ", dataset_data_Y)
            if self.is_recording:
                self.append_to_csv(dataset_data_X, dataset_data_Y)
            self.limo_position = limo_position_new
            if self.data_count >= self.max_data_count:
                rospy.loginfo("I got enough data")


if __name__ == '__main__':
    try:
        action_recorder = ActionRecorder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
