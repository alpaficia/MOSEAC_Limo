#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
import numpy as np
import pandas as pd
import tensorrt as trt
import pycuda.driver as cuda
from threading import Lock
import math
import socket
from MapEnv import MapINF3995
from OptiTrackPython import NatNetClient
from Adapter import *

cuda.init()
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


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

    def __init__(
        self,
        client_ip="192.168.0.119",  # change it to your LIMO IP address
        server_ip="192.168.0.100",  # change it to your OPTITRACK PC IP address
        multicast_address="239.255.42.99",  # change it to your multicast address
        rigidbody_names2track=["limo0"],  # change it to your limo name on the OPTITRACK system
    ):
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
        self.streamingClient = NatNetClient(client_ip, server_ip, multicast_address)

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

    def receive_rigid_body_frame(
        self, timestamp, id, position, rotation, rigidBodyDescriptor
    ):
        if rigidBodyDescriptor:
            for rbname in self.rigidbody_names2track:
                if rbname in rigidBodyDescriptor:
                    if id == rigidBodyDescriptor[rbname][0]:
                        # skips this message if still locked
                        if self.lock_opti.acquire(False):
                            try:
                                # rotation is a quaternion!
                                self.optitrack_reading[rbname] = [
                                    timestamp,
                                    position,
                                    rotation,
                                ]
                            finally:
                                self.lock_opti.release()


def distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def adapt_output(output):
    output = output.squeeze()
    adapted_output = np.zeros(output.shape, dtype=np.float32)
    adapted_output[0] = np.interp(output[0], [0, 6], [0.02, 0.5])  # Adapt to seconds
    adapted_output[1] = np.interp(output[1], [-1, 1], [-1.0, 1.0])
    adapted_output[2] = np.interp(output[2], [-1, 1], [-1.0, 1.0])
    return adapted_output


def is_point_in_polygon(point, polygon):
    """
    检测点是否在多边形内部。

    参数:
    - point: 待检测的点，格式为(x, y)。
    - polygon: 多边形的顶点列表，格式为[(x1, y1), (x2, y2), ...]。

    返回:
    - 如果点在多边形内部，返回True；否则返回False。
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


class ActionPublisher:
    def __init__(self):
        rospy.init_node("action_publisher", anonymous=True)
        self.pub_cmd_vel = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.limo_position = np.array([-0.2, -0.5])  # 初始化不能在0，0，那里是禁区
        self.control_duration = np.array([0.0])  
        self.cmd_old = np.array([0.0, 0.0])
        self.linear_speed = np.array([0.0])
        self.angular_speed = np.array([0.0])
        self.world_size = 1.5
        self.map = MapINF3995()
        self.state_recorder = []
        self.action_recorder = []
        # self.goal = self.map.generate_point_outside_region()  # modify this to make a fixed goal position
        self.goal = np.array([1.2, -1.2])
        self.model_path = (
            "CHANGE THE PATH TO YOUR MODEL"
        )
        self.device = cuda.Device(0)
        self.context_cuda = self.device.make_context()
        self.engine = load_engine(self.model_path)
        self.context = self.engine.create_execution_context()
        self.judge_dis = 0.2
        self.min_time = 0.02
        self.max_time = 1.0
        self.max_action_s = 1.0
        self.out_walls = [
            tuple(point) for segment in self.map.out_walls for point in segment
        ]

    def publish_cmd_vel(self, linear_velocity, angular_velocity):
        twist = Twist()
        twist.linear.x = linear_velocity
        twist.angular.z = angular_velocity
        self.pub_cmd_vel.publish(twist)

    def predict_function(self, input_data):
        # Ensure the input data is a numpy array with the correct shape and type (float32).
        input_data = np.asarray(input_data, dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0)
        # assert input_data.shape == (1, 121), "Input data shape mismatch"

        # Allocate CUDA memory for inputs and outputs.
        self.context_cuda.push()
        d_input = cuda.mem_alloc(int(input_data.nbytes))

        # Assuming the output shape is (1, 3) and the data type is float32
        output_shape = (1, 3)
        d_output = cuda.mem_alloc(
            int(np.prod(output_shape) * input_data.dtype.itemsize)
        )

        bindings = [int(d_input), int(d_output)]

        # Create a stream to manage CUDA operations.
        stream = cuda.Stream()

        # Transfer input data to the device.
        cuda.memcpy_htod_async(d_input, input_data, stream)

        # Execute the model.

        self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # Create a buffer to hold the output data and transfer it from device to host.
        output_data = np.empty(
            output_shape, dtype=np.float32
        )  # Assuming output data type is float32
        cuda.memcpy_dtoh_async(output_data, d_output, stream)

        # Wait for all operations to finish.
        stream.synchronize()
        self.context_cuda.pop()

        return output_data

    def dynamic_publish_loop(self):
        practitioner = Optitrack()
        t = 0
        while not rospy.is_shutdown():
            start_time = rospy.Time.now()
            practitioner.read_new_data()
            self.limo_position = practitioner.limo_position
            if np.isnan(self.limo_position).any():
                rospy.loginfo("Optitrack bad, waiting for correction...")
                continue
            if t == 0:
                sleep_duration = 1.0
                linear_velocity = 0.0
                angular_velocity = 0.0
                self.publish_cmd_vel(linear_velocity, angular_velocity)
                rospy.loginfo("init the wheels position for the first launch...")
                rospy.sleep(sleep_duration)
                t += 1
            else:
                radar = self.map.radar_intersections(self.limo_position)
                radar = radar.reshape(-1)
                limo_position_uni = self.limo_position / self.world_size
                radar_no_shape_uni = radar / self.world_size
                goal_uni = self.goal / self.world_size
                control_duration_uniform = (self.control_duration - self.min_time) / (self.max_time - self.min_time)
                state = np.concatenate(
                    [
                        limo_position_uni,
                        goal_uni,
                        self.linear_speed,
                        self.angular_speed,
                        control_duration_uniform,
                        self.cmd_old,
                        radar_no_shape_uni,
                    ],
                    axis=0,
                )
                self.state_recorder.append(state)
                output_data = self.predict_function(state)
                a_t = output_data[0][0]
                a_s = output_data[0][1:3]
                act_s = Action_adapter(a_s, self.max_action_s)
                act_t = Action_t_relu6_adapter(a_t, self.min_time, self.max_time)
                act_t = np.array([act_t])
                act = np.concatenate([act_t, act_s], axis=0)
                self.action_recorder.append(act)
                dis2goal = distance(self.limo_position, self.goal)
                tuple_pos = tuple(self.limo_position.tolist())
                in_area = is_point_in_polygon(tuple_pos, self.out_walls)
                if dis2goal >= self.judge_dis and in_area:
                    self.control_duration = np.array([float(act[0])])
                    linear_velocity = float(act[1])
                    angular_velocity = float(act[2])
                    self.publish_cmd_vel(linear_velocity, angular_velocity)
                    rospy.loginfo("Sending velocity command...")
                    rospy.loginfo("Published control duration: %s", act[0])
                else:
                    linear_velocity = 0.0
                    angular_velocity = 0.0
                    self.publish_cmd_vel(linear_velocity, angular_velocity)
                    rospy.loginfo("I have arrived the goal or I'm out of area, stopped")
                    self.control_duration = np.array([0.5])
                self.cmd_old = np.array([linear_velocity, angular_velocity])
                self.linear_speed = np.array([linear_velocity])
                self.angular_speed = np.array([angular_velocity])
                elapsed_time = rospy.Time.now() - start_time
                sleep_duration = float(self.control_duration - elapsed_time.to_sec())
                if sleep_duration > 0.0:
                    rospy.sleep(sleep_duration)
        df_s = pd.DataFrame(self.state_recorder)
        df_a = pd.DataFrame(self.action_recorder)
        df_s.to_csv('CHANGE PATH TO YOUR DATASET', index=False)
        df_a.to_csv('CHANGE PATH TO YOUR DATASET', index=False)


if __name__ == "__main__":
    try:
        action_publisher = ActionPublisher()
        action_publisher.dynamic_publish_loop()
    except rospy.ROSInterruptException:
        pass
