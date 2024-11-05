import gymnasium as gym
from gymnasium import spaces
from MapEnv import MapINF3995
import pygame
import numpy as np
import torch
import torch.nn as nn
import math


def coordinate_system_conversion(coordinate, window_size):
    """
    将给定坐标沿着x轴旋转180度进行转换。
    coordinate: 输入坐标，假定基于中心为原点的坐标系。
    window_size: 窗口的尺寸 (width, height)，用于计算转换后的坐标。

    返回:
    转换后的坐标，适合 Pygame 等图形库的渲染。
    """
    origin_coordinate = np.array([window_size[0] / 2, window_size[1] / 2])
    scale = min(window_size[0], window_size[1]) / 1.5

    # 保持x坐标的计算不变
    target_coordinate_x = origin_coordinate[0] + scale * coordinate[0]

    target_coordinate_y = origin_coordinate[1] + scale * coordinate[1]

    # 确保坐标在窗口范围内
    target_coordinate_x = np.clip(target_coordinate_x, 0, window_size[0])
    target_coordinate_y = np.clip(target_coordinate_y, 0, window_size[1])

    return np.array([target_coordinate_x, target_coordinate_y])


def distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def select_random_goal():
    # 定义8个点的数组
    points = np.array([
        [1.2, 1.2],
        [1.2, -1.2],
        [-1.2, 1.2],
        [-1.2, -1.2],
        [1.2, 0],
        [0, 1.2],
        [-1.2, 0],
        [0, -1.2]
    ])
    random_index = np.random.randint(len(points))

    # 通过索引选择并返回一个点
    return points[random_index]


def limo_model_with_friction_and_power(X, mu_k, power_factor, g=9.81):
    """
    考虑摩擦力、发动机功率和车辆质量的limo模型，power_factor可能为负数。

    Parameters:
    - X: torch tensor, 形状为 (N, 7)，包含 N 个数据点，每个数据点有以下7个特征。
    - mu_k: 动摩擦系数。
    - power_factor: 发动机功率因子，可能为负数。
    - g: 重力加速度，默认值为9.81 m/s^2。

    Returns:
    - torch tensor, 预测位置（x, y）。
    """
    M = 4.2  # limo的重量，单位kg
    L = 0.204  # limo前后轮的中心位置距离，单位m

    # 提取特征
    current_x = X[:, 0].unsqueeze(1)
    current_y = X[:, 1].unsqueeze(1)
    v = X[:, 2].unsqueeze(1)
    theta = X[:, 3].unsqueeze(1)
    dt = X[:, 4].unsqueeze(1)
    target_v = X[:, 5].unsqueeze(1)
    delta = X[:, 6].unsqueeze(1)

    # 计算摩擦力导致的减速
    F_friction = -mu_k * M * g
    a_friction = F_friction / M

    # 考虑发动机功率因子，它可能为负数，表示发动机提供反向力
    a_power = power_factor / M

    # 估算加速度，考虑摩擦力和发动机功率
    a_net = a_power + (target_v - v) / dt + a_friction

    # 更新速度
    v_new = v + a_net * dt

    # 考虑速度可能变为负数，表示车辆反向移动
    # 如果需要限制车辆不后退，可以在这里设置 v_new = torch.clamp(v_new, min=0)

    # 更新位置和朝向
    theta_new = theta + (v_new / L) * torch.tan(delta) * dt
    x_new = current_x + v_new * torch.cos(theta_new) * dt
    y_new = current_y + v_new * torch.sin(theta_new) * dt

    return torch.cat((x_new, y_new), dim=1)


def sigmoid(x, k, x0):
    return 1 / (1 + math.exp(-k * (x - x0)))


class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.1):
        super(SimpleTransformer, self).__init__()
        self.input_fc = nn.Linear(input_dim, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.transformer_block = nn.TransformerEncoderLayer(d_model=128, nhead=8, dropout=dropout_rate,
                                                            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_block, num_layers=3)
        self.output_fc = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # 取最后一个时间步的输出作为结果
        x = self.output_fc(x)
        x[:, 0] = torch.relu(x[:, 0])  # 确保摩擦系数非负
        return x


class LimoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode='rgb_array', size=2.0):
        self.adaptive_elastic_parameters = False
        self.map = MapINF3995()
        self.world_wide = 1.5
        self.success_reward = 500.0
        self.cross_punish = -30.0
        self.dead_punish = -100.0
        self._wall_location = self.map.get_wall()
        self._wall_location_uniform = self._wall_location.reshape(-1)
        self._target_location = None  # target location info, in (x, y)
        self.original_dis = None
        self._agent_location = None  # agent location info, in (x, y)
        self._start_location = None
        self.radar = None
        self.agent_linear_speed = np.zeros(1,)  # in m/s
        self.agent_yaw_speed = np.zeros(1,)
        self.judgment_distance = 0.2  # in meters
        self.time_step_duration = np.ones(1,)
        self.control = np.zeros(2,)
        self.reset_mark = False  # if the env get reset, set this mark to True to clean some history data
        self.size = size  # The size of the pygame square world
        self.window_size = 512  # The size of the PyGame window
        self.max_linear_speed = 1.0  # maximum linear speed of the agent
        self.max_angular_speed = 1.0  # maximum angular speed of the agent
        self.alpha_m_max = 3.0  # max alpha_m that alpha_m can reach
        self.alpha_m = 1.0  # the gain factor for the reward parts on finished task
        self.alpha_eps = 0.1  # the gain factor for the reward parts on time cost per step
        self.t_min = 0.02
        self.t_max = 0.5

        self.model_path = 'CHANGE THE PATH TO YOUR ENVIRONMENT MODEL'
        self.model = SimpleTransformer(input_dim=7, output_dim=2)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(-self.world_wide, self.world_wide, shape=(2,), dtype=np.float64),
                "target": spaces.Box(-self.world_wide, self.world_wide, shape=(2,), dtype=np.float64),
                "linear_speed": spaces.Box(-1.0 * self.max_linear_speed, self.max_linear_speed, shape=(1,),
                                           dtype=np.float64),
                "yaw": spaces.Box(-1.0 * self.max_angular_speed, self.max_angular_speed, shape=(1,), dtype=np.float64),
                "time": spaces.Box(self.t_min, self.t_max, shape=(1,), dtype=np.float64),
                "control": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float64),
                "radar": spaces.Box(-self.world_wide, self.world_wide, shape=(40,), dtype=np.float64),
            }
        )
        # state_dim: 49
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct frame rate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent_pos": self._agent_location, "target": self._target_location,
                "linear_speed": self.agent_linear_speed, "yaw": self.agent_yaw_speed,
                "time": self.time_step_duration, "control": self.control,
                "radar": self.radar
                }

    def _get_info(self):
        return {
            "distance_2_goal": distance(self._agent_location, self._target_location)
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.agent_linear_speed = np.zeros(1,)  # reset to 0.0
        self.agent_yaw_speed = np.zeros(1,)
        self.time_step_duration = np.ones(1,)  # reset the time of one time step
        self.control = np.zeros(2,)  # reset the records of true movement
        self._agent_location = np.array([-0.2 + np.random.uniform(-0.05, 0.05), -0.5 + np.random.uniform(-0.05, 0.05)])
        self.radar = self.map.radar_intersections(self._agent_location)
        self.radar = self.radar.reshape(-1)
        self._target_location = select_random_goal()
        # self._target_location = np.array([1.2, 0.0])
        # points = np.array([
        #     [1.2, 1.2],
        #     [1.2, -1.2],
        #     [-1.2, 1.2],
        #     [-1.2, -1.2],
        #     [1.2, 0],
        #     [0, 1.2],
        #     [-1.2, 0],
        #     [0, -1.2]
        # ])
        # change the goal to a fixed value while validating the model.
        self.original_dis = float(distance(self._agent_location, self._target_location))
        ###########################################################################################################
        observation = self._get_obs()
        info = self._get_info()
        self.reset_mark = True
        if self.render_mode == "human":
            self._render_frame()
        return observation, info

    def pad_sequence(self, input_data, sequence_length=10):
        # 当数据不足10步时，用零填充
        if input_data.shape[0] < sequence_length:
            padding = torch.zeros(sequence_length - input_data.shape[0], input_data.shape[1])
            input_data = torch.cat((padding, input_data), dim=0)
        return input_data.unsqueeze(0)  # 添加 batch 维度

    def step(self, action):

        action_time = np.array([action[0]])  # the time for current action
        action_control = np.array(action[1:3])  # the control cmd
        input2model = np.concatenate([self._agent_location, self.agent_linear_speed, self.agent_yaw_speed,
                                      action_time, action_control], axis=0)
        input2model_tensor = torch.tensor(input2model)
        input2model_tensor = self.pad_sequence(input2model_tensor, sequence_length=10)
        with torch.no_grad():
            bag = self.model(input2model_tensor.float())
            mu_k = bag[:, 0:1]
            power_factor = bag[:, 1:2]
            agent_pos_new = limo_model_with_friction_and_power(input2model_tensor, mu_k, power_factor, g=9.81)
        agent_pos_new = agent_pos_new.squeeze(0)
        agent_pos_new = np.array(agent_pos_new)
        #  predict the pos with the env model
        if self.map.is_collision_with_inner_wall(self._agent_location, agent_pos_new):
            reward_task = self.cross_punish
            terminated = False

        elif distance(agent_pos_new, self._target_location) <= self.judgment_distance:
            reward_task = self.success_reward
            terminated = True
        elif self.map.is_collision_with_out_wall(self._agent_location, agent_pos_new):
            reward_task = self.dead_punish
            terminated = True
        else:
            reward_task = self.original_dis - distance(agent_pos_new, self._target_location)
            terminated = False

        reward_tau = self.remapping(action_time)
        reward = self.alpha_m * reward_task * reward_tau - self.alpha_eps

        self._agent_location = agent_pos_new
        self.agent_linear_speed = np.array([action_control[0]])
        self.agent_yaw_speed = np.array([action_control[1]])
        self.time_step_duration = action_time
        self.control = action_control
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return observation, float(reward), terminated, False, info

    def update_gain_r(self, increase_amount):
        # 更新 self.gain
        self.alpha_m += increase_amount
        if self.alpha_m >= self.alpha_m_max:
            self.alpha_m = self.alpha_m_max
        # 使用逆向 Sigmoid 函数更新 self.gain_eps
        # 选择k和x0的值以确保在gain为1时gain_eps为0.1
        k = 1  # 控制sigmoid函数的斜率
        x0 = 1  # 控制sigmoid中心为1，使得gain为1时输出特定值
        # 更新self.gain_eps，将sigmoid输出乘以10，并减去1使得gain为1时gain_eps为0.1
        self.alpha_eps = 0.2 * (1 - sigmoid(self.alpha_m, k, x0))

    def remapping(self, time):
        alpha_tau = self.t_min / time
        return alpha_tau

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # 白色背景
        pix_square_size = self.window_size / self.size

        # 转换坐标的辅助函数
        def convert_coords(coords):
            # 坐标系范围从-1.5到1.5
            scale_x = self.window_size / (self.world_wide * 2)
            scale_y = self.window_size / (self.world_wide * 2)

            # x坐标转换：
            x = (coords[0] + self.world_wide) * scale_x

            # y坐标转换：需要翻转并根据新的范围调整
            # Pygame的原点在左上角，所以我们需要翻转y坐标
            y = self.window_size - (coords[1] + self.world_wide) * scale_y

            return np.array([x, y]).astype(int)

        # 绘制所有墙体
        all_walls = self.map.get_wall()
        for wall in all_walls:
            # 假设每堵墙的 wall 是一个由起点和终点坐标组成的 numpy 数组
            start_pos, end_pos = convert_coords(wall[0]), convert_coords(wall[1])
            pygame.draw.line(canvas, (0, 0, 255), start_pos, end_pos, 5)  # 蓝色墙体

        # 绘制代理位置
        agent_screen_pos = convert_coords(self._agent_location)
        pygame.draw.circle(canvas, (0, 255, 0), agent_screen_pos, int(pix_square_size / 15))  # 绿色代理

        # 绘制终点位置
        goal_screen_pos = convert_coords(self._target_location)
        pygame.draw.circle(canvas, (255, 0, 0), goal_screen_pos, int(pix_square_size / 15))  # 红色终点

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        # 可选：如果需要生成rgb_array模式的输出
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
