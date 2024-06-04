import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import pandas as pd


def load_csv_to_tensors(filepath):
    # Read the csv file using pandas, skip the first row
    data = pd.read_csv(filepath, skiprows=1)

    # Extract the X columns (the first seven columns)
    X = data.iloc[:, :7].values

    # Extract the Y columns (the last two columns)
    Y = data.iloc[:, -2:].values

    # Convert the numpy arrays to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    return X_tensor, Y_tensor

X, y = load_csv_to_tensors("CHANGE IT TO THE PATH TO YOUR DATASET")

# 划分数据集为训练集、验证集和测试集
total_size = len(X)
train_size = int(0.7 * total_size)
valid_size = int(0.15 * total_size)
test_size = total_size - train_size - valid_size
dataset = TensorDataset(X, y)
train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

# DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


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


# 定义四轮车辆运动模型函数
def update_vehicle_state_with_friction_4wheels(X, forces, mu_0, mu_1, mp_positions):
    x = X[:, 0].unsqueeze(1)
    y = X[:, 1].unsqueeze(1)
    v = X[:, 2].unsqueeze(1)
    theta = X[:, 3].unsqueeze(1)
    t = X[:, 4].unsqueeze(1)
    omega = X[:, 6].unsqueeze(1)

    mass = 4.2  # in Kg
    forces = torch.tensor(forces, dtype=torch.float32)
    m_x, m_y = mp_positions
    wheel_positions = torch.tensor([
        [m_x + 0.175 / 2, m_y + 0.204 / 2],  # 第一个轮子相对于质心的位置, in meters
        [m_x - 0.175 / 2, m_y + 0.204 / 2],  # 第二个轮子相对于质心的位置, in meters
        [m_x + 0.175 / 2, m_y - 0.204 / 2],  # 第三个轮子相对于质心的位置, in meters
        [m_x - 0.175 / 2, m_y - 0.204 / 2]  # 第四个轮子相对于质心的位置, in meters
    ], dtype=torch.float32)

    # 计算合成力
    total_force_x = torch.sum(forces[:, 0])
    total_force_y = torch.sum(forces[:, 1])

    # 根据车辆当前状态选择摩擦系数
    mu = mu_1 if torch.abs(v) > 0.0 else mu_0

    # 考虑摩擦力后的合成力
    friction_force = mu * mass * 9.81  # 假设所有力都是水平作用的
    effective_force_x = total_force_x - friction_force
    effective_force_y = total_force_y - friction_force

    # 转换为全局坐标系下的合成力
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    total_force_global_x = effective_force_x * cos_theta - effective_force_y * sin_theta
    total_force_global_y = effective_force_x * sin_theta + effective_force_y * cos_theta

    # 计算加速度
    ax = total_force_global_x / mass
    ay = total_force_global_y / mass

    # 更新线速度和位置
    v_new = v + (ax * t)
    x_new = x + (v * t) + (0.5 * ax * t ** 2)
    y_new = y + (v * t) + (0.5 * ay * t ** 2)

    # 计算力矩和角加速度
    torque = torch.sum(torch.cross(wheel_positions, forces, dim=1)[:, 2])
    wheelbase = torch.norm(wheel_positions[0, :] - wheel_positions[2, :])
    alpha = torque / (mass * (wheelbase / 2) ** 2)

    # 更新角速度和朝向
    omega_new = omega + (alpha * t)
    theta_new = theta + (omega * t) + (0.5 * alpha * t ** 2)

    return torch.cat((x_new, y_new), dim=1)


# 训练函数
def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        bag = model(data)
        mu_k = bag[:, 0:1]
        power_factor = bag[:, 1:2]
        output = limo_model_with_friction_and_power(data, mu_k, power_factor, g=9.81)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss


# 验证函数
def validate(model, valid_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in valid_loader:
            bag = model(data)
            mu_k = bag[:, 0:1]
            power_factor = bag[:, 1:2]
            output = limo_model_with_friction_and_power(data, mu_k, power_factor, g=9.81)
            loss = criterion(output, target)
            total_loss += loss.item()
    avg_loss = total_loss / len(valid_loader)
    return avg_loss


def evaluate_model(test_loader, model_path):
    """
    加载训练好的模型并使用测试集评估其性能。

    Parameters:
    - test_loader: DataLoader，为测试集数据。
    - model_path: str，保存的模型状态字典的路径。
    """
    # 确保模型架构与保存时相同
    model = SimpleTransformer(input_dim=7, output_dim=2)

    # 加载模型状态
    model.load_state_dict(torch.load(model_path))

    # 设置为评估模式
    model.eval()

    # 准备评估指标
    criterion = nn.MSELoss()
    total_loss = 0

    # 不计算梯度，以节省计算资源
    with torch.no_grad():
        for data, target in test_loader:
            # 预测
            bag = model(data)
            mu_k = bag[:, 0:1]
            power_factor = bag[:, 1:2]
            output = limo_model_with_friction_and_power(data, mu_k, power_factor, g=9.81)

            # 计算损失
            loss = criterion(output, target)
            total_loss += loss.item()

    # 计算平均损失
    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss:.4f}')


class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.1):
        super(SimpleTransformer, self).__init__()
        self.input_fc = nn.Linear(input_dim, 128)
        self.dropout = nn.Dropout(dropout_rate)  # 添加Dropout层
        self.transformer_block = nn.TransformerEncoderLayer(d_model=128, nhead=8, dropout=dropout_rate,
                                                            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_block, num_layers=3)
        self.output_fc = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)
        x = self.output_fc(x)
        x[:, 0] = torch.relu(x[:, 0])  # 摩擦系数应该非负，使用ReLU，确保非负
        return x

class FourWheelsTransformer(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.1):
        super(FourWheelsTransformer, self).__init__()
        self.input_fc = nn.Linear(input_dim, 128)
        self.dropout = nn.Dropout(dropout_rate)  # 添加Dropout层
        self.transformer_block = nn.TransformerEncoderLayer(d_model=128, nhead=8, dropout=dropout_rate,
                                                            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_block, num_layers=3)
        self.output_fc = nn.Linear(128, 12)  # 输出维度为12

    def forward(self, x):
        x = self.input_fc(x)
        x = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)
        x = self.output_fc(x)
        x[:, -2:] = torch.relu(x[:, -2:])  # 对最后两个维度使用ReLU激活函数确保非负
        return x

model = SimpleTransformer(input_dim=7, output_dim=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 早期停止参数
early_stopping_patience = 10
early_stopping_counter = 0
best_loss = float('inf')

# 训练和验证循环
epochs = 200
for epoch in range(1, epochs + 1):
    train_loss = train(model, train_loader, criterion, optimizer)
    valid_loss = validate(model, valid_loader, criterion)
    print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

    # 检查是否需要保存模型（当获得更佳的验证损失时）
    if valid_loss < best_loss:
        best_loss = valid_loss
        early_stopping_counter = 0
        # 指定模型保存路径
        save_path = 'CHANGE IT TO THE PATH WHERE YOU WANT TO SAVE YOUR MODEL FOLDER'
        model_name = 'simple_transformer_best.pth'
        # 确保路径存在
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 保存模型
        torch.save(model.state_dict(), os.path.join(save_path, model_name))
        print(f'Model saved to {os.path.join(save_path, model_name)}')
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print(f'Early stopping triggered at epoch {epoch}.')
            break

model_path = 'CHANGE IT TO YOUR MODEL PATH'

# 调用函数，评估模型
evaluate_model(test_loader, model_path)
print("The training has been done")
