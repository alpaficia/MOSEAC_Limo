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

def create_sequences(X, y, sequence_length=10):
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length + 1):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length - 1])
    return torch.stack(X_seq), torch.stack(y_seq)

X, y = load_csv_to_tensors("CHANGE IT TO YOUR DATASET")
X, y = create_sequences(X, y, sequence_length=10)

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


def split_tensor(X):
    """
    Split the input tensor X with shape (batch_size, 12) into four parts:
    - forces: The first 8 numbers reshaped into (batch_size, 4, 2).
    - m_p: The next 2 numbers reshaped into (batch_size, 2).
    - mu_0: The 11th number, reshaped into (batch_size, 1).
    - mu_1: The 12th (last) number, reshaped into (batch_size, 1).

    Parameters:
    - X: A tensor of shape (batch_size, 12) containing all the input data.

    Returns:
    - forces, m_p, mu_0, mu_1: The split and reshaped parts of the input tensor.
    """
    # Ensure the second dimension of X is 12
    if X.shape[1] != 12:
        raise ValueError("The second dimension of input tensor X must be 12")

    forces = X[:, :8].view(-1, 4, 2)
    m_p = X[:, 8:10].view(-1, 2)
    mu_0 = X[:, 10].unsqueeze(-1)  # Reshape to (batch_size, 1) for consistency
    mu_1 = X[:, 11].unsqueeze(-1)  # Reshape to (batch_size, 1) for consistency

    return forces, m_p, mu_0, mu_1


# 定义四轮车辆运动模型函数
def limo_update_vehicle_state_with_friction_4wheels(X, forces, mu_0, mu_1, mp_positions):
    x = X[:, 0].unsqueeze(1)
    y = X[:, 1].unsqueeze(1)
    v = X[:, 2].unsqueeze(1)
    theta = X[:, 3].unsqueeze(1)
    t = X[:, 4].unsqueeze(1)
    omega = X[:, 6].unsqueeze(1)

    mass = 4.2  # in Kg

    # 计算轮子位置
    wheel_offsets = torch.tensor([
        [0.175 / 2, 0.204 / 2],
        [-0.175 / 2, 0.204 / 2],
        [0.175 / 2, -0.204 / 2],
        [-0.175 / 2, -0.204 / 2],
    ], dtype=torch.float32)

    mp_positions_expanded = mp_positions.unsqueeze(1)
    wheel_positions = mp_positions_expanded + wheel_offsets

    total_force_x = torch.sum(forces[:, :, 0], dim=1, keepdim=True)
    total_force_y = torch.sum(forces[:, :, 1], dim=1, keepdim=True)

    # 使用 torch.where 来选择 mu
    mu = torch.where(torch.abs(v) > 0.0, mu_1, mu_0)

    friction_force = mu * mass * 9.81
    effective_force_x = total_force_x - friction_force
    effective_force_y = total_force_y - friction_force

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    total_force_global_x = effective_force_x * cos_theta - effective_force_y * sin_theta
    total_force_global_y = effective_force_x * sin_theta + effective_force_y * cos_theta

    ax = total_force_global_x / mass
    ay = total_force_global_y / mass

    v_new = v + (ax * t)
    x_new = x + (v * t) + (0.5 * ax * t ** 2)
    y_new = y + (v * t) + (0.5 * ay * t ** 2)

    torque_per_wheel = wheel_positions[:, :, 0] * forces[:, :, 1] - wheel_positions[:, :, 1] * forces[:, :, 0]
    # 然后对所有轮子的力矩求和，得到总力矩
    torque = torch.sum(torque_per_wheel, dim=1, keepdim=True)
    wheelbase = torch.norm(wheel_positions[:, 0, :] - wheel_positions[:, 2, :], dim=1, keepdim=True)
    alpha = torque / (mass * (wheelbase / 2) ** 2)

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
        forces, m_p, mu_0, mu_1 = split_tensor(bag)
        output = limo_update_vehicle_state_with_friction_4wheels(data, forces, mu_0, mu_1, m_p)
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
            forces, m_p, mu_0, mu_1 = split_tensor(bag)
            output = limo_update_vehicle_state_with_friction_4wheels(data, forces, mu_0, mu_1, m_p)
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
    model = FourWheelsTransformer(input_dim=7)

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
            forces, m_p, mu_0, mu_1 = split_tensor(bag)
            output = limo_update_vehicle_state_with_friction_4wheels(data, forces, mu_0, mu_1, m_p)

            # 计算损失
            loss = criterion(output, target)
            total_loss += loss.item()

    # 计算平均损失
    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss:.4f}')


class FourWheelsTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.1):
        super(FourWheelsTransformer, self).__init__()
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


model = FourWheelsTransformer(input_dim=7)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 早期停止参数
early_stopping_patience = 10
early_stopping_counter = 0
best_loss = float('inf')

# 训练和验证循环
epochs = 500
for epoch in range(1, epochs + 1):
    train_loss = train(model, train_loader, criterion, optimizer)
    valid_loss = validate(model, valid_loader, criterion)
    print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

    # 检查是否需要保存模型（当获得更佳的验证损失时）
    if valid_loss < best_loss:
        best_loss = valid_loss
        early_stopping_counter = 0
        # 指定模型保存路径
        save_path = 'CHANGE IT TO YOUR PATH OF YOUR MODEL FOLDER'
        model_name = 'fw_transformer_best.pth'
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

model_path = 'CHANG IT TO YOUR PATH OF YOUR MODEL'

# 调用函数，评估模型
evaluate_model(test_loader, model_path)
print("The training has been done")
