import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
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

X, y = load_csv_to_tensors("CHANGE IT TO THE PATH TO YOUR DATASET")
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


def pad_sequence(input_data, sequence_length=10):
    # 当数据不足10步时，用零填充
    if input_data.shape[0] < sequence_length:
        padding = torch.zeros(sequence_length - input_data.shape[0], input_data.shape[1])
        input_data = torch.cat((padding, input_data), dim=0)
    return input_data.unsqueeze(0)  # 添加 batch 维度

def limo_model_with_friction_and_power(X, mu_k, power_factor, g=9.81):
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

    # 更新位置和朝向
    theta_new = theta + (v_new / L) * torch.tan(delta) * dt
    x_new = current_x + v_new * torch.cos(theta_new) * dt
    y_new = current_y + v_new * torch.sin(theta_new) * dt

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

# 在部署时使用的函数
def deploy_model_with_padding(single_step_data, model, model_path):
    # 将单步数据转换为10步序列
    padded_data = pad_sequence(single_step_data, sequence_length=10)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        bag = model(padded_data)
        mu_k = bag[:, 0:1]
        power_factor = bag[:, 1:2]
        output = limo_model_with_friction_and_power(padded_data, mu_k, power_factor, g=9.81)
    return output
