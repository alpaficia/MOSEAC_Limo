import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import pandas as pd


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


# 加载CSV文件到张量的函数
def load_csv_to_tensors(filepath):
    data = pd.read_csv(filepath, skiprows=1)
    X = data.iloc[:, :7].values
    Y = data.iloc[:, -2:].values
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    return X_tensor, Y_tensor


# 加载数据
X, y = load_csv_to_tensors("CHANGE IT TO THE PATH TO YOUR DATASET")

# 划分数据集为训练集、验证集和测试集
total_size = len(X)
train_size = int(0.7 * total_size)
valid_size = int(0.15 * total_size)
test_size = total_size - train_size - valid_size
dataset = TensorDataset(X, y)
train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

# 创建DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# 定义模型
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.1):
        super(SimpleTransformer, self).__init__()
        self.input_fc = nn.Linear(input_dim, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.transformer_block = nn.TransformerEncoderLayer(d_model=128, nhead=8, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_block, num_layers=3)
        self.output_fc = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)
        x = self.output_fc(x)
        x[:, 0] = torch.relu(x[:, 0])  # 确保摩擦系数为非负
        return x


# 冻结模型前几层的参数
def freeze_layers(model, num_layers_to_freeze):
    for i, (name, param) in enumerate(model.named_parameters()):
        if i < num_layers_to_freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True


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


# 逐步解冻层的训练和验证循环
def fine_tune_model(model, train_loader, valid_loader, criterion, optimizer, num_layers_to_freeze):
    # 冻结前几层
    freeze_layers(model, num_layers_to_freeze)
    best_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = 10
    epochs = 20
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, criterion, optimizer)
        valid_loss = validate(model, valid_loader, criterion)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
        if valid_loss < best_loss:
            best_loss = valid_loss
            early_stopping_counter = 0
            save_path = 'CHANGE IT TO THE PATH WHERE YOU WANT TO SAVE YOUR MODEL FOLDER'
            model_name = f'simple_transformer_best_freeze_{num_layers_to_freeze}.pth'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), os.path.join(save_path, model_name))
            print(f'Model saved to {os.path.join(save_path, model_name)}')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f'Early stopping triggered at epoch {epoch}.')
                break


# 创建模型、损失函数和优化器
model = SimpleTransformer(input_dim=7, output_dim=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 逐步解冻层并进行微调
for num_layers_to_freeze in reversed(range(4)):  # 模型有4层可解冻
    fine_tune_model(model, train_loader, valid_loader, criterion, optimizer, num_layers_to_freeze)


# 评估模型
def evaluate_model(test_loader, model_path):
    model = SimpleTransformer(input_dim=7, output_dim=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            bag = model(data)
            mu_k = bag[:, 0:1]
            power_factor = bag[:, 1:2]
            output = limo_model_with_friction_and_power(data, mu_k, power_factor, g=9.81)
            loss = criterion(output, target)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss:.4f}')


# 调用评估函数
model_path = 'CHANGE IT TO YOUR MODEL PATH'
evaluate_model(test_loader, model_path)
print("The training has been done")