import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def build_net(layer_shape, activation, output_activation):
    # Build net with for loop
    layers = []
    for j in range(len(layer_shape) - 1):
        if j < len(layer_shape) - 2:
            act = activation
        else:
            act = output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j + 1]), act()]
    return nn.Sequential(*layers)


# 定义你的模型结构
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


# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, states, locations):
        self.states = states
        self.locations = locations

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        location = self.locations[idx]
        return torch.tensor(state, dtype=torch.float32), torch.tensor(location, dtype=torch.float32)


# 加载数据
df = pd.read_csv('CHANGE IT TO YOUR DATASET PATH')
X = df.iloc[:, :49].values
y = df.iloc[:, -4:-2].values

# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 创建数据加载器
train_dataset = MyDataset(X_train, y_train)
val_dataset = MyDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 加载预训练模型
index = "CHANGE IT TO YOUR MODEL ID"
model = SimpleTransformer(input_dim=7, output_dim=2)
model.load_state_dict(torch.load("CHANGE IT TO YOUR MODEL PATH"))

# 微调设置
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练模型
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for states, true_locs in train_loader:
        optimizer.zero_grad()
        outputs = model(states)
        loss = criterion(outputs, true_locs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

    # 验证模型
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for states, true_locs in val_loader:
            outputs = model(states)
            loss = criterion(outputs, true_locs)
            val_loss += loss.item()
    print(f'Validation Loss: {val_loss / len(val_loader)}')