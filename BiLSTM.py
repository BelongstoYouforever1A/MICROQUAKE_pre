import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

# ========== 基础路径配置 ==========
DATA_DIR = r"E:\\shijian\\retrain"
SAVE_PATH_1 = r"E:\\shijian\\modelpart1.pt"
SAVE_PATH_2 = r"E:\\shijian\\modelpart2.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 单段型模型超参数 ==========
INPUT_DIM = 5
CNN_CHANNELS1 = [32, 64]
HIDDEN_SIZE1 = 128
LINEAR_DIM1 = 32
EPOCHS1 = 50
BATCH_SIZE1 = 16
LEARNING_RATE1 = 5e-3
POS_WEIGHT1 = 3
PATIENCE1 = 5

# ========== 双段型模型超参数 ==========
CNN_CHANNELS2 = [16, 32]
HIDDEN_SIZE2 = 96
LINEAR_DIM2 = 64
EPOCHS2 = 50
BATCH_SIZE2 = 16
LEARNING_RATE2 = 2e-3
POS_WEIGHT2 = 2.5
PATIENCE2 = 6

# ========== 标签分类函数 ==========
def classify_type(label_array):
    label_array = np.array(label_array).astype(np.float32)
    label_array = (label_array > 0.5).astype(np.int32)
    changes = np.diff(np.pad(label_array, (1, 1), constant_values=0))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    num_segments = len(starts)
    return 1 if num_segments <= 1 else 2

# ========== 数据集加载类 ==========
class SeismicDataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        df = pd.read_csv(self.files[idx])
        X = df.iloc[:, :-1].values
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
        y = df.iloc[:, -1].values
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ========== 单段型模型 CNN+LSTM ==========
class CNNLSTMNet(nn.Module):
    def __init__(self, input_dim=INPUT_DIM):
        super(CNNLSTMNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, CNN_CHANNELS1[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(CNN_CHANNELS1[0], CNN_CHANNELS1[1], kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=CNN_CHANNELS1[1], hidden_size=HIDDEN_SIZE1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_SIZE1, LINEAR_DIM1),
            nn.ReLU(),
            nn.Linear(LINEAR_DIM1, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        out = self.classifier(lstm_out)
        return out.squeeze(-1)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

# ========== 双段型模型 BiLSTM + Attention ==========
class BiLSTM_Attention_Net(nn.Module):
    def __init__(self, input_dim=INPUT_DIM):
        super(BiLSTM_Attention_Net, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, CNN_CHANNELS2[0], kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(CNN_CHANNELS2[0], CNN_CHANNELS2[1], kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=CNN_CHANNELS2[1],
                            hidden_size=HIDDEN_SIZE2,
                            batch_first=True,
                            bidirectional=True)

        self.attn_linear = nn.Linear(HIDDEN_SIZE2 * 2, HIDDEN_SIZE2 * 2)
        self.dropout = nn.Dropout(0.3)  # 新增 Dropout

        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_SIZE2 * 2, LINEAR_DIM2),
            nn.ReLU(),
            nn.Linear(LINEAR_DIM2, 1)
        )

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)

        # 改进 attention 权重（更平滑）
        attn_score = torch.sigmoid(self.attn_linear(lstm_out)) * 0.8 + 0.1
        x = lstm_out * attn_score

        x = self.dropout(x)  # 防止过拟合
        out = self.classifier(x)
        return out.squeeze(-1)


# ========== 通用训练函数 ==========
def train_model(model, dataloader, save_path, lr, pos_weight_val, patience, epochs):
    model = model.to(DEVICE)
    pos_weight = torch.tensor([pos_weight_val]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in dataloader:
            X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[{save_path}] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {lr}")

        # Early stopping
        if best_loss - avg_loss > 1e-4:
            best_loss = avg_loss
            counter = 0
            best_model_state = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                print(f"早停触发（{patience}轮无明显改善），训练结束。")
                break

    if best_model_state:
        torch.save(best_model_state, save_path)
        print(f"最佳模型已保存至 {save_path}")
    else:
        torch.save(model.state_dict(), save_path)
        print(f"无明显提升，保存最后一轮模型至 {save_path}")

# ========== 主流程 ==========
def main():
    files = sorted(glob(os.path.join(DATA_DIR, "*_features.csv")))
    print(f"共找到训练样本：{len(files)} 个")

    type1_files = []
    type2_files = []

    print("🚥 正在分类样本...")
    for file in tqdm(files):
        df = pd.read_csv(file)
        labels = df.iloc[:, -1].values
        typ = classify_type(labels)
        if typ == 1:
            type1_files.append(file)
        else:
            type2_files.append(file)

    print(f"单段型样本: {len(type1_files)}，双段型样本: {len(type2_files)}")

    if len(type1_files) == 0 or len(type2_files) == 0:
        print("某一类样本数量为0，无法训练。请检查数据。")
        return

    loader1 = DataLoader(SeismicDataset(type1_files), batch_size=BATCH_SIZE1, shuffle=True, pin_memory=True)
    loader2 = DataLoader(SeismicDataset(type2_files), batch_size=BATCH_SIZE2, shuffle=True, pin_memory=True)

    model1 = CNNLSTMNet(input_dim=INPUT_DIM)
    model2 = BiLSTM_Attention_Net(input_dim=INPUT_DIM)

    print("模型1参数量: %.2fM" % (model1.get_num_params() / 1e6))
    print("模型2参数量: %.2fM" % (model2.get_num_params() / 1e6))

    print("开始训练模型1（单段型）...")
    train_model(model1, loader1, SAVE_PATH_1, lr=LEARNING_RATE1, pos_weight_val=POS_WEIGHT1, patience=PATIENCE1, epochs=EPOCHS1)

    print("开始训练模型2（双段型）...")
    train_model(model2, loader2, SAVE_PATH_2, lr=LEARNING_RATE2, pos_weight_val=POS_WEIGHT2, patience=PATIENCE2, epochs=EPOCHS2)

if __name__ == '__main__':
    main()