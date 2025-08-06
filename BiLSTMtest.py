import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import accuracy_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader

# ========== 显式超参数配置 ==========
TEST_DIR = r"/home/liuyt6522/microquake_project/retest"
MODEL_PATH_1 = r"/home/liuyt6522/microquake_project/modelpart1.pt"
MODEL_PATH_2 = r"/home/liuyt6522/microquake_project/modelpart2.pt"
RAW_DIR = r"/home/liuyt6522/microquake_project/data"
OUT_DIR = r"/home/liuyt6522/microquake_project/output"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 5
CNN_CHANNEL_1 = 32
CNN_CHANNEL_2 = 64
LSTM_HIDDEN = 128
LINEAR_HIDDEN = 32

CNN_CHANNELS2 = [16, 32]
LSTM_HIDDEN2 = 96
LINEAR_HIDDEN2 = 64

THRESHOLD = 0.5

os.makedirs(OUT_DIR, exist_ok=True)

# ========== 平滑处理函数 ==========
def smooth_prediction(pred, min_len=30, merge_gap=10):
    pred = (pred > 0.5).astype(int)
    changes = np.diff(np.pad(pred, (1, 1), constant_values=0))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    new_pred = np.zeros_like(pred)
    kept_intervals = []

    for s, e in zip(starts, ends):
        if e - s >= min_len:
            new_pred[s:e] = 1
            kept_intervals.append((s, e))

    merged = []
    for s, e in kept_intervals:
        if not merged:
            merged.append([s, e])
        else:
            prev_s, prev_e = merged[-1]
            if s - prev_e < merge_gap:
                merged[-1][1] = e
            else:
                merged.append([s, e])

    new_pred = np.zeros_like(pred)
    for s, e in merged:
        new_pred[s:e] = 1

    return new_pred

# ========== 聚类跳变平滑 ==========
def cluster_prediction(pred, gap=50):
    pred = (pred > 0.5).astype(int)
    changes = np.diff(np.pad(pred, (1, 1), constant_values=0))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    if len(starts) == 0 or len(ends) == 0:
        return pred

    new_starts, new_ends = [], []
    current_start, current_end = starts[0], ends[0]

    for s, e in zip(starts[1:], ends[1:]):
        if s - current_end <= gap:
            current_end = e
        else:
            new_starts.append(current_start)
            new_ends.append(current_end)
            current_start, current_end = s, e
    new_starts.append(current_start)
    new_ends.append(current_end)

    new_pred = np.zeros_like(pred)
    for s, e in zip(new_starts, new_ends):
        new_pred[s:e] = 1

    return new_pred

# ========== 类型分类 ==========
def classify_type(label_array):
    label_array = np.array(label_array).astype(np.float32)
    label_array = (label_array > 0.5).astype(np.int32)
    changes = np.diff(np.pad(label_array, (1, 1), constant_values=0))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    num_segments = len(starts)
    return 1 if num_segments <= 1 else 2

# ========== 数据集 ==========
class SeismicDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(glob(os.path.join(data_dir, '*_features.csv')))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        df = pd.read_csv(path)
        X = df.iloc[:, :-1].values
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
        y = df.iloc[:, -1].values
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), path

# ========== 单段模型 ==========
class CNNLSTMNet(nn.Module):
    def __init__(self, input_dim=INPUT_DIM):
        super(CNNLSTMNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, CNN_CHANNEL_1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(CNN_CHANNEL_1, CNN_CHANNEL_2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=CNN_CHANNEL_2, hidden_size=LSTM_HIDDEN, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, LINEAR_HIDDEN),
            nn.ReLU(),
            nn.Linear(LINEAR_HIDDEN, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        out = self.classifier(lstm_out)
        return out.squeeze(-1)

# ========== 双段模型 ==========
class BiLSTM_Attention_Net(nn.Module):
    def __init__(self, input_dim=INPUT_DIM):
        super(BiLSTM_Attention_Net, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, CNN_CHANNELS2[0], kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(CNN_CHANNELS2[0], CNN_CHANNELS2[1], kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=CNN_CHANNELS2[1], hidden_size=LSTM_HIDDEN2, batch_first=True, bidirectional=True)
        self.attn_linear = nn.Linear(LSTM_HIDDEN2 * 2, LSTM_HIDDEN2 * 2)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(LSTM_HIDDEN2 * 2, LINEAR_HIDDEN2),
            nn.ReLU(),
            nn.Linear(LINEAR_HIDDEN2, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        attn_score = torch.sigmoid(self.attn_linear(lstm_out)) * 0.8 + 0.1
        x = lstm_out * attn_score
        x = self.dropout(x)
        out = self.classifier(x)
        return out.squeeze(-1)

# ========== 绘图函数 ==========
def plot_prediction(true, pred, raw_wave, fname):
    plt.figure(figsize=(12, 4))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(raw_wave, color='gray', alpha=0.5, label='Raw Waveform')
    ax2.plot(true, label='True Label', color='blue', linewidth=1)
    ax2.plot(pred, label='Predicted', color='red', linestyle='--', linewidth=1)

    ax1.set_ylabel('Amplitude')
    ax2.set_ylabel('Label')
    ax1.set_xlabel('Time Index')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Raw Waveform + Prediction')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

# ========== 推理主函数 ==========
def evaluate():
    dataset = SeismicDataset(TEST_DIR)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model1 = CNNLSTMNet().to(DEVICE)
    model1.load_state_dict(torch.load(MODEL_PATH_1, map_location=DEVICE))
    model1.eval()

    model2 = BiLSTM_Attention_Net().to(DEVICE)
    model2.load_state_dict(torch.load(MODEL_PATH_2, map_location=DEVICE))
    model2.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y, path in loader:
            X = X.to(DEVICE)
            y_np = y.numpy().flatten()
            label_type = classify_type(y_np)

            model = model1 if label_type == 1 else model2
            logits = model(X)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            if label_type == 2:
                preds = smooth_prediction(probs)
            else:
                preds = (probs > THRESHOLD).astype(int)

            # 新增：聚类合并预测跳变
            preds = cluster_prediction(preds, gap=50)

            all_preds.extend(preds)
            all_labels.extend(y_np)

            # 新增：读取原始波形并绘图保存
            fname = os.path.basename(path[0]).split('_')[0]
            raw_path = os.path.join(RAW_DIR, f"{fname}.csv")
            raw_wave = pd.read_csv(raw_path, header=None).iloc[:, 0].values

#             save_path = os.path.join(OUT_DIR, f"{fname}.png")
#             plot_prediction(y_np, preds, raw_wave, save_path)

    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"\n✅ 测试完成！")
    print(f"Accuracy: {acc:.4f}")
    print(f"Recall:   {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == '__main__':
    evaluate()
