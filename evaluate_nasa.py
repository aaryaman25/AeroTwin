import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import math

# --- CONFIG ---
FEATURES = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 
            's12', 's13', 's14', 's15', 's17', 's20', 's21']
SEQUENCE_LENGTH = 50
RUL_CLIP = 125

# --- DEFINE MODEL (Must match training) ---
class SOTA_LSTM(nn.Module):
    def __init__(self, input_size):
        super(SOTA_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.linear1 = nn.Linear(128 * 2, 64)
        self.linear2 = nn.Linear(64, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        avg_pool = torch.mean(lstm_out, dim=1) 
        x = self.linear1(avg_pool)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

# --- SCORING FUNCTION (The NASA Formula) ---
def compute_nasa_score(y_true, y_pred):
    score = 0
    for true, pred in zip(y_true, y_pred):
        diff = pred - true
        if diff < 0:
            # Early Prediction (Safe-ish) -> Penalty grows slower
            score += math.exp(-diff / 13) - 1
        else:
            # Late Prediction (Dangerous!) -> Penalty grows FAST
            score += math.exp(diff / 10) - 1
    return score

# --- MAIN EVALUATION ---
print("⏳ Loading Data & Model...")
cols = ['unit_number', 'time_cycle', 'setting_1', 'setting_2', 'setting_3']
cols += [f's{i}' for i in range(1, 22)]
test_df = pd.read_csv('test_FD001.txt', sep='\s+', header=None, names=cols)
y_true_df = pd.read_csv('RUL_FD001.txt', sep='\s+', header=None, names=['RUL'])

# Load Assets
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

model = SOTA_LSTM(input_size=len(FEATURES))
model.load_state_dict(torch.load('sota_model.pth', map_location=torch.device('cpu')))
model.eval()

# Prepare Data
print("⚙️ Processing Test Engines...")
preds = []
trues = []

unit_ids = test_df['unit_number'].unique()

for unit_id in unit_ids:
    # 1. Get Engine Data
    unit_data = test_df[test_df['unit_number'] == unit_id][FEATURES].values
    
    # 2. Check if enough data
    if len(unit_data) >= SEQUENCE_LENGTH:
        # Take last 50 cycles
        seq = unit_data[-SEQUENCE_LENGTH:]
        
        # Scale
        seq_scaled = scaler.transform(seq)
        
        # Predict
        with torch.no_grad():
            tensor_in = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)
            pred = model(tensor_in).item()
            
        preds.append(pred)
        
        # Get True RUL (Clipped to 125 for fairness)
        true_rul = y_true_df.iloc[unit_id-1]['RUL']
        true_clipped = min(true_rul, RUL_CLIP)
        trues.append(true_clipped)

# --- RESULTS ---
rmse = np.sqrt(np.mean((np.array(preds) - np.array(trues))**2))
nasa_score = compute_nasa_score(trues, preds)

print(f"\n=============================================")
print(f"   📊 FINAL ENGINEERING REPORT")
print(f"=============================================")
print(f"   RMSE (Data Science Metric): {rmse:.2f}")
print(f"   NASA Score (Safety Metric): {nasa_score:.2f}")
print(f"=============================================")

if nasa_score < 500:
    print("🏆 RESULT: Top Tier (Winner Circle)")
elif nasa_score < 1500:
    print("✅ RESULT: Excellent (Industry Standard)")
else:
    print("⚠️ RESULT: Good, but penalizes late predictions heavily.")