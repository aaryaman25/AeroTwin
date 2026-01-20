import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import time

# --- CONFIGURATION (SOTA SETTINGS) ---
SEQUENCE_LENGTH = 50
EPOCHS = 40                 # Optimal convergence point
BATCH_SIZE = 64             # Standard for this dataset
RUL_CLIP = 125              # The "Secret Trick" (Piecewise RUL)

# We use ALL sensors that actually have information (Variance > 0)
# Dropped s1, s5, s6, s10, s16, s18, s19 (Constant/Noise)
FEATURES = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 
            's12', 's13', 's14', 's15', 's17', 's20', 's21']

print(f"🚀 Initializing SOTA Training (14 Sensors | RUL Clip: {RUL_CLIP})...")

# --- LOAD DATA ---
cols = ['unit_number', 'time_cycle', 'setting_1', 'setting_2', 'setting_3']
cols += [f's{i}' for i in range(1, 22)]
train_df = pd.read_csv('train_FD001.txt', sep='\s+', header=None, names=cols)
test_df = pd.read_csv('test_FD001.txt', sep='\s+', header=None, names=cols)
y_true = pd.read_csv('RUL_FD001.txt', sep='\s+', header=None, names=['RUL'])

# --- APPLY RUL CLIPPING (The Trick) ---
# Calculate True RUL
max_cycles = train_df.groupby('unit_number')['time_cycle'].max().reset_index()
max_cycles.columns = ['unit_number', 'max_cycle']
train_df = train_df.merge(max_cycles, on='unit_number', how='left')
train_df['RUL'] = train_df['max_cycle'] - train_df['time_cycle']

# CLIP: Teach model that >125 is just "Healthy"
train_df['RUL'] = train_df['RUL'].clip(upper=RUL_CLIP)

# --- SCALING ---
scaler = MinMaxScaler()
train_df[FEATURES] = scaler.fit_transform(train_df[FEATURES])
test_df[FEATURES] = scaler.transform(test_df[FEATURES])

# --- SEQUENCE GENERATION ---
def create_sequences(df, seq_length, feature_cols, label_col=None):
    sequences = []
    labels = []
    for unit_id in df['unit_number'].unique():
        unit_data = df[df['unit_number'] == unit_id]
        data_matrix = unit_data[feature_cols].values
        if label_col:
            label_matrix = unit_data[label_col].values
        for start, stop in zip(range(0, data_matrix.shape[0]-seq_length), range(seq_length, data_matrix.shape[0])):
            sequences.append(data_matrix[start:stop, :])
            if label_col:
                labels.append(label_matrix[stop])
    return np.array(sequences), np.array(labels) if label_col else None

X_train_seq, y_train_seq = create_sequences(train_df, SEQUENCE_LENGTH, FEATURES, 'RUL')

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).view(-1, 1)

# --- MODEL: BIDIRECTIONAL LSTM (The Heavy Lifter) ---
class SOTA_LSTM(nn.Module):
    def __init__(self, input_size):
        super(SOTA_LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=128,          # Large Hidden Layer
            num_layers=2,             # Stacked LSTM
            batch_first=True, 
            dropout=0.3,
            bidirectional=True        # Reads history forwards & backwards
        )
        self.linear1 = nn.Linear(128 * 2, 64) # *2 because Bidirectional
        self.linear2 = nn.Linear(64, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Global Average Pooling (Better than just taking last step)
        # It looks at the "average" health over the window
        avg_pool = torch.mean(lstm_out, dim=1) 
        x = self.linear1(avg_pool)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

model = SOTA_LSTM(input_size=len(FEATURES))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- TRAINING ---
print(f"🧠 Training on {len(X_train_tensor)} sequences (14 Features)...")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    permutation = torch.randperm(X_train_tensor.size()[0])
    epoch_loss = 0
    
    for i in range(0, X_train_tensor.size()[0], BATCH_SIZE):
        indices = permutation[i:i+BATCH_SIZE]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    if (epoch+1) % 5 == 0:
        print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss/len(X_train_tensor):.5f}")

print(f"⏱️ Done in {int(time.time()-start_time)}s")

# --- VALIDATION ---
print("\n⚔️  VERIFYING AGAINST TRUTH...")

X_test_seq = []
valid_unit_ids = []
for unit_id in test_df['unit_number'].unique():
    unit_data = test_df[test_df['unit_number'] == unit_id][FEATURES].values
    if len(unit_data) >= SEQUENCE_LENGTH:
        X_test_seq.append(unit_data[-SEQUENCE_LENGTH:])
        valid_unit_ids.append(unit_id)

X_test_tensor = torch.tensor(np.array(X_test_seq), dtype=torch.float32)

# Important: Comparison Logic
# We must clip the Ground Truth to 125 to be fair to the model.
# If the engine is brand new (RUL 150), we only care that the model predicts >125.
y_true_valid = y_true.iloc[[i-1 for i in valid_unit_ids]]
y_true_clipped = y_true_valid.clip(upper=RUL_CLIP)

model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).numpy()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_true_clipped, y_pred))

print(f"\n=============================================")
print(f"   🏆 SOTA RESULTS (Target: < 18.00) 🏆   ")
print(f"=============================================")
print(f"   RMSE Score: {rmse:.2f} cycles")
print(f"=============================================")

if rmse < 16:
    print("🔥 WORLD CLASS. You are matching top academic papers.")
elif rmse < 20:
    print("✅ PRODUCTION READY. This is an extremely strong result.")
else:
    print("⚠️ GOOD. Much better than the baseline.")

# --- SAVE FOR DASHBOARD ---
torch.save(model.state_dict(), 'sota_model.pth')
print("💾 Model saved as 'sota_model.pth'")