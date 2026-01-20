import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# --- CONFIG ---
SEQUENCE_LENGTH = 50
FEATURES = ['s2', 's14', 's21']
EPOCHS = 30
BATCH_SIZE = 200

print("⏳ Loading Data...")
# Load Data
cols = ['unit_number', 'time_cycle', 'setting_1', 'setting_2', 'setting_3']
cols += [f's{i}' for i in range(1, 22)]
train_df = pd.read_csv('train_FD001.txt', sep='\s+', header=None, names=cols)
test_df = pd.read_csv('test_FD001.txt', sep='\s+', header=None, names=cols)
y_true = pd.read_csv('RUL_FD001.txt', sep='\s+', header=None, names=['RUL'])

# --- PREPROCESSING ---
print("⚙️ Scaling Data...")
scaler = MinMaxScaler()
train_df[FEATURES] = scaler.fit_transform(train_df[FEATURES])
test_df[FEATURES] = scaler.transform(test_df[FEATURES])

# Calculate RUL for Training
max_cycles = train_df.groupby('unit_number')['time_cycle'].max().reset_index()
max_cycles.columns = ['unit_number', 'max_cycle']
train_df = train_df.merge(max_cycles, on='unit_number', how='left')
train_df['RUL'] = train_df['max_cycle'] - train_df['time_cycle']

# --- HELPER: CREATE SEQUENCES ---
def create_sequences(df, seq_length, feature_cols, label_col=None):
    sequences = []
    labels = []
    
    for unit_id in df['unit_number'].unique():
        # Get data for this engine
        unit_data = df[df['unit_number'] == unit_id]
        data_matrix = unit_data[feature_cols].values
        
        # If we have labels (Training), get them
        if label_col:
            label_matrix = unit_data[label_col].values
            
        # Create sliding windows
        num_elements = data_matrix.shape[0]
        for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
            sequences.append(data_matrix[start:stop, :])
            if label_col:
                labels.append(label_matrix[stop])
                
    return np.array(sequences), np.array(labels) if label_col else None

print("🎞️ Creating Sequences (This might take a moment)...")
X_train_seq, y_train_seq = create_sequences(train_df, SEQUENCE_LENGTH, FEATURES, 'RUL')

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).view(-1, 1)

# --- MODEL 1: RANDOM FOREST ---
print("\n🌲 Training Random Forest (Baseline)...")
X_train_rf = train_df[FEATURES]
y_train_rf = train_df['RUL']
rf_model = RandomForestRegressor(n_estimators=100, max_depth=12)
rf_model.fit(X_train_rf, y_train_rf)

# --- MODEL 2: PYTORCH LSTM ---
print("🧠 Training PyTorch LSTM (Deep Learning)...")

class AirLSTM(nn.Module):
    def __init__(self, input_size):
        super(AirLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=60, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear1 = nn.Linear(60, 30)
        self.linear2 = nn.Linear(30, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take the output of the last time step
        last_step = lstm_out[:, -1, :] 
        x = self.dropout(last_step)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

model = AirLSTM(input_size=len(FEATURES))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    # Mini-batch training
    permutation = torch.randperm(X_train_tensor.size()[0])
    
    for i in range(0, X_train_tensor.size()[0], BATCH_SIZE):
        indices = permutation[i:i+BATCH_SIZE]
        batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
    print(f"   Epoch {epoch+1}/{EPOCHS} complete...")

# --- EVALUATION ---
print("\n⚔️  THE SHOWDOWN: EVALUATING ON TEST DATA...")

# Prepare Test Data (Last Sequence only per engine)
X_test_seq = []
valid_unit_ids = []

for unit_id in test_df['unit_number'].unique():
    unit_data = test_df[test_df['unit_number'] == unit_id][FEATURES].values
    if len(unit_data) >= SEQUENCE_LENGTH:
        X_test_seq.append(unit_data[-SEQUENCE_LENGTH:])
        valid_unit_ids.append(unit_id)

X_test_tensor = torch.tensor(np.array(X_test_seq), dtype=torch.float32)

# Get True RUL for these valid engines
# (Filter y_true to match engines that had enough data)
y_true_valid = y_true.iloc[[i-1 for i in valid_unit_ids]] # unit_id is 1-based index

# 1. Random Forest Predict
# RF needs the LAST ROW of the test data (Instantaneous)
X_test_rf = test_df.groupby('unit_number').last().reset_index()
X_test_rf = X_test_rf[X_test_rf['unit_number'].isin(valid_unit_ids)][FEATURES]
y_pred_rf = rf_model.predict(X_test_rf)

# 2. LSTM Predict
model.eval()
with torch.no_grad():
    y_pred_lstm = model(X_test_tensor).numpy()

# Scores
rmse_rf = np.sqrt(mean_squared_error(y_true_valid, y_pred_rf))
rmse_lstm = np.sqrt(mean_squared_error(y_true_valid, y_pred_lstm))

print(f"\n=============================================")
print(f"   🏆 FINAL RESULTS (LOWER ERROR IS BETTER) 🏆   ")
print(f"=============================================")
print(f"🌲 Random Forest RMSE:  {rmse_rf:.2f} cycles")
print(f"🧠 PyTorch LSTM RMSE:   {rmse_lstm:.2f} cycles")
print(f"=============================================")

if rmse_lstm < rmse_rf:
    print("✅ CONCLUSION: Deep Learning (LSTM) wins! It understands the history.")
else:
    print("⚠️ CONCLUSION: Random Forest won. (LSTM might need more epochs).")