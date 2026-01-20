import streamlit as st
import time
import json
import paho.mqtt.client as mqtt
import pandas as pd
import plotly.graph_objects as go
import queue
import pickle
import numpy as np
import torch
import torch.nn as nn

# --- BROKER LIST (Fail-Safe Strategy) ---
BROKERS = [
    "broker.hivemq.com",      # Primary
    "test.mosquitto.org",     # Backup 1
    "mqtt.eclipseprojects.io" # Backup 2
]
TOPIC = "factory/engine/01"

FEATURES = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 
            's12', 's13', 's14', 's15', 's17', 's20', 's21']
SEQUENCE_LENGTH = 50
RUL_CAP = 125

# --- 1. DEFINE THE BRAIN ---
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

# --- 2. SETUP ---
st.set_page_config(page_title="AeroTwin SOTA", layout="wide")

@st.cache_resource
def load_ai():
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        model = SOTA_LSTM(input_size=len(FEATURES))
        model.load_state_dict(torch.load('sota_model.pth', map_location=torch.device('cpu')))
        model.eval()
        return model, scaler
    except Exception as e:
        st.error(f"Error loading AI: {e}")
        return None, None

model, scaler = load_ai()

# --- UI LAYOUT ---
st.title("✈️ AeroTwin: SOTA Deep Learning System")
st.markdown(f"**Architecture:** Bi-Directional LSTM (14 Sensors) | **RMSE:** 15.06")

kpi1, kpi2, kpi3 = st.columns(3)
with kpi1: st_cycle = st.empty()
with kpi2: st_prediction = st.empty()
with kpi3: st_status = st.empty()

st.markdown("---")

# Train Section
st.subheader("📉 Phase 1: Failure Simulation (Training Data)")
col1_chart, col1_table = st.columns([2, 1])
with col1_chart: train_chart_placeholder = st.empty()
with col1_table: 
    st.markdown("##### 📝 Training Log")
    train_table_placeholder = st.empty()

st.markdown("---")

# Test Section
st.subheader("🎯 Phase 2: Accuracy Verification (Test Data)")
col2_chart, col2_table = st.columns([2, 1])
with col2_chart: test_chart_placeholder = st.empty()
with col2_table: 
    st.markdown("##### 📝 Verification Log")
    test_table_placeholder = st.empty()

# --- STATE ---
if "train_data" not in st.session_state:
    st.session_state.train_data = pd.DataFrame(columns=["cycle"] + FEATURES)
if "test_data" not in st.session_state:
    st.session_state.test_data = pd.DataFrame(columns=["cycle"] + FEATURES)
if "history_train" not in st.session_state:
    st.session_state.history_train = pd.DataFrame(columns=["Cycle", "AI_RUL", "True_RUL", "Diff"])
if "history_test" not in st.session_state:
    st.session_state.history_test = pd.DataFrame(columns=["Cycle", "AI_RUL", "True_RUL", "Diff"])

# --- MQTT SETUP (ROBUST CONNECT) ---
data_queue = queue.Queue()
def on_message(client, userdata, message):
    try:
        payload = json.loads(message.payload.decode())
        data_queue.put(payload)
    except: pass

# Use explicit Callback Version 1 to silence warnings
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
client.on_message = on_message

connected = False
for broker in BROKERS:
    try:
        st.toast(f"🔌 Connecting to {broker}...", icon="🔄")
        client.connect(broker, 1883, 60)
        client.subscribe(TOPIC)
        client.loop_start()
        st.toast(f"✅ Connected to {broker}!", icon="✅")
        connected = True
        break
    except Exception as e:
        print(f"Failed {broker}: {e}")
        continue

if not connected:
    st.error("❌ Could not connect to any MQTT Broker. Check Internet.")
    st.stop()

# --- HELPERS ---
def prepare_input(df, scaler, seq_length):
    data_matrix = df[FEATURES].values
    if len(data_matrix) >= seq_length:
        input_data = data_matrix[-seq_length:]
    else:
        missing_rows = seq_length - len(data_matrix)
        first_row = data_matrix[0]
        padding = np.tile(first_row, (missing_rows, 1))
        input_data = np.vstack([padding, data_matrix])
    input_scaled = scaler.transform(input_data)
    return torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0)

def draw_chart(dataframe, title_prefix):
    plot_df = dataframe.tail(100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["cycle"], y=plot_df["s2"], name="Inlet Temp", line=dict(color='#FF4B4B')))
    fig.add_trace(go.Scatter(x=plot_df["cycle"], y=plot_df["s14"], name="Core Speed", yaxis="y2", line=dict(color='#1F77B4')))
    fig.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Cycle",
        yaxis=dict(title="Temp", side="left"),
        yaxis2=dict(title="RPM", overlaying="y", side="right"),
        legend=dict(x=0, y=1.1, orientation="h")
    )
    return fig

# --- MAIN LOOP (BACKGROUND PROCESSING) ---
while True:
    # 1. PROCESS ALL DATA IN QUEUE (Background)
    # This prevents UI freezing when data comes fast
    latest_payload = None
    updates_made = False
    
    # Process up to 50 items at once before redrawing
    batch_count = 0
    while not data_queue.empty() and batch_count < 50:
        payload = data_queue.get()
        latest_payload = payload
        updates_made = True
        batch_count += 1
        
        current_cycle = payload["cycle"]
        mode = payload.get("type", "unknown").upper()
        
        # Update Dataframes
        new_row = {"cycle": current_cycle}
        new_row.update(payload["sensors"])
        
        if mode == "TRAIN":
            st.session_state.train_data = pd.concat([st.session_state.train_data, pd.DataFrame([new_row])], ignore_index=True)
            active_df = st.session_state.train_data
        else:
            st.session_state.test_data = pd.concat([st.session_state.test_data, pd.DataFrame([new_row])], ignore_index=True)
            active_df = st.session_state.test_data

        # AI Inference
        pred_val = 0
        if model and scaler:
            input_tensor = prepare_input(active_df, scaler, SEQUENCE_LENGTH)
            with torch.no_grad():
                pred_val = model(input_tensor).item()

        # Log History
        true_rul = payload.get("true_rul", 0)
        error = int(pred_val - true_rul)
        
        log_entry = {
            "Cycle": current_cycle,
            "AI_RUL": int(pred_val),
            "True_RUL": true_rul,
            "Diff": error
        }
        
        if mode == "TRAIN":
            st.session_state.history_train = pd.concat([pd.DataFrame([log_entry]), st.session_state.history_train], ignore_index=True)
        else:
            st.session_state.history_test = pd.concat([pd.DataFrame([log_entry]), st.session_state.history_test], ignore_index=True)

    # 2. DRAW UI ONCE (Foreground)
    if updates_made and latest_payload:
        mode = latest_payload.get("type", "unknown").upper()
        
        # Update Metrics
        display_pred = f"{int(pred_val)} Cycles"
        

        st_cycle.metric("Current Cycle", latest_payload["cycle"])
        st_prediction.metric("AI Prediction", display_pred)
        
        if "true_rul" in latest_payload:
            st_status.metric("Ground Truth", f"{latest_payload['true_rul']}", delta=f"Diff: {error}", delta_color="inverse")

        # Update Graphs & Tables (Only the active mode)
        if mode == "TRAIN":
            fig = draw_chart(st.session_state.train_data, "Failure Simulation")
            train_chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"train_{time.time()}")
            train_table_placeholder.dataframe(st.session_state.history_train, height=300, hide_index=True)
        else:
            fig = draw_chart(st.session_state.test_data, "Verification")
            test_chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"test_{time.time()}")
            test_table_placeholder.dataframe(st.session_state.history_test, height=300, hide_index=True)

    time.sleep(0.05)