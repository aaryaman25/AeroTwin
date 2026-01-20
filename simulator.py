import pandas as pd
import time
import json
import paho.mqtt.client as mqtt
import socket

# --- BROKER LIST (Fail-Safe) ---
BROKERS = [
    "broker.hivemq.com",      # Primary
    "test.mosquitto.org",     # Backup 1
    "mqtt.eclipseprojects.io" # Backup 2
]
PORT = 1883
TOPIC = "factory/engine/01"
SPEED = 1.0  # <--- FIXED SPEED (1 Second per Cycle)

# The 14 Key Sensors
FEATURES = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 
            's12', 's13', 's14', 's15', 's17', 's20', 's21']

print("Loading Data...")
cols = ['unit_number', 'time_cycle', 'setting_1', 'setting_2', 'setting_3']
cols += [f's{i}' for i in range(1, 22)]
train_df = pd.read_csv('train_FD001.txt', sep='\s+', header=None, names=cols)
test_df = pd.read_csv('test_FD001.txt', sep='\s+', header=None, names=cols)
rul_true = pd.read_csv('RUL_FD001.txt', sep='\s+', header=None, names=['RUL'])

# --- ROBUST CONNECT FUNCTION ---
def connect_mqtt():
    # Use Version 1 to silence warnings
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    
    for broker in BROKERS:
        try:
            print(f"🔌 Trying to connect to {broker}...")
            client.connect(broker, PORT, 60)
            print(f"✅ Connected to {broker}!")
            return client
        except socket.gaierror:
            print(f"❌ DNS Error on {broker}. Trying next...")
        except Exception as e:
            print(f"❌ Failed to connect to {broker}: {e}")
            
    raise ConnectionError("Could not connect to ANY public broker. Check your internet/DNS.")

client = connect_mqtt()

def stream_engine(dataframe, engine_id, dataset_type, true_rul_value=None):
    engine_data = dataframe[dataframe['unit_number'] == engine_id]
    max_cycle = engine_data['time_cycle'].max()
    
    print(f"\n--- STARTING {dataset_type.upper()} SIMULATION (Engine {engine_id}) ---")
    
    for index, row in engine_data.iterrows():
        current_cycle = int(row['time_cycle'])
        
        sensor_data = {feat: float(row[feat]) for feat in FEATURES}
        payload = {
            "cycle": current_cycle,
            "type": dataset_type,
            "sensors": sensor_data
        }
        
        if dataset_type == 'train':
            payload["true_rul"] = 192 - current_cycle
        elif dataset_type == 'test' and true_rul_value is not None:
            cycles_remaining = max_cycle - current_cycle
            payload["true_rul"] = int(true_rul_value + cycles_remaining)

        client.publish(TOPIC, json.dumps(payload))
        
        # FIXED: Constant Cinematic Speed
        print(f"[TX] {dataset_type} | Cycle {current_cycle}")
        time.sleep(SPEED)

try:
    # 1. Train Run
    stream_engine(train_df, engine_id=1, dataset_type='train')
    
    print("Waiting 3 seconds before Test run...")
    time.sleep(3)
    
    # 2. Test Run
    correct_rul = rul_true.iloc[0]['RUL'] 
    stream_engine(test_df, engine_id=1, dataset_type='test', true_rul_value=correct_rul)

except KeyboardInterrupt:
    client.disconnect()