import numpy as np
import serial
import matplotlib.pyplot as plt
from pydmd import DMDc
from sklearn.preprocessing import StandardScaler
import time
import keyboard
import os
from datetime import datetime
import pickle

# Serial port configuration
SERIAL_PORT = 'COM5'  # Change this to your Arduino's serial port
BAUD_RATE = 115200
TIMEOUT = 1

# Data collection parameters
CHIRP_DURATION = 30  # seconds
SAMPLING_RATE = 50   # Hz (matches your 20ms loop)
TOTAL_SAMPLES = int(CHIRP_DURATION * SAMPLING_RATE)

# Create data directory if it doesn't exist
DATA_DIR = "system_id_runs"
os.makedirs(DATA_DIR, exist_ok=True)

def save_raw_data(timestamps, positions, velocities, angles, angular_velocities, chirp_signals, metadata):
    """Save raw data immediately after collection"""
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{DATA_DIR}/raw_run_{timestamp_str}.pkl"
    
    data_dict = {
        'timestamps': timestamps,
        'positions': positions,
        'velocities': velocities,
        'angles': angles,
        'angular_velocities': angular_velocities,
        'chirp_signals': chirp_signals,
        'metadata': metadata
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"\nRaw data saved to {filename}")
    return filename

def collect_data_with_manual_trigger():
    """Collect data with manual trigger for chirp signal"""
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
    time.sleep(2)  # Wait for serial connection to establish
    
    # Initialize arrays to store data
    timestamps = []
    positions = []
    velocities = []
    angles = []
    angular_velocities = []
    chirp_signals = []
    
    print("\n" + "="*50)
    print("Inverted Pendulum Data Collection")
    print("="*50)
    print("\nWaiting for manual trigger...")
    print("Press 's' to start chirp signal and data collection")
    print("Press 'q' to quit at any time")
    
    collecting = False
    start_time = None
    metadata = {
        'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'description': ''
    }
    
    try:
        while True:
            if keyboard.is_pressed('s') and not collecting:
                # Send 'i' command to start chirp
                ser.write(b'i')
                print("\nChirp signal started! Collecting data...")
                collecting = True
                start_time = time.time()
                metadata['chirp_start'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                desc = input("\nEnter a brief description for this run (optional): ")
                if desc:
                    metadata['description'] = desc
            
            if keyboard.is_pressed('q'):
                print("\nUser requested to stop")
                break
            
            if ser.in_waiting:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    try:
                        data = [float(x) for x in line.split('\t')]
                        if len(data) == 5 and collecting:
                            timestamps.append(time.time() - start_time)
                            positions.append(data[0])
                            velocities.append(data[1])
                            angles.append(data[2])
                            angular_velocities.append(data[3])
                            chirp_signals.append(data[4])
                            
                            if len(timestamps) % 50 == 0:
                                print(f"Collected {len(timestamps)}/{TOTAL_SAMPLES} samples")
                            
                            if len(timestamps) >= TOTAL_SAMPLES:
                                print(f"\nSuccessfully collected {TOTAL_SAMPLES} samples")
                                break
                    except ValueError:
                        continue
    except KeyboardInterrupt:
        print("\nData collection interrupted")
    finally:
        ser.close()
        metadata['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata['samples_collected'] = len(timestamps)
        
        if len(timestamps) > 0:
            print(f"\nCollection summary:")
            print(f"- Duration: {timestamps[-1]:.2f} seconds")
            print(f"- Samples: {len(timestamps)}")
            
            # Save raw data immediately
            raw_filename = save_raw_data(
                np.array(timestamps),
                np.array(positions),
                np.array(velocities),
                np.array(angles),
                np.array(angular_velocities),
                np.array(chirp_signals),
                metadata
            )
            
            return raw_filename, metadata
        else:
            print("No data collected")
            return None, None

def perform_identification(raw_filename):
    """Load raw data and perform system identification"""
    if not raw_filename or not os.path.exists(raw_filename):
        print("No valid raw data file provided")
        return None
    
    print(f"\nLoading raw data from {raw_filename}")
    with open(raw_filename, 'rb') as f:
        data = pickle.load(f)
    
    # Prepare data for DMDc
    X = np.vstack([data['positions'], 
                   data['velocities'],
                   data['angles'], 
                   data['angular_velocities']])
    U = data['chirp_signals'].reshape(1, -1)
    
    # Perform DMDc
    dt = 1/SAMPLING_RATE
    print("\nPerforming DMDc analysis...")
    dmd, A, B, scaler = perform_dmdc(X, U, dt, rank=4)
    
    print("\nIdentified system matrix A:")
    print(A)
    print("\nIdentified input matrix B:")
    print(B)
    
    # Create results filename
    results_filename = raw_filename.replace("raw_run", "identified_run")
    
    # Save results
    results = {
        'A_matrix': A,
        'B_matrix': B,
        'scaler': scaler,
        'identification_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'original_data_file': raw_filename,
        'metadata': data['metadata']
    }
    
    with open(results_filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nIdentification results saved to {results_filename}")
    return results_filename

def main():
    # Step 1: Collect and save raw data
    raw_filename, metadata = collect_data_with_manual_trigger()
    
    if not raw_filename:
        return
    
    # Ask user if they want to perform identification now
    perform_now = input("\nPerform system identification now? (y/n): ").lower()
    
    if perform_now == 'y':
        # Step 2: Perform identification on saved data
        results_filename = perform_identification(raw_filename)
        
        if results_filename:
            print("\nSystem identification complete!")
            print(f"Raw data: {raw_filename}")
            print(f"Results: {results_filename}")
    else:
        print("\nRaw data saved. You can perform identification later by running:")
        print(f"perform_identification('{raw_filename}')")

if __name__ == "__main__":
    main()