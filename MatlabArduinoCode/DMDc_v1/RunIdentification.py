import numpy as np
import matplotlib.pyplot as plt
from pydmd import DMDc
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime
import scipy.signal as signal

# Configuration
DATA_DIR = "system_id_runs"
PLOTS_DIR = os.path.join(DATA_DIR, "analysis_plots")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def list_saved_runs():
    """List all available saved runs in the data directory"""
    print("\nAvailable saved runs:")
    runs = []
    for idx, filename in enumerate(sorted(os.listdir(DATA_DIR))):
        if filename.endswith(".pkl") and (filename.startswith("raw_run") or filename.startswith("analysis_run")):
            run_time = filename[8:-4] if filename.startswith("raw_run") else filename[16:-4]
            runs.append(filename)
            print(f"{idx + 1}: {filename} ({run_time})")
    return runs

def load_run_data(filename):
    """Load data from a saved run"""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def perform_dmdc_analysis(X, U, dt, rank=None):
    """
    Perform DMDc analysis with corrected matrix handling
    X: (n_states, n_samples)
    U: (n_inputs, n_samples)
    """
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.T).T
    
    # Prepare input matrix with correct dimensions
    # DMDc expects U to be (n_inputs, n_samples-1)
    U_processed = U[:, :-1]
    
    print("\nMatrix dimensions for DMDc:")
    print(f"X_scaled[:, :-1]: {X_scaled[:, :-1].shape} (states at time k)")
    print(f"X_scaled[:, 1:]: {X_scaled[:, 1:].shape} (states at time k+1)")
    print(f"U_processed: {U_processed.shape} (inputs at time k)")
    
    # Create DMDc object with proper settings
    dmd = DMDc(svd_rank=rank, opt=True)
    
    try:
        # The key fix: properly aligned matrices
        dmd.fit(X_scaled[:, :-1], X_scaled[:, 1:], U_processed)
    except Exception as e:
        print(f"\nDMDc fitting failed with error: {str(e)}")
        return None
    
    return {
        'A': dmd.A,
        'B': dmd.B,
        'eigenvalues': dmd.eigs,
        'modes': dmd.modes.T,
        'scaler': scaler,
        'dmd': dmd
    }

def plot_states_and_input(X, U, timestamps, save_path=None):
    """
    Plot the state variables and input signal with proper formatting
    
    Args:
        X: State matrix (4 x n_samples)
        U: Input matrix (1 x n_samples)
        timestamps: Time vector (n_samples,)
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(12, 10))
    
    # Create subplots
    ax1 = plt.subplot(5, 1, 1)
    ax2 = plt.subplot(5, 1, 2)
    ax3 = plt.subplot(5, 1, 3)
    ax4 = plt.subplot(5, 1, 4)
    ax5 = plt.subplot(5, 1, 5)
    
    # Plot state variables
    ax1.plot(timestamps, X[0,:], 'b-', linewidth=1.5)
    ax1.set_ylabel('Position (cm)')
    ax1.grid(True)
    
    ax2.plot(timestamps, X[1,:], 'r-', linewidth=1.5)
    ax2.set_ylabel('Velocity (cm/s)')
    ax2.grid(True)
    
    ax3.plot(timestamps, X[2,:], 'g-', linewidth=1.5)
    ax3.set_ylabel('Angle (rad)')
    ax3.grid(True)
    
    ax4.plot(timestamps, X[3,:], 'm-', linewidth=1.5)
    ax4.set_ylabel('Angular Velocity (rad/s)')
    ax4.grid(True)
    
    # Plot input signal
    ax5.plot(timestamps, U[0,:], 'k-', linewidth=1.5)
    ax5.set_ylabel('Chirp Input (cm/s²)')
    ax5.set_xlabel('Time (s)')
    ax5.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def analyze_saved_run():
    """Main analysis function with robust error handling"""
    runs = list_saved_runs()
    if not runs:
        print("No saved runs found in the data directory")
        return
    
    try:
        selection = int(input("\nEnter the number of the run to analyze: ")) - 1
        filename = runs[selection]
    except (ValueError, IndexError):
        print("Invalid selection")
        return
    
    print(f"\nLoading {filename}...")
    data = load_run_data(filename)
    
    # Prepare data with proper dimensions
    X = np.vstack([
        data['positions'],
        data['velocities'],
        data['angles'], 
        data['angular_velocities']
    ])
    U = np.array(data['chirp_signals']).reshape(1, -1)
    plot_states_and_input(X, U, data['timestamps'])


    # print(f"\nOriginal data dimensions:")
    # print(f"X shape: {X.shape} (states × samples)")
    # print(f"U shape: {U.shape} (inputs × samples)")
    
    # # Perform DMDc analysis
    # print("\nPerforming DMDc analysis...")
    # dt = np.mean(np.diff(data['timestamps']))
    # results = perform_dmdc_analysis(X, U, dt, rank=4)
    
    # if results is None:
    #     print("\nAnalysis failed. Please check:")
    #     print("1. All signals should have the same length")
    #     print("2. The chirp signal should be properly excited")
    #     print("3. The system should be sufficiently excited by the input")
    #     print("4. Try adjusting the svd_rank parameter if needed")
    #     return
    
    # # Save and visualize results
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # base_filename = filename[:-4]  # Remove .pkl extension
    # results_filename = os.path.join(DATA_DIR, f"analysis_{base_filename}_{timestamp}.pkl")
    
    # with open(results_filename, 'wb') as f:
    #     pickle.dump({
    #         'original_data': data,
    #         'system_matrices': {
    #             'A': results['A'],
    #             'B': results['B']
    #         },
    #         'eigenvalues': results['eigenvalues'],
    #         'analysis_time': timestamp
    #     }, f)
    
    # print("\n" + "="*50)
    # print("Analysis Complete")
    # print("="*50)
    # print(f"\nSystem Matrix A:\n{results['A']}")
    # print(f"\nInput Matrix B:\n{results['B']}")
    # print(f"\nEigenvalues:\n{results['eigenvalues']}")
    # print(f"\nResults saved to: {results_filename}")

if __name__ == "__main__":
    print("\nInverted Pendulum DMDc Analysis Tool")
    print("===================================")
    analyze_saved_run()