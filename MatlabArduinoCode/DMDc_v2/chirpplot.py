import os
import pickle
import matplotlib.pyplot as plt

def plot_all_chirp_signals_stacked(folder='system_id_runs'):
    files = [f for f in os.listdir(folder) if f.endswith('.pkl')]
    if not files:
        print("⚠️ No .pkl files found in the folder.")
        return

    files.sort()  # Optional: to plot them in order
    num_files = len(files)

    fig, axes = plt.subplots(num_files, 1, figsize=(10, 3 * num_files), sharex=True)

    if num_files == 1:
        axes = [axes]  # Ensure it's iterable even with one subplot

    for ax, f in zip(axes, files):
        file_path = os.path.join(folder, f)
        with open(file_path, 'rb') as pkl_file:
            data = pickle.load(pkl_file)

        timestamps = data.get('timestamps')
        chirp_signal = data.get('chirp_signals')

        if timestamps is not None and chirp_signal is not None:
            ax.plot(timestamps, chirp_signal)
            ax.set_ylabel("Chirp (cm/s²)")
            ax.set_title(f)
            ax.grid(True)
        else:
            ax.set_title(f"{f} (invalid)")
            ax.axis('off')

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

# Run the function
plot_all_chirp_signals_stacked()
