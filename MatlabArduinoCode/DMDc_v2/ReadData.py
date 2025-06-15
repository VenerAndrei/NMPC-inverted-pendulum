
import pickle
import sys
import os
import matplotlib.pyplot as plt

def print_and_plot_data(filename):
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    print("\n=== Metadata ===")
    for key, value in data.get("metadata", {}).items():
        print(f"{key}: {value}")

    print("\n=== Keys and Lengths ===")
    for key in data:
        if key != "metadata":
            print(f"{key}: {len(data[key])}")

    print("\n=== First 5 Samples ===")
    for i in range(min(5, len(data['timestamps']))):
        row = [f"{data[k][i]:.3f}" for k in data if k != 'metadata']
        print(f"{i+1:2d}: " + ", ".join(row))

    timestamps = data['timestamps']
    keys_to_plot = [k for k in data.keys() if k not in ('metadata', 'timestamps', 'pot_values')]

    plt.figure(figsize=(12, 2 * len(keys_to_plot)))
    for i, key in enumerate(keys_to_plot):
        plt.subplot(len(keys_to_plot), 1, i + 1)
        plt.plot(timestamps, data[key], linewidth=1.5)
        plt.ylabel(key.replace('_', ' ').title())
        plt.grid(True)
        if i == len(keys_to_plot) - 1:
            plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python PrintAndPlotPKLData.py <filename.pkl>")
    else:
        print_and_plot_data(sys.argv[1])
