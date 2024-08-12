import matplotlib.pyplot as plt
import os
import numpy as np

def plot_loss(loss_list, path, filename):
    """
    Plot the evolution of the loss over epochs and save the plot to a file.

    Parameters:
    - loss_list: List of loss values (floats or ints) corresponding to each epoch.
    - path: Directory path where the plot should be saved.
    - filename: Name of the file to save the plot (including extension, e.g., 'loss_plot.png').
    """
    
    # Ensure the directory exists; if not, create it
    if not os.path.exists(path):
        os.makedirs(path)

    # Create the plot
    plt.figure(figsize=(15, 10))
    plt.plot(loss_list, label='Loss')
    plt.title('Loss Evolution Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Save the plot to the specified file
    full_path = os.path.join(path, filename)
    plt.savefig(full_path)
    plt.close()  # Close the plot to free up memory

def plot_binned_loss(loss_list, path, filename, bins=50):
    """
    Plot the evolution of the loss over epochs with binned values and save the plot to a file.

    Parameters:
    - loss_list: List of loss values (floats or ints) corresponding to each epoch.
    - path: Directory path where the plot should be saved.
    - filename: Name of the file to save the plot (including extension, e.g., 'loss_plot.png').
    - bins: Number of bins to group the loss values.
    """

    # Ensure the directory exists; if not, create it
    if not os.path.exists(path):
        os.makedirs(path)

    # Compute the bin size
    bin_size = len(loss_list) // bins
    
    # Binned loss values
    binned_loss = [np.mean(loss_list[i * bin_size:(i + 1) * bin_size]) for i in range(bins)]

    # Create the plot
    plt.figure(figsize=(15, 10))
    plt.plot(binned_loss, label='Binned Loss')
    plt.title(f'Binned Loss Evolution Over Epochs ({bins} bins)')
    plt.xlabel('Binned Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Save the plot to the specified file
    full_path = os.path.join(path, filename)
    plt.savefig(full_path)
    plt.close() 