import matplotlib.pyplot as plt
import os
import numpy as np
import torch

def plot_loss(loss_lists, path, filename):
    """
    Plot the evolution of two loss lists over epochs and save the plot to a file.

    Parameters:
    - loss_lists: List of two lists, each containing loss values (floats or ints) corresponding to each epoch.
    - path: Directory path where the plot should be saved.
    - filename: Name of the file to save the plot (including extension, e.g., 'loss_plot.png').
    """
    
    # Ensure the directory exists; if not, create it
    if not os.path.exists(path):
        os.makedirs(path)

    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Plot each loss list with different labels and colors
    colors = ['blue', 'red', 'green', 'yellow']
    for idx, (label, loss_list) in enumerate(loss_lists.items()):
        plt.plot(loss_list, label=label, color=colors[idx])
    
    
    plt.title('Loss Evolution Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Save the plot to the specified file
    full_path = os.path.join(path, filename)
    plt.savefig(full_path)
    plt.close()  # Close the plot to free up memory

def save_model_state(model, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    filename = path + filename
    print('saving model to', filename)
    torch.save(model.state_dict(), filename)