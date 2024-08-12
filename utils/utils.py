import matplotlib.pyplot as plt
import os

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
    plt.figure(figsize=(10, 6))
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
