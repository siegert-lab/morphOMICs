import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def vect_dim_dist(vectors, save_plot = False, save_filepath = None):

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot box plots for each dimension with Seaborn
    sns.boxplot(data=vectors, ax=ax, palette="Set3")

    # Set titles and labels
    ax.set_title('Box Plot of Each Dimension of the Vector')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Values')
    ax.set_xticklabels([f'Dim {i+1}' for i in range(vectors.shape[1])])

    plt.show()

    if save_plot:
        # Save the plot as an HTML file
        fig.write_html(save_filepath)
        print(f"Plot saved as {save_filepath}")

