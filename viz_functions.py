import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps, cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from pandas.plotting import parallel_coordinates
 
figsize = (11,6)
fontsize = 14
fig_dir = 'temp_figs/'


def violin_plot(dataframe, title="", x_axis="x", y_axis="y", save=True):
    # Set the font to serif
    sns.set(font='serif')

    # Create a folder to save the plot if it doesn't exist
    folder_path = "experiment_results"

    # Extract only the suffixes from the column names
    viz_data = dataframe.rename(columns=lambda x: x.split('_')[-1])

    # Melt the DataFrame to long format for plotting
    viz_data_melted = viz_data.melt(var_name='Outcome')

    # Create the violin plot with customized parameters
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Outcome', y='value', data=viz_data_melted, color='#00A6D6', alpha=0.99)
    plt.title('Violin Plot of Outcomes for {}'.format(title))
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.tight_layout()

    # Save the plot to the experiment_results folder with the given experiment name
    if save:
        file_name = os.path.join(folder_path, title + '_violin_plot.png')
        plt.savefig(file_name)

    # Show the plot
    plt.show()
    return

def plot_parallel_axis(dataframe, title, y_axis, hue_variable, hue_label):
    """
    Plot parallel axis plot.

    Parameters:
        dataframe (pd.DataFrame): Input DataFrame.
        title (str): Title of the plot.
        y_axis (str): Label for the y-axis.
        hue_variable (str): Column name for the hue variable.
        hue_label (str): Label for the hue variable in the colorbar.

    Returns:
        None
    """
    # Set the font to serif
    plt.rcParams['font.family'] = 'serif'

    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get the maximum value of the hue variable
    max_hue = dataframe[hue_variable].max()

    # Plot each row with a color gradient based on the hue variable
    for index, row in dataframe.iterrows():
        ax.plot([col.split('_')[-1] for col in dataframe.columns[:-1]], row[:-1], color=sns.color_palette("coolwarm", as_cmap=True)(row[hue_variable]/max_hue), alpha=0.9)

    # Customize the plot
    ax.set_title(title)
    ax.set_ylabel(y_axis)

    # Create a dummy ScalarMappable object for the colorbar
    sm = plt.cm.ScalarMappable(cmap=sns.color_palette("coolwarm", as_cmap=True), norm=plt.Normalize(vmin=0, vmax=max_hue))
    sm.set_array([])  # Set an empty array

    # Add the colorbar
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(hue_label)

    plt.tight_layout()

    # Save the plot
    plt.savefig(f"experiment_results/{title}_parallel_axis_plot.png")

    # Show the plot
    plt.show()
    return 


def clustered_box_plot(dataframe, title="", x_axis="x", y_axis="y", save=True):
    # Set the font to serif
    sns.set(font='serif')

    # Create a folder to save the plot if it doesn't exist
    folder_path = "experiment_results"

    # Melt the DataFrame to long format for plotting
    viz_data_melted = dataframe.melt(id_vars='experiment', var_name='Variable', value_name='Value')

    # Create the clustered box plot with customized parameters
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Variable', y='Value', hue='experiment', data=viz_data_melted, palette='colorblind')
    plt.title('Clustered Box Plot of Outcomes for {}'.format(title))
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.tight_layout()

    # Save the plot to the experiment_results folder with the given experiment name
    if save:
        file_name = os.path.join(folder_path, title + '_clustered_box_plot.png')
        plt.savefig(file_name)

    # Show the plot
    plt.show()
    return