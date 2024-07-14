import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
from matplotlib import colormaps, cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from pandas.plotting import parallel_coordinates
from ema_workbench.analysis import parcoords

figsize = (11, 6)
fontsize = 14
fig_dir = 'temp_figs/'

def visualize_best_policies(best_policies_df, objectives_dict):
    ZA_names = ["PP1", "PP2", "PP3", "Toluquilla", "Pozos"]

    # Identify the columns that indicate best performance (ending in '_min', '_max', or '_compromise')
    objectives_min = ['supplied_demand_deficit_PP1',
                      'supplied_demand_deficit_PP2', 
                      'supplied_demand_deficit_PP3',
                      'supplied_demand_deficit_Toluquilla', 
                      'supplied_demand_deficit_Pozos',
                      "supplied_demand_GINI",
                      "supply_percapita_GINI",
                      "energy_costs"]

    objectives_max = ['supplied_demand_PP1', 
                      'supplied_demand_PP2', 
                      'supplied_demand_PP3',
                      'supplied_demand_Toluquilla', 
                      'supplied_demand_Pozos',
                      'supply_percapita_PP1', 
                      'supply_percapita_PP2', 
                      'supply_percapita_PP3',
                      'supply_percapita_Toluquilla', 
                      'supply_percapita_Pozos', 
                      "supply_percapita_average"]
    
    # Ensure that the null policy will always be displayed
    best_performance_columns = ["no_policy"]

    for obj in objectives_dict.keys():
        if objectives_dict[obj]:
            if obj in objectives_min:
                best_performance_columns.append(f"{obj}_min")
            elif obj in objectives_max:
                best_performance_columns.append(f"{obj}_max")
            best_performance_columns.append(f"{obj}_compromise")

    # Print the columns expected for performance
    print("Expected best performance columns:", best_performance_columns)

    # Print the columns present in the DataFrame
    print("Columns in best_policies_df:", best_policies_df.columns.tolist())

    # Check if the best_performance_columns exist in the DataFrame
    missing_columns = [col for col in best_performance_columns if col not in best_policies_df.columns]
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} are missing from the DataFrame.")

    # Create a dictionary for labeling the policies
    policy_labels = {
        f"{obj}_min": f"Best {obj}" for obj in objectives_dict.keys() if objectives_dict[obj] and obj in objectives_min
    }
    policy_labels.update({
        f"{obj}_max": f"Best {obj}" for obj in objectives_dict.keys() if objectives_dict[obj] and obj in objectives_max
    })
    policy_labels.update({
        f"{obj}_compromise": f"Compromise policy" for obj in objectives_dict.keys() if objectives_dict[obj]
    })
    policy_labels.update({
        "no_policy": "No policy"
    })

    # Filter rows where at least one of the best performance columns is True
    best_performing_policies_df = best_policies_df[best_policies_df[best_performance_columns].any(axis=1)]

    # Debugging: print the best_performing_policies_df to check if it's populated correctly
    print("Best Performing Policies DataFrame:")
    print(best_performing_policies_df)

    # Create a dictionary to map index to labels
    index_labels = {}
    for col, label in policy_labels.items():
        if col in best_performing_policies_df.columns:
            indices = best_performing_policies_df[best_performing_policies_df[col] == True].index
            for idx in indices:
                if idx in index_labels:
                    index_labels[idx] += f", {label}"
                else:
                    index_labels[idx] = label

    # Add a new column for policy labels
    best_performing_policies_df['policy_labels'] = best_performing_policies_df.index.map(index_labels)

    # Select the supply_per_capita columns for the five zones of analysis (ZA)
    supply_per_capita_columns = [
        'supply_percapita_PP1', 'supply_percapita_PP2', 'supply_percapita_PP3',
        'supply_percapita_Toluquilla', 'supply_percapita_Pozos'
    ]
    data = best_performing_policies_df[supply_per_capita_columns]

    # Debugging: print the data to be plotted
    print("Data to be plotted:")
    print(data)

    # Get limits for parallel coordinates plot
    limits = pd.read_csv("results/limits.csv")

    # Create the parallel axes plot
    paraxes = parcoords.ParallelAxes(limits)

    # Plot each row
    colors = plt.cm.tab10.colors
    for i, (index, row) in enumerate(data.iterrows()):
        label = index_labels.get(index, str(index))
        if 'Compromise policy' in label:
            paraxes.plot(row.to_frame().T, label=label, color='#00A6D6', linewidth=4)
        else:
            color = 'darkgrey' if 'No policy' in label else colors[i % len(colors)]
            paraxes.plot(row.to_frame().T, label=label, color=color, linewidth=2)

    # Add horizontal lines for specific values
    for ax in paraxes.axes:
        ax.axhline(y=0.86, color='green', linestyle='--', linewidth=1)
        ax.axhline(y=0.58, color='orange', linestyle='--', linewidth=1)
        ax.axhline(y=0.29, color='red', linestyle='--', linewidth=1)

        # Add labels to the horizontal lines
        ax.text(1.01, 0.86, '147', transform=ax.get_yaxis_transform(), color='green', ha='left', va='center')
        ax.text(1.01, 0.58, '100', transform=ax.get_yaxis_transform(), color='orange', ha='left', va='center')
        ax.text(1.01, 0.29, '50', transform=ax.get_yaxis_transform(), color='red', ha='left', va='center')

    # Add x-axis label
    paraxes.fig.text(0.5, 0.04, 'Supply per capita [l/day/person]', ha='center', va='center', fontsize="large")

    # Update axis labels
    for ax in paraxes.axes:
        labels = [label.get_text().replace('supply_percapita_', '') for label in ax.get_xticklabels()]
        ax.set_xticklabels(labels)

    # Add legend and show plot
    paraxes.legend()
    plt.show()
    
    return best_performing_policies_df.index.tolist(), best_performing_policies_df['policy_labels']



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

def plot_colored_parallel_axis(light_grey_data, colored_data, color):
    """
    Plot parallel axis plot with lines colored based on a separate DataFrame.

    Parameters:
        light_grey_data (pd.DataFrame): DataFrame containing data to be plotted in light gray.
        colored_data (pd.DataFrame): DataFrame containing data to be colored.

    Returns:
        None
    """
    # Get unique indices for the minima and maxima across all objectives
    max_indices = colored_data.idxmax()
    min_indices = colored_data.idxmin()
    
    # Combine and filter unique indices
    indices = pd.concat([max_indices, min_indices]).unique()
    indices = [idx for idx in indices if idx in colored_data.index]
    
    # Get limits for the parallel axes
    limits = parcoords.get_limits(light_grey_data)
    
    # Create parallel axes
    axes = parcoords.ParallelAxes(limits)

    # Plot data in light gray
    axes.plot(light_grey_data, color='lightgrey', lw=0.5, alpha=0.5)
    
    # Plot colored data
    
    axes.plot(colored_data, color=color, lw=1)
    
    # Invert axis if needed
    for column in light_grey_data.columns:
        if '_P' in column:
            axes.invert_axis(column)

    # Set figure size
    fig = plt.gcf()
    fig.set_size_inches((8, 4))

    # Show the plot
    plt.show()


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