import matplotlib.pyplot as plt
import seaborn as sns
import os


def violin_plot(dataframe, title, x_axis, y_axis, save):
    # Assuming 'results' is your dataframe and 'viz_outcomes' is your list of columns
    titlee = "optimization supplied demand deficit, 10,000 nfe, rounding=5"
    viz_outcomes = [f"supplied_demand_deficit_{ZA}" for ZA in ZA_names]
    x_axis = "Supply Area"
    y_axis = "Absolute supplied demand deficit"

    # Set the font to serif
    sns.set(font='serif')


    # Create a folder to save the plot if it doesn't exist
    folder_path = "experiment_results"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Subset the DataFrame to include only the columns specified in viz_outcomes
    viz_data = dataframe

    # Melt the DataFrame to long format for plotting
    viz_data_melted = viz_data.melt(var_name='Outcome')

    # Create the violin plot with customized parameters
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Outcome', y='value', data=viz_data_melted, color='#00A6D6', alpha=0.99)
    plt.title('Violin Plot of Outcomes for {}'.format(title))
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.xticks(ticks=range(len(ZA_names)), labels=ZA_names, rotation=45)
    plt.tight_layout()

    # Save the plot to the experiment_results folder with the given experiment name
    file_name = os.path.join(folder_path, experiment_name + '_violin_plot.png')
    plt.savefig(file_name)

    # Show the plot
    plt.show()