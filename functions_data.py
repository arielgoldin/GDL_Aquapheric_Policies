import pandas as pd
from AMG_function import AMG_model_function
import numpy as np


def full_dataframe(df, scenarios_in_dataframe, function=AMG_model_function, experiment_name="", scenario={}, folder="experiment_results"):
    '''
    This function generates a comprehensive DataFrame by applying a specified function (default: AMG_model_function) to each row 
    of an input DataFrame, which contains scenario data and aquifer operation policies. The function iterates over the rows, applies 
    the specified function to simulate the outcomes of these policies, and appends the results to the original data.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing aquifer operation policies and scenario data.
    scenarios_in_dataframe (bool): Flag indicating if scenario data is included within the input DataFrame. If True, scenario data 
                                   will be extracted from the DataFrame; otherwise, it will be taken from the provided scenario dict.
    function (function): The function to be applied to each row of the input DataFrame to simulate outcomes (default: AMG_model_function).
    experiment_name (str): Name of the experiment, used for labeling and saving the output DataFrame (default: empty string).
    scenario (dict): Dictionary containing scenario data to be passed to the function if scenarios_in_dataframe is False (default: empty dict).
    folder (str): Directory path to save the output DataFrame (default: "experiment_results").

    Returns:
    results_df (pd.DataFrame): A DataFrame containing the original policies/scenario data concatenated with the results from the simulation function.
    '''
    
    # Define the names of the columns that could be used as arguments for the function
    aqp_segments = ["aqp4_Toluquilla_to_PP1", "aqp1_PP2_to_PP3", "aqp2_PP3_to_Pozos", "aqp3_Pozos_to_Toluquilla"]
    flows = ["chapala_flow", "calderon_lared_flow", "pozos_flow", "toluquilla_flow","scenario"]

    
    
    # Defining what arguments we need to keep on the dataframe that we will input to the function
    if scenarios_in_dataframe == True: 
        arguments=aqp_segments+flows
    else: 
        # If the flows are inputed as a dict, we don't need the flows in the df
        arguments=aqp_segments
    
    #Preparing the dataframe with the columns that will be inputed to the function
    input_data = df[arguments]

    if "name" in scenario.keys():
        scenario.pop("name")

    # Create an empty DataFrame to store the results
    results_df = pd.DataFrame()

    # Iterate over the rows of the policies DataFrame
    for index, row in input_data.iterrows():
        # Call the AMG_model function with the values from the current row
        outputs = AMG_model_function(**row.to_dict(),**scenario)  # Pass the row values as keyword arguments
        
        # Append the outputs as a new row to the results DataFrame
        results_df = results_df.append(outputs, ignore_index=True)

    # Concatenate the original policies DataFrame with the results DataFrame column-wise
    
    results_df = pd.concat([input_data, results_df], axis=1)
    if "experiment_name" in df.columns:
        experiment_names = df["experiment_name"]
        results_df = pd.concat([results_df, experiment_names], axis = 1)  

    else: results_df["experiment"]=experiment_name

    


    #results_df.to_csv(f"{folder}/{experiment_name}.csv", index=False)

    return results_df


def find_best_policies_for_specified_objectives(df, objectives_dict, scenario):
    '''
    This function filters and identifies the best policies from a DataFrame based on specified objectives. It marks the best policies
    according to the selected objectives (minimizing or maximizing specific metrics) and adds a "No Policy" scenario as a baseline 
    for comparison.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing various policies and their corresponding metrics.
    objectives_dict (dict): A dictionary where keys are objective names and values are booleans indicating whether to include the objective in the filtering.
    scenario (dict): A dictionary containing the scenario data (e.g., flows) to simulate the "No Policy" scenario.

    Returns:
    pd.DataFrame: A DataFrame with additional columns indicating the best-performing policies based on the selected objectives and the 
                  compromise solution. Includes a "No Policy" scenario for baseline comparison.
    '''

    df_copy = df.copy()  # To avoid modifying the original DataFrame

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
    
    # Filter objectives based on the dictionary
    selected_objectives = [obj for obj, use in objectives_dict.items() if use]

    # Add columns to the DataFrame for min or max, and compromise objectives only
    for obj in selected_objectives:
        if obj in objectives_min:
            df_copy[f'{obj}_min'] = False
        elif obj in objectives_max:
            df_copy[f'{obj}_max'] = False
        df_copy[f'{obj}_compromise'] = False
    
    # Determine if each objective should find the min or max
    for obj in selected_objectives:
        if obj in objectives_min:
            min_value = df_copy[obj].min()
            min_indices = df_copy[df_copy[obj] == min_value].index
            df_copy.loc[min_indices, f'{obj}_min'] = True
        elif obj in objectives_max:
            max_value = df_copy[obj].max()
            max_indices = df_copy[df_copy[obj] == max_value].index
            df_copy.loc[max_indices, f'{obj}_max'] = True
    
    # Find the compromise solution
    compromise_index = find_compromise(df_copy[selected_objectives])
    if compromise_index is not None:
        df_copy.loc[compromise_index, [f'{obj}_compromise' for obj in selected_objectives]] = True

    # Add a "No Policy" row with all AQP flows set to zero
    no_policy_flows = {'aqp1_PP2_to_PP3': 0, 'aqp2_PP3_to_Pozos': 0, 'aqp3_Pozos_to_Toluquilla': 0, 'aqp4_Toluquilla_to_PP1': 0}
    no_policy_row = pd.Series(AMG_model_function(**scenario, **no_policy_flows))
    
    # Create a column to identify the "no policy" policy
    df_copy["no_policy"] = False
    no_policy_row["no_policy"] = True
    df_copy = pd.concat([df_copy, no_policy_row.to_frame().T], ignore_index=True)
    df_copy.fillna(0, inplace=True)

    return df_copy


def find_compromise(refSet):
    '''
    This function identifies the compromise solution from a set of policy objectives. The compromise solution is the one that 
    balances the trade-offs between the objectives, minimizing the overall distance from the ideal performance across all objectives.

    The function considers two types of objectives: those to be minimized and those to be maximized. It normalizes the objectives 
    and calculates the Euclidean distance of each policy from the ideal point, selecting the one closest to it as the compromise 
    solution.

    Parameters:
    refSet (pd.DataFrame): A DataFrame containing the performance metrics of different policies across various objectives. Each 
                           column represents an objective, and each row represents a policy.

    Returns:
    int: The index of the compromise solution (the policy that is closest to the ideal point across all objectives).
    '''

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
    
    nobjs = refSet.shape[1]
    normObjs = np.zeros(refSet.shape)

    for i in range(refSet.shape[0]):
        for j in range(nobjs):
            if refSet.columns[j] in objectives_max:
                normObjs[i, j] = (-refSet.iloc[i, j] + refSet.iloc[:, j].mean()) / refSet.iloc[:, j].std()
            elif refSet.columns[j] in objectives_min:
                normObjs[i, j] = (-refSet.iloc[:, j].mean() + refSet.iloc[i, j]) / refSet.iloc[:, j].std()

    dists = np.zeros(refSet.shape[0])
    for i in range(len(dists)):
        for j in range(nobjs):
            dists[i] += (normObjs[i, j] - np.min(normObjs[:, j])) ** 2

    compromise = np.argmin(dists)
    return compromise



def find_best_policies(df, objectives_min, objectives_max, compromise_objectives):
    """
    Identifies the best-performing policies within a DataFrame of optimization results, 
    including policies that are optimal for specific objectives (minimization or maximization) 
    as well as a compromise solution that balances multiple objectives.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing policy performance metrics across multiple objectives. 
        Each row represents a policy and each column represents an objective or a performance measure.

    objectives_min : list of str
        A list of column names corresponding to objectives that should be minimized.

    objectives_max : list of str
        A list of column names corresponding to objectives that should be maximized.

    compromise_objectives : list of str
        A list of column names corresponding to objectives that should be considered for finding 
        a compromise solution, which balances trade-offs among these objectives.

    Returns:
    --------
    pd.DataFrame
        A DataFrame similar to the input, but with additional boolean columns indicating whether a 
        policy is optimal for each specified objective or is the compromise solution. These columns 
        have the suffixes '_min', '_max', and '_compromise' for minimization, maximization, and 
        compromise objectives respectively.
        
    Notes:
    ------
    - The function works by grouping the DataFrame based on the 'experiment_name' column, 
    allowing the identification of best policies within each experiment.
    - The compromise solution is found within each group of policies by considering the 
    specified compromise objectives, using the `find_compromise` function.
    """
    
    df_copy = df.copy()  # To avoid modifying the original DataFrame
    
    # Group the DataFrame by 'experiment_name'
    grouped = df_copy.groupby('experiment_name')
    
    # Add columns to the DataFrame for min and max objectives only
    for obj in objectives_min:
        df_copy[f'{obj}_min'] = False
    for obj in objectives_max:
        df_copy[f'{obj}_max'] = False
    # Add columns for compromise objectives
    for obj in compromise_objectives:
        df_copy[f'{obj}_compromise'] = False
    
    # Iterate over each group
    for name, group in grouped:
        # Find the indices of the min and max for each objective within the group
        for obj in objectives_min:
            min_value = group[obj].min()
            min_indices = group[group[obj] == min_value].index
            df_copy.loc[min_indices, f'{obj}_min'] = True
        for obj in objectives_max:
            max_value = group[obj].max()
            max_indices = group[group[obj] == max_value].index
            df_copy.loc[max_indices, f'{obj}_max'] = True
        
        # Find the compromise solution within the group
        compromise_index = find_compromise(group[compromise_objectives], deficitIndex=0)
        for obj in compromise_objectives:
            df_copy.loc[group.index[compromise_index], f'{obj}_compromise'] = True
    
    return df_copy


def find_minmax_values(full_df, 
                       objectives_max=['supplied_demand_PP1', 'supplied_demand_PP2', 'supplied_demand_PP3',
                                       'supplied_demand_Toluquilla', 'supplied_demand_Pozos',"supply_percapita_average", 
                                       "supply_percapita_PP1", "supply_percapita_PP2", "supply_percapita_PP3", "supply_percapita_Toluquilla", "supply_percapita_Pozos"],
                       objectives_min = ['supplied_demand_deficit_PP1',
                                          'supplied_demand_deficit_PP2', 'supplied_demand_deficit_PP3',
                                          'supplied_demand_deficit_Toluquilla', 'supplied_demand_deficit_Pozos',
                                          "supplied_demand_GINI","supply_percapita_GINI",
                                          'ZAs_below_142','ZAs_below_100', 'ZAs_below_50',
                                          "energy_costs"],
                       compromise_objectives = ["energy_costs","supply_percapita_average",]):
       
       ''' Finds the maximum or minimum value for each objective for each formulation'''
       
       df = full_df.copy()
       
       grouped = df.groupby("experiment_name")

       min_max_df = pd.DataFrame()

       for name, group in grouped:

              # Create DataFrame for min values
              min_values = group[objectives_min].min().to_frame().T

              # Create DataFrame for max values
              max_values = group[objectives_max].max().to_frame().T

              compromise_pol = find_compromise(group[compromise_objectives])


              # Combine both DataFrames
              combined_df = pd.concat([min_values, max_values], axis=1)

              for objective in compromise_objectives:
                     combined_df[f"comp_{objective}"]= full_df[objective][compromise_pol]

              combined_df["formulation"]=name

              min_max_df = pd.concat([min_max_df,combined_df], axis=0)

       
       min_max_df = min_max_df.set_index("formulation")
       return min_max_df