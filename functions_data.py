import pandas as pd
from AMG_function import AMG_model_function
import numpy as np


def full_dataframe(df, function=AMG_model_function, experiment_name="",scenarios_in_dataframe=False, scenario={}, folder="experiment_results"):
    
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

def find_compromise(refSet, deficitIndex):

    refSet=refSet.values
    # normalize objectives for calculation of best compromise solution
    nobjs = np.shape(refSet)[1]
    normObjs = np.zeros([np.shape(refSet)[0],nobjs])
    
    for i in range(np.shape(refSet)[0]):
        for j in range(nobjs):
            # take the square root of the deficit so it's less skewed
            if j == deficitIndex:
                normObjs[i,j] = (np.sqrt(refSet[i,j])-np.mean(np.sqrt(refSet[:,j])))/np.std(np.sqrt(refSet[:,j]))
            else:
                 normObjs[i,j] = (refSet[i,j]-np.mean(refSet[:,j]))/np.std(refSet[:,j])
    
    # find best comprommise solution (solution closest to ideal point)
    dists = np.zeros(np.shape(refSet)[0])
    for i in range(len(dists)):
        for j in range(nobjs):
            dists[i] = dists[i] + (normObjs[i,j]-np.min(normObjs[:,j]))**2
            
    compromise = np.argmin(dists)
    
    return compromise

def add_objective_columns(df, objectives_min, objectives_max, compromise_objectives):
    df_copy = df.copy()  # To avoid modifying the original DataFrame
    
    # Group the DataFrame by 'experiment_name'
    grouped = df_copy.groupby('experiment_name')
    
    # Iterate over each group
    for name, group in grouped:
        # Find the indices of the min and max for each objective within the group
        min_indices = {obj: group[obj] == group[obj].min() for obj in objectives_min}
        max_indices = {obj: group[obj] == group[obj].max() for obj in objectives_max}
        
        # Find the compromise solution within the group
        compromise_index = find_compromise(group[compromise_objectives], deficitIndex=0)
        
        # Add columns to the DataFrame for min and max objectives only
        for obj in objectives_min:
            df_copy.loc[group.index, f'{obj}_min'] = False
        for obj in objectives_max:
            df_copy.loc[group.index, f'{obj}_max'] = False
        # Add columns for compromise objectives
        for obj in compromise_objectives:
            df_copy.loc[group.index, f'{obj}_compromise'] = False
        
        # Set True for min, max, and compromise within the group
        for obj in objectives_min:
            df_copy.loc[min_indices[obj].index, f'{obj}_min'] = True
        
        for obj in objectives_max:
            df_copy.loc[max_indices[obj].index, f'{obj}_max'] = True
        
        # Compromise solution
        for obj in compromise_objectives:
            df_copy.loc[group.index[compromise_index], f'{obj}_compromise'] = True
    
    return df_copy

def find_minmax_values(full_df, 
                       objectives_max=['supplied_demand_PP1', 'supplied_demand_PP2', 'supplied_demand_PP3',
                                       'supplied_demand_Toluquilla', 'supplied_demand_Pozos'],
                       objectives_min = ['supplied_demand_deficit_PP1',
                                          'supplied_demand_deficit_PP2', 'supplied_demand_deficit_PP3',
                                          'supplied_demand_deficit_Toluquilla', 'supplied_demand_deficit_Pozos',
                                          "supplied_demand_GINI","supply_percapita_GINI",
                                          'ZAs_below_142','ZAs_below_100', 'ZAs_below_50',
                                          "energy_costs"],
                       compromise_objectives = ["energy_costs","supplied_demand_GINI"]):
       
       ''' Finds the maximum or minimum value for each objective for each formulation'''
       
       df = full_df.copy()
       
       grouped = df.groupby("experiment_name")

       min_max_df = pd.DataFrame()

       for name, group in grouped:

              # Create DataFrame for min values
              min_values = group[objectives_min].min().to_frame().T

              # Create DataFrame for max values
              max_values = group[objectives_max].max().to_frame().T

              compromise_pol = find_compromise(group[compromise_objectives], 100)


              # Combine both DataFrames
              combined_df = pd.concat([min_values, max_values], axis=1)

              for objective in compromise_objectives:
                     combined_df[f"comp_{objective}"]= full_df[objective][compromise_pol]

              combined_df["formulation"]=name

              min_max_df = pd.concat([min_max_df,combined_df], axis=0)

       
       min_max_df = min_max_df.set_index("formulation")
       return min_max_df