import pandas as pd
from AMG_function import AMG_model_function
import numpy as np

def full_dataframe_from_df(df, function=AMG_model_function,  experiment_name="", folder="experiment_results"):
    #define the names of the columns that could be used as arguments for the funciton
    aqp_segments = ["aqp4_Toluquilla_to_PP1", "aqp1_PP2_to_PP3", "aqp2_PP3_to_Pozos", "aqp3_Pozos_to_Toluquilla"]
    flows = ["chapala_flow", "calderon_lared_flow", "pozos_flow", "toluquilla_flow"]
    
    #defining what arguments we need to keep on the dataframe that we will input to the function
    
    arguments=aqp_segments+flows


    #Preparing the dataframe with the columns that will be inputed to the function
    input_data = df[arguments]

    # Create an empty DataFrame to store the results
    results_df = pd.DataFrame()

    # Iterate over the rows of the policies DataFrame
    for index, row in input_data.iterrows():
        # Call the AMG_model function with the values from the current row
        outputs = AMG_model_function(**row.to_dict())  # Pass the row values as keyword arguments
        
        # Append the outputs as a new row to the results DataFrame
        results_df = results_df.append(outputs, ignore_index=True)

    # Concatenate the original policies DataFrame with the results DataFrame column-wise
    results_df = pd.concat([input_data, results_df], axis=1)
    results_df["experiment"]=experiment_name


    results_df.to_csv(f"{folder}/{experiment_name}.csv", index=False)

    return results_df
def full_dataframe(df, function=AMG_model_function, experiment_name="",scenarios_in_dataframe=False, scenario={}, folder="experiment_results"):
    #define the names of the columns that could be used as arguments for the funciton
    aqp_segments = ["aqp4_Toluquilla_to_PP1", "aqp1_PP2_to_PP3", "aqp2_PP3_to_Pozos", "aqp3_Pozos_to_Toluquilla"]
    flows = ["chapala_flow", "calderon_lared_flow", "pozos_flow", "toluquilla_flow","scenario"]
    
    #defining what arguments we need to keep on the dataframe that we will input to the function
    if scenarios_in_dataframe == True: 
        arguments=aqp_segments+flows
        scenario
    else: 
        arguments=aqp_segments

    #Preparing the dataframe with the columns that will be inputed to the function
    input_data = df[arguments]

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
    results_df["experiment"]=experiment_name


    results_df.to_csv(f"{folder}/{experiment_name}.csv", index=False)

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