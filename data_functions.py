import pandas as pd
from AMG_function import AMG_model_function

def full_dataframe (results, function=AMG_model_function, experiment_name="",scenario={}):
    aqp_segments = ["aqp4_Toluquilla_to_PP1", "aqp1_PP2_to_PP3", "aqp2_PP3_to_Pozos", "aqp3_Pozos_to_Toluquilla"]

    # Create an empty DataFrame to store the results
    policies = results[aqp_segments]
    results_df = pd.DataFrame()
    scenario_name =scenario.pop("name")
    

    # Iterate over the rows of the policies DataFrame
    for index, row in policies.iterrows():
        # Call the AMG_model function with the values from the current row
        outputs = AMG_model_function(**row.to_dict(),**scenario)  # Pass the row values as keyword arguments
        # Append the outputs as a new row to the results DataFrame
        results_df = results_df.append(outputs, ignore_index=True)

    # Concatenate the original policies DataFrame with the results DataFrame column-wise
    results_df = pd.concat([policies, results_df], axis=1)
    results_df["experiment"]=experiment_name
    results_df["scenario"]=scenario_name


    results_df.to_csv(f"experiment_results/{experiment_name}.csv")

    return results_df