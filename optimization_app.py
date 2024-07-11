import streamlit as st
import pandas as pd
from AMG_optimization import run_optimization, AMG_model_function, full_dataframe
from functions_data import find_best_policies_for_specified_objectives
from functions_viz import visualize_best_policies
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from AMG_function import AMG_model_function, AMG_model_function_int
from functions_data import full_dataframe
from AMG_drought_indicator import get_drought_state

st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the representative scenarios data
representative_scenarios_df = pd.read_csv("results/representative_scenarios.csv", index_col=0)

# Sidebar inputs for the scenario
st.sidebar.header('Scenario Selection')
scenario_name = st.sidebar.selectbox('Scenario', representative_scenarios_df.index)

# Objectives selection
st.sidebar.header('Select Objectives')
objectives_dict = {
    'supplied_demand_deficit_PP1': st.sidebar.checkbox('Supplied Demand Deficit PP1', value=False),
    'supplied_demand_deficit_PP2': st.sidebar.checkbox('Supplied Demand Deficit PP2', value=False),
    'supplied_demand_deficit_PP3': st.sidebar.checkbox('Supplied Demand Deficit PP3', value=False),
    'supplied_demand_deficit_Toluquilla': st.sidebar.checkbox('Supplied Demand Deficit Toluquilla', value=False),
    'supplied_demand_deficit_Pozos': st.sidebar.checkbox('Supplied Demand Deficit Pozos', value=False),
    'supplied_demand_GINI': st.sidebar.checkbox('Supplied Demand GINI', value=False),
    'supply_percapita_GINI': st.sidebar.checkbox('Supply Per Capita GINI', value=False),
    'energy_costs': st.sidebar.checkbox('Energy Costs', value=False),
    'supplied_demand_PP1': st.sidebar.checkbox('Supplied Demand PP1', value=False),
    'supplied_demand_PP2': st.sidebar.checkbox('Supplied Demand PP2', value=False),
    'supplied_demand_PP3': st.sidebar.checkbox('Supplied Demand PP3', value=False),
    'supplied_demand_Toluquilla': st.sidebar.checkbox('Supplied Demand Toluquilla', value=False),
    'supplied_demand_Pozos': st.sidebar.checkbox('Supplied Demand Pozos', value=False),
    'supply_percapita_average': st.sidebar.checkbox('Supply Per Capita Average', value=False)
}

# Main section
st.title('Water Supply Optimization Application')
st.write("""
This app allows you to select a water supply optimization scenario and visualize the best policies based on the selected objectives.
""")

# Button to read the CSV and display results
if st.sidebar.button('Load and Visualize'):
    # Close any existing figures
    plt.close('all')

    # Load the corresponding CSV file for the selected scenario
    csv_file_path = f"results/optimization_results_sd1ep0.04nfe20000.csv"
    full_optimization_results = pd.read_csv(csv_file_path, index_col="policy")
    full_optimization_results = full_optimization_results.loc[full_optimization_results["experiment_name"] == "Sup. Dem. Deficit, Energy & Sup. PerCap. GINI"]
    full_optimization_results = full_optimization_results.loc[full_optimization_results["scenario"] == scenario_name]

    # Extract the scenario details from the dataframe
    scenario = representative_scenarios_df.loc[scenario_name].to_dict()

    # Find the best policies
    best_policies_df = find_best_policies_for_specified_objectives(full_optimization_results, objectives_dict)

    # Display the drought state
    st.header('Drought State')
    drought_state = get_drought_state(scenario)
    st.write(drought_state)

    # Display the parallel coordinates plot
    st.header('Parallel Coordinates Plot of Best Policies')
    best_policy_indices = visualize_best_policies(best_policies_df, objectives_dict)
    st.pyplot()

    # Display the AQP flows for the best performing policies
    st.header('AQP Flows for Best Performing Policies')
    aqp_flows = ['aqp1_PP2_to_PP3', 'aqp2_PP3_to_Pozos', 'aqp3_Pozos_to_Toluquilla', 'aqp4_Toluquilla_to_PP1']
    best_policies_aqp_flows = best_policies_df.loc[best_policy_indices, aqp_flows]
    st.dataframe(best_policies_aqp_flows)

# Main section
st.title('Water Supply Optimization Application')
st.write("""
This app allows you to select a water supply optimization scenario and visualize the best policies based on the selected objectives.
""")
