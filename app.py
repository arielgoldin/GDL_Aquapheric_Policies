import streamlit as st
import pandas as pd
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

# Set the layout to wide
st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Custom CSS to make the app wider
st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 90%;
        padding-left: 5%;
        padding-right: 5%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the representative scenarios data
representative_scenarios_df = pd.read_csv("results/representative_scenarios.csv", index_col=0)

# Scenario descriptions
scenario_descriptions = {
    "chapala_incident": "In May 2024, a heat wave caused the national electric grid to malfunction leading to considerable damage to the main Chapala pumping system that lasted a few days. Due to lack of data, this scenario was artificially generated where Chapala flow is set to 1.8 m3/s representing the water flowing usually through the old system and the rest are set as baseline.",
    "2021_drought": "In 2021, due to a combination of an intense drought and a spike in consumption due to COVID, the Calderon Dam reached critical levels and had to be shut down leaving around half a million people without water supply for almost 4 months. This scenario motivated the completion of the Aquapheric.",
    "groundwater_scarcity" : "The aquifers that provide water to Guadalajara are over-exploited. This fictional but possible scenario set the Pozos flow to three times less than its 2020 average values and Toluquilla to half."
}

# Sidebar inputs for the scenario
st.sidebar.header('Scenario Selection')
scenario_name = st.sidebar.selectbox('Scenario', representative_scenarios_df.index)

# Display the scenario description
st.sidebar.write(scenario_descriptions.get(scenario_name, "No description available for this scenario."))

# Objectives selection
st.sidebar.header('Select Objectives')
objectives_dict = {
    'supplied_demand_GINI': st.sidebar.checkbox('Egalitarian - Supplied Demand', value=False),
    'supply_percapita_GINI': st.sidebar.checkbox('Egalitarian - Supply Per capita', value=False),
    'supply_percapita_average': st.sidebar.checkbox('Utilitarian - Supply Per Capita Average', value=False),
    'energy_costs': st.sidebar.checkbox('Energy Costs Efficiency', value=False),
    'supplied_demand_deficit_PP1': st.sidebar.checkbox('Prioritize PP1', value=False),
    'supplied_demand_deficit_PP2': st.sidebar.checkbox('Prioritize PP2', value=False),
    'supplied_demand_deficit_PP3': st.sidebar.checkbox('Prioritize PP3', value=False),
    'supplied_demand_deficit_Toluquilla': st.sidebar.checkbox('Prioritize Toluquilla', value=False),
    'supplied_demand_deficit_Pozos': st.sidebar.checkbox('Prioritize Pozos', value=False)
}

# Main section
st.title('Distribution Policies Decision Support Tool')
st.write("""
This app facilitates an exploration of objective-based distribution policies for Guadalajara's Aquapheric. These policies were found using an Evolutionary Multi-Objective Optimization Algorithm and an a posteriori filtering approach for selecting the best performing policies for each scenario.
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
    flows = ["chapala_flow", "calderon_lared_flow", "pozos_flow", "toluquilla_flow"]
    scenario = representative_scenarios_df.loc[scenario_name, flows].to_dict()

    # Find the best policies
    best_policies_df = find_best_policies_for_specified_objectives(full_optimization_results, objectives_dict, scenario)

    # Display the drought state
    st.header('Drought State')
    drought_state = get_drought_state(scenario)
    st.write("Based on the current water flows, the urban drought state is considered ", drought_state)

    # Display the parallel coordinates plot
    st.header('Distribution Policies Based on your Objectives')
    best_policy_indices, policy_labels = visualize_best_policies(best_policies_df, objectives_dict)
    st.pyplot()

    # Display the AQP flows for the best performing policies
    st.header('AqP Flows for Best Performing Policies')
    aqp_flows = ['aqp1_PP2_to_PP3', 'aqp2_PP3_to_Pozos', 'aqp3_Pozos_to_Toluquilla', 'aqp4_Toluquilla_to_PP1']
    additional_columns = ['supply_percapita_GINI', 'energy_costs', 'supply_percapita_average']
    best_policies_aqp_flows = best_policies_df.loc[best_policy_indices, aqp_flows + additional_columns]
    best_policies_aqp_flows['policy_labels'] = policy_labels
    best_policies_aqp_flows.set_index('policy_labels', inplace=True)
    st.dataframe(best_policies_aqp_flows)



