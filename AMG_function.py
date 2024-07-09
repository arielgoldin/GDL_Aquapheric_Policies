import numpy as np
import pandas as pd


conversion_m3s = 1/(24*60*60*1000) # converts l/day into m3/s

def calculate_demand(population, ZA, domestic_consumption):
    return population[f"population_{ZA}"] * domestic_consumption * conversion_m3s

def calculate_supplied_demand(outcomes, ZA):
    return outcomes[f"supplied_{ZA}"] / outcomes[f"demand_{ZA}"]

def calculate_supply_percapita(outcomes, population, ZA):
    return outcomes[f"supplied_{ZA}"] / population[f"population_{ZA}"] / conversion_m3s

def calculate_supplied_demand_deficit(outcomes, ZA):
    return abs(1 - outcomes[f"supplied_demand_{ZA}"])

def calculate_GINI(outcomes):
    objectives = list(outcomes.values())
    n = len(objectives)
    sorted_objectives = np.sort(objectives)
    diffs = np.abs(np.subtract.outer(sorted_objectives, sorted_objectives)).flatten()
    return np.sum(diffs) / (2.0 * n * np.sum(sorted_objectives))

def AMG_model_function(
    chapala_flow, calderon_lared_flow, pozos_flow, toluquilla_flow,
    aqp4_Toluquilla_to_PP1, aqp1_PP2_to_PP3, aqp2_PP3_to_Pozos, aqp3_Pozos_to_Toluquilla,
    crowding_factor=3.55, 
    chapalaPP1_to_chapalaPP2=0.19, loss_grid=0.35, loss_potabilisation=0.07,
    rounding_outcomes=2, rounding_levers=3, sufficientarian_thresholds=[142, 100, 50, 128],
    scenario="unspecified", experiment_name = "unspecified"):

    ZA_names = ["PP1", "PP2", "PP3", "Toluquilla", "Pozos"]
    input_data_dict = pd.read_csv("data/input_data.csv").set_index('Variable').to_dict()['Value']

    # Calculate population and demand
    population_dict = {f"population_{ZA}": int(input_data_dict[f"domestic_intakes_{ZA}"] * crowding_factor) for ZA in ZA_names}
    demand_outputs = {
        f"demand_{ZA}": 
            ((population_dict[f"population_{ZA}"] * input_data_dict["domestic_consumption"] +
             input_data_dict[f"service_intakes_{ZA}"] * input_data_dict["service_consumption"] +
             input_data_dict[f"industry_intakes_{ZA}"] * input_data_dict["industry_consumption"] +
             input_data_dict[f"public_intakes_{ZA}"] * input_data_dict["public_consumption"]) * conversion_m3s )
             for ZA in ZA_names}
    #total_demand = sum(demand_outputs.values())

    # Calculate extraction and delivered water
    total_extraction = chapala_flow + calderon_lared_flow + toluquilla_flow + pozos_flow
    delivered_outputs = {
        "delivered_PP1": chapala_flow * (1 - chapalaPP1_to_chapalaPP2),
        "delivered_PP2": chapala_flow * chapalaPP1_to_chapalaPP2,
        "delivered_PP3": calderon_lared_flow,
        "delivered_Pozos": pozos_flow,
        "delivered_Toluquilla": toluquilla_flow
    }

    # Calculate potabilized water
    potabilized_outputs = {
        "potabilized_PP1": delivered_outputs["delivered_PP1"] * (1 - loss_potabilisation),
        "potabilized_PP2": delivered_outputs["delivered_PP2"] * (1 - loss_potabilisation),
        "potabilized_PP3": delivered_outputs["delivered_PP3"] * (1 - loss_potabilisation),
        "potabilized_Toluquilla": delivered_outputs["delivered_Toluquilla"] * (1 - loss_potabilisation),
        "potabilized_Pozos": delivered_outputs["delivered_Pozos"]  # no loss for Pozos
    }

    # Calculate additional flows and supplied water
    additional_flows = {
        "additional_flow_PP1": potabilized_outputs["potabilized_PP1"] * -0.12,
        "additional_flow_PP2": 0,
        "additional_flow_PP3": potabilized_outputs["potabilized_Pozos"] * 0.2,
        "additional_flow_Toluquilla": (potabilized_outputs["potabilized_PP1"] * 0.12 + potabilized_outputs["potabilized_Pozos"] * 0.02),
        "additional_flow_Pozos": potabilized_outputs["potabilized_Pozos"] * -0.22
    }
    #Two following lines of codes are from the implementation with integre levers
    #scale = 10**(-rounding_levers) 
    aqp_flows = [aqp1_PP2_to_PP3, aqp2_PP3_to_Pozos, aqp3_Pozos_to_Toluquilla, aqp4_Toluquilla_to_PP1]
    aqp_flow_PP1 = aqp4_Toluquilla_to_PP1
    aqp_flow_PP2 = -aqp1_PP2_to_PP3
    aqp_flow_PP3 = (aqp1_PP2_to_PP3 - aqp2_PP3_to_Pozos )
    aqp_flow_Toluquilla = (aqp3_Pozos_to_Toluquilla - aqp4_Toluquilla_to_PP1)
    aqp_flow_Pozos = (aqp2_PP3_to_Pozos - aqp3_Pozos_to_Toluquilla)
    supplied_outputs = {
        "supplied_PP1": (potabilized_outputs["potabilized_PP1"] + additional_flows["additional_flow_PP1"] + aqp_flow_PP1) * (1 - loss_grid),
        "supplied_PP2": (potabilized_outputs["potabilized_PP2"] + aqp_flow_PP2) * (1 - loss_grid),
        "supplied_PP3": (potabilized_outputs["potabilized_PP3"] + additional_flows["additional_flow_PP3"] + aqp_flow_PP3) * (1 - loss_grid),
        "supplied_Toluquilla": (potabilized_outputs["potabilized_Toluquilla"] + additional_flows["additional_flow_Toluquilla"] + aqp_flow_Toluquilla) * (1 - loss_grid),
        "supplied_Pozos": (potabilized_outputs["potabilized_Pozos"] + additional_flows["additional_flow_Pozos"] + aqp_flow_Pozos) * (1 - loss_grid)
    }
    total_supplied = sum(supplied_outputs.values())
    supplied_outputs["total_supplied"] = total_supplied

    # Combine demand and supplied outputs
    model_outputs = {**demand_outputs, **supplied_outputs}

    # Calculate individual outcomes
    supplied_demand_outcomes = {f"supplied_demand_{ZA}": calculate_supplied_demand(model_outputs, ZA) for ZA in ZA_names}
    supplied_demand_deficit_outcomes = {f"supplied_demand_deficit_{ZA}": calculate_supplied_demand_deficit(supplied_demand_outcomes, ZA) for ZA in ZA_names}
    supply_percapita_outcomes = {f"supply_percapita_{ZA}": calculate_supply_percapita(model_outputs, population_dict, ZA) for ZA in ZA_names}
    individual_outcomes = {**supplied_demand_outcomes, **supplied_demand_deficit_outcomes, **supply_percapita_outcomes}

    # Calculate justice objectives
    min_supplied_demand = min(supplied_demand_outcomes.values())
    supplied_demand_average = np.average(list(supplied_demand_outcomes.values()))
    supply_percapita_average = np.average(list(supply_percapita_outcomes.values()))
    #average_supply_percapita = np.sum([supply_percapita_outcomes[f"supply_percapita_{ZA}"] * population_dict[f"population_{ZA}"] for ZA in ZA_names]) // np.sum(list(population_dict.values()))
    supply_percapita_GINI = calculate_GINI(supply_percapita_outcomes)
    supplied_demand_GINI = calculate_GINI(supplied_demand_outcomes)

    ZAs_below_threshold = {f"ZAs_below_{threshold}": len([x for x in supply_percapita_outcomes.values() if x < threshold]) for threshold in sufficientarian_thresholds}

    energy_cost_fraction = sum(abs(aqp_flow) for aqp_flow in aqp_flows) / 4

    aggregated_outcomes = {
        "supplied_demand_average": supplied_demand_average,
        "supply_percapita_average": supply_percapita_average,
        "min_supplied_demand": min_supplied_demand,
        "supply_percapita_GINI": supply_percapita_GINI,
        "supplied_demand_GINI": supplied_demand_GINI,
        "energy_costs": energy_cost_fraction,
        **ZAs_below_threshold
    }

    all_outcomes_dict= {**individual_outcomes, **aggregated_outcomes}

    '''for val in all_outcomes_dict.keys():
        all_outcomes_dict[val] = np.round(all_outcomes_dict[val], rounding_outcomes)'''

    all_model_outputs_dict = {**delivered_outputs, **model_outputs}
    
    '''for val in all_model_outputs_dict.keys():
        all_model_outputs_dict[val] = np.round(all_model_outputs_dict[val],3)'''

    return {**all_model_outputs_dict, **all_outcomes_dict} #, **{"scenario":scenario}}

def AMG_model_function_int(
    chapala_flow, calderon_lared_flow, pozos_flow, toluquilla_flow,
    aqp4_Toluquilla_to_PP1, aqp1_PP2_to_PP3, aqp2_PP3_to_Pozos, aqp3_Pozos_to_Toluquilla,
    crowding_factor=3.55, 
    chapalaPP1_to_chapalaPP2=0.19, loss_grid=0.35, loss_potabilisation=0.07,
    rounding_outcomes=3, rounding_levers=3, sufficientarian_thresholds=[142, 100, 50, 128],
    scenario="unspecified", experiment_name = "unspecified"
):

    ZA_names = ["PP1", "PP2", "PP3", "Toluquilla", "Pozos"]
    input_data_dict = pd.read_csv("data/input_data.csv").set_index('Variable').to_dict()['Value']

    # Calculate population and demand
    population_dict = {f"population_{ZA}": int(input_data_dict[f"domestic_intakes_{ZA}"] * crowding_factor) for ZA in ZA_names}
    demand_outputs = {
        f"demand_{ZA}": 
            ((population_dict[f"population_{ZA}"] * input_data_dict["domestic_consumption"] +
             input_data_dict[f"service_intakes_{ZA}"] * input_data_dict["service_consumption"] +
             input_data_dict[f"industry_intakes_{ZA}"] * input_data_dict["industry_consumption"] +
             input_data_dict[f"public_intakes_{ZA}"] * input_data_dict["public_consumption"]) * conversion_m3s )
             for ZA in ZA_names}
    #total_demand = sum(demand_outputs.values())

    # Calculate extraction and delivered water
    total_extraction = chapala_flow + calderon_lared_flow + toluquilla_flow + pozos_flow
    delivered_outputs = {
        "delivered_PP1": chapala_flow * (1 - chapalaPP1_to_chapalaPP2),
        "delivered_PP2": chapala_flow * chapalaPP1_to_chapalaPP2,
        "delivered_PP3": calderon_lared_flow,
        "delivered_Pozos": pozos_flow,
        "delivered_Toluquilla": toluquilla_flow
    }

    # Calculate potabilized water
    potabilized_outputs = {
        "potabilized_PP1": delivered_outputs["delivered_PP1"] * (1 - loss_potabilisation),
        "potabilized_PP2": delivered_outputs["delivered_PP2"] * (1 - loss_potabilisation),
        "potabilized_PP3": delivered_outputs["delivered_PP3"] * (1 - loss_potabilisation),
        "potabilized_Toluquilla": delivered_outputs["delivered_Toluquilla"] * (1 - loss_potabilisation),
        "potabilized_Pozos": delivered_outputs["delivered_Pozos"]  # no loss for Pozos
    }

    # Calculate additional flows and supplied water
    additional_flows = {
        "additional_flow_PP1": potabilized_outputs["potabilized_PP1"] * -0.12,
        "additional_flow_PP2": 0,
        "additional_flow_PP3": potabilized_outputs["potabilized_Pozos"] * 0.2,
        "additional_flow_Toluquilla": (potabilized_outputs["potabilized_PP1"] * 0.12 + potabilized_outputs["potabilized_Pozos"] * 0.02),
        "additional_flow_Pozos": potabilized_outputs["potabilized_Pozos"] * -0.22
    }
    scale = 10**(-rounding_levers)
    aqp_flows = [flow * scale for flow in [aqp1_PP2_to_PP3, aqp2_PP3_to_Pozos, aqp3_Pozos_to_Toluquilla, aqp4_Toluquilla_to_PP1]]
    aqp_flow_PP1 = aqp4_Toluquilla_to_PP1 * scale
    aqp_flow_PP2 = -aqp1_PP2_to_PP3 * scale
    aqp_flow_PP3 = (aqp1_PP2_to_PP3 - aqp2_PP3_to_Pozos ) * scale
    aqp_flow_Toluquilla = (aqp3_Pozos_to_Toluquilla - aqp4_Toluquilla_to_PP1) * scale
    aqp_flow_Pozos = (aqp2_PP3_to_Pozos - aqp3_Pozos_to_Toluquilla) * scale

    supplied_outputs = {
        "supplied_PP1": (potabilized_outputs["potabilized_PP1"] + additional_flows["additional_flow_PP1"] + aqp_flow_PP1) * (1 - loss_grid),
        "supplied_PP2": (potabilized_outputs["potabilized_PP2"] + aqp_flow_PP2) * (1 - loss_grid),
        "supplied_PP3": (potabilized_outputs["potabilized_PP3"] + additional_flows["additional_flow_PP3"] + aqp_flow_PP3) * (1 - loss_grid),
        "supplied_Toluquilla": (potabilized_outputs["potabilized_Toluquilla"] + additional_flows["additional_flow_Toluquilla"] + aqp_flow_Toluquilla) * (1 - loss_grid),
        "supplied_Pozos": (potabilized_outputs["potabilized_Pozos"] + additional_flows["additional_flow_Pozos"] + aqp_flow_Pozos) * (1 - loss_grid)
    }
    total_supplied = sum(supplied_outputs.values())
    supplied_outputs["total_supplied"] = total_supplied

    # Combine demand and supplied outputs
    model_outputs = {**demand_outputs, **supplied_outputs}

    # Calculate individual outcomes
    supplied_demand_outcomes = {f"supplied_demand_{ZA}": calculate_supplied_demand(model_outputs, ZA) for ZA in ZA_names}
    supplied_demand_deficit_outcomes = {f"supplied_demand_deficit_{ZA}": calculate_supplied_demand_deficit(supplied_demand_outcomes, ZA) for ZA in ZA_names}
    supply_percapita_outcomes = {f"supply_percapita_{ZA}": calculate_supply_percapita(model_outputs, population_dict, ZA) for ZA in ZA_names}
    individual_outcomes = {**supplied_demand_outcomes, **supplied_demand_deficit_outcomes, **supply_percapita_outcomes}

    # Calculate justice objectives
    min_supplied_demand = min(supplied_demand_outcomes.values())
    supplied_demand_average = np.average(list(supplied_demand_outcomes.values()))
    supply_percapita_average = np.average(list(supply_percapita_outcomes.values()))
    #average_supply_percapita = np.sum([supply_percapita_outcomes[f"supply_percapita_{ZA}"] * population_dict[f"population_{ZA}"] for ZA in ZA_names]) // np.sum(list(population_dict.values()))
    supply_percapita_GINI = calculate_GINI(supply_percapita_outcomes)
    supplied_demand_GINI = calculate_GINI(supplied_demand_outcomes)

    ZAs_below_threshold = {f"ZAs_below_{threshold}": len([x for x in supply_percapita_outcomes.values() if x < threshold]) for threshold in sufficientarian_thresholds}

    energy_cost_fraction = sum(abs(aqp_flow) for aqp_flow in aqp_flows) / 4

    aggregated_outcomes = {
        "supplied_demand_average": supplied_demand_average,
        "supply_percapita_average": supply_percapita_average,
        "min_supplied_demand": min_supplied_demand,
        "supply_percapita_GINI": supply_percapita_GINI,
        "supplied_demand_GINI": supplied_demand_GINI,
        "energy_costs": energy_cost_fraction,
        **ZAs_below_threshold
    }

    all_outcomes_dict= {**individual_outcomes, **aggregated_outcomes}

    '''for val in all_outcomes_dict.keys():
        all_outcomes_dict[val] = np.round(all_outcomes_dict[val], rounding_outcomes)'''

    all_model_outputs_dict = {**delivered_outputs, **model_outputs}
    
    '''for val in all_model_outputs_dict.keys():
        all_model_outputs_dict[val] = np.round(all_model_outputs_dict[val],3)'''

    return {**all_model_outputs_dict, **all_outcomes_dict} #, **{"scenario":scenario}}

