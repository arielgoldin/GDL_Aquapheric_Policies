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
    crowding_factor=3.55, chapala_flow=6.9, calderon_lared_flow=1, pozos_flow=2.3, toluquilla_flow=0.5,
    chapalaPP1_to_chapalaPP2=0.19, loss_grid=0.35, loss_potabilisation=0.07,
    aqp4_Toluquilla_to_PP1=0, aqp1_PP2_to_PP3=0, aqp2_PP3_to_Pozos=0, aqp3_Pozos_to_Toluquilla=0,
    rounding_outcomes=3, rounding_levers=2, sufficientarian_thresholds=[142, 100, 50, 128]
):

    ZA_names = ["PP1", "PP2", "PP3", "Toluquilla", "Pozos"]
    input_data_dict = pd.read_csv("data/input_data.csv").set_index('Variable').to_dict()['Value']

    # Calculate population and demand
    population_dict = {f"population_{ZA}": int(input_data_dict[f"domestic_intakes_{ZA}"] * crowding_factor) for ZA in ZA_names}
    demand_outputs = {
        f"demand_{ZA}": np.round(
            (population_dict[f"population_{ZA}"] * input_data_dict["domestic_consumption"] +
             input_data_dict[f"service_intakes_{ZA}"] * input_data_dict["service_consumption"] +
             input_data_dict[f"industry_intakes_{ZA}"] * input_data_dict["industry_consumption"] +
             input_data_dict[f"public_intakes_{ZA}"] * input_data_dict["public_consumption"]) * conversion_m3s, 
            rounding_outcomes) 
        for ZA in ZA_names
    }
    total_demand = sum(demand_outputs.values())

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
    supplied_demand_outcomes = {f"supplied_demand_{ZA}": np.round(calculate_supplied_demand(model_outputs, ZA), rounding_outcomes) for ZA in ZA_names}
    supplied_demand_deficit_outcomes = {f"supplied_demand_deficit_{ZA}": calculate_supplied_demand_deficit(supplied_demand_outcomes, ZA) for ZA in ZA_names}
    supply_percapita_outcomes = {f"supply_percapita_{ZA}": np.round(calculate_supply_percapita(model_outputs, population_dict, ZA), 0) for ZA in ZA_names}
    individual_outcomes = {**supplied_demand_outcomes, **supplied_demand_deficit_outcomes, **supply_percapita_outcomes}

    # Calculate justice objectives
    min_supplied_demand = min(supplied_demand_outcomes.values())
    supplied_demand_average = np.average(list(supplied_demand_outcomes.values()))
    supply_percapita_average = np.average(list(supply_percapita_outcomes.values()))
    average_supply_percapita = np.sum([supply_percapita_outcomes[f"supply_percapita_{ZA}"] * population_dict[f"population_{ZA}"] for ZA in ZA_names]) // np.sum(list(population_dict.values()))
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
        "average_supply_percapita": average_supply_percapita,
        "energy_costs": energy_cost_fraction,
        **ZAs_below_threshold
    }

    return {**delivered_outputs, **model_outputs, **individual_outcomes, **aggregated_outcomes}




def calculate_energy_cost(flow_rates, lengths, elevation_changes, diameters, friction_factors, efficiencies):
    g = 9.81  # m/s²
    rho = 1000  # kg/m³

    total_power = 0

    for i in range(4):
        Q_i = flow_rates[i]  # m³/s
        L_i = lengths[i]  # m
        ΔH_i = elevation_changes[i]  # m
        D_i = diameters[i]  # m
        f_i = friction_factors[i]
        n_i = efficiencies[i]
        
        # Cross-sectional area
        A_i = np.pi * (D_i**2) / 4

        # Head loss due to friction (always positive)
        h_fi = f_i * (L_i / D_i) * (Q_i**2) / (2 * g * A_i**2)
        
        # Total head requirement (considering flow direction)
        H_i = h_fi + (ΔH_i if Q_i >= 0 else -ΔH_i)

        # Hydraulic power
        P_hi = rho * g * abs(Q_i) * H_i  # in Watts (J/s)

        # Electrical power
        P_ei = P_hi / n_i  # accounting for pump efficiency
        
        total_power += P_ei

    return total_power  # in Watts (J/s)


'''print({"Supplied demand SA1 =": supplied_demand_PP1,"Supplied demand SA2 =": supplied_demand_PP2,"Supplied demand SA3 =": supplied_demand_PP3,"Supplied demand SA4 =": supplied_demand_Toluquilla,"Supplied demand SA5 =": supplied_demand_Pozos}
)'''

'''              ,cost_loss_a = 500 #million per percentage
              ,cost_loss_b = 50 #million
              ,cost_reuse_a = 5000 #million per m3/day
              ,cost_reuse_b = 100 #million
              ,cost_rfc = 0.005 #million per m2
                            ,reduction_loss_grid = 0 
              ,reduction_loss_potabilisation = 0
                            ,rfh_houses_PP1 = 0, rfh_houses_PP2 = 0, rfh_houses_PP3 = 0, rfh_houses_Toluquilla = 0
                            ,water_reuse = 0'''

'''#1.1 Rainfall capture
conversion_m3s = 30.4 * 24 * 60 / 1000 #converts mm/month to m/s
rfh_PP1_flow = (rfh_proportion_PP1 * domestic_intakes_PP1) * average_household_area * harvest_coeficient * rainfall * conversion_m3s
rfh_PP2_flow = (rfh_proportion_PP2 * domestic_intakes_PP2) * average_household_area * harvest_coeficient * rainfall * conversion_m3s
rfh_PP3_flow = (rfh_proportion_PP3 * domestic_intakes_PP3) * average_household_area * harvest_coeficient * rainfall * conversion_m3s
rfh_Toluquilla_flow = (rfh_proportion_Toluquilla * domestic_intakes_Toluquilla) * average_household_area * harvest_coeficient * rainfall * conversion_m3s
rfh_Pozos_flow = (rfh_proportion_Pozos * domestic_intakes_Pozos) * average_household_area * harvest_coeficient * rainfall * conversion_m3s'''

