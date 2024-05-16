import math
import numpy as np
import pandas as pd
#import rbf_functions
from platypus import EpsNSGAII, ProcessPoolEvaluator
#import rbf_functions

'''import sysdomestic_intakes_PP1

supplied_areas=pd.read_csv("data/supplied_areas.csv")
'''
conversion_m3s = 1/(24*60*60*1000) #converts l/day into m3/s

def demand (population, ZA, domestic_consumption):
      
      demand_value = population[f"population_{ZA}"] * domestic_consumption * conversion_m3s

      return demand_value


def supplied_demand (outcomes, ZA):
    supplied_demand_value = outcomes[f"supplied_{ZA}"] / outcomes[f"demand_{ZA}"]

    return supplied_demand_value


def supply_percapita (outcomes, population, ZA):
    supply_percapita_value = outcomes[f"supplied_{ZA}"] / population[f"population_{ZA}"] / conversion_m3s

    return supply_percapita_value


def supplied_demand_deficit (outcomes, ZA):
      supplied_demand_deficit_value = abs(1-outcomes[f"supplied_demand_{ZA}"])

      return supplied_demand_deficit_value


def sq_deficit (outcomes, ZA):
      sq_deficit_value = (outcomes[f"supplied_{ZA}"] - outcomes[f"demand_{ZA}"])**2

      return sq_deficit_value
 

def AMG_model_function(crowding_factor = 3.55 #people per household
                       ,chapala_flow = 6.9, calderon_flow = 1, zapotillo_flow = 1, pozos_flow = 2.3, toluquilla_flow = 0.5
                       ,chapalaPP1_to_chapalaPP2 = 0.19 #Proportion of water that is sent from the supplied line of PP1 to PP2
                       ,loss_grid = 0.35 # in %
                       ,loss_potabilisation = 0.07 # losses in potabilization and transport in %
                       ,aqp4_Toluquilla_to_PP1 = 0, aqp1_PP2_to_PP3 = 0, aqp2_PP3_to_Pozos = 0, aqp3_Pozos_to_Toluquilla = 0
                       ,rounding_outcomes = 5, rounding_levers = 2,
                       equity_indicator="supplied_demand", equity_calc = "GINI"):
        

        ZA_names = ["PP1", "PP2", "PP3", "Toluquilla", "Pozos"]
        model_vars = ["demand", "delivered", "potabilized", "supplied"]

        input_data_dict = pd.read_csv("data/input_data.csv").set_index('Variable').to_dict()['Value']

        
        #1. DEMAND
        conversion_m3s = 1/(24*60*60*1000) #converts l/day into m3/s
        

        population_dict = {f"population_{ZA}":int(input_data_dict[f"domestic_intakes_{ZA}"]*crowding_factor) for ZA in ZA_names}
        
        demand_outputs = {f"demand_{ZA}":np.round((population_dict[f"population_{ZA}"] * input_data_dict["domestic_consumption"] + 
                                                  input_data_dict[f"service_intakes_{ZA}"] * input_data_dict["service_consumption"] + 
                                                  input_data_dict[f"industry_intakes_{ZA}"] * input_data_dict["industry_consumption"] + 
                                                  input_data_dict[f"public_intakes_{ZA}"] * input_data_dict["public_consumption"])
                                                  
                                                  *conversion_m3s, rounding_outcomes) 
                                          for ZA in ZA_names}
                                          


        #1.1 Household demand
        #demand_outputs = {f"demand_{ZA}":np.round(demand(population_dict, ZA, domestic_consumption),rounding_outcomes) for ZA in ZA_names}

        total_demand = sum(demand_outputs.values())
        

        #2. EXTRACTION
        total_extraction = chapala_flow + calderon_flow + toluquilla_flow + pozos_flow

        #DELIVERED - water delivered to each potabilization plant
        delivered_PP1 = chapala_flow * (1-chapalaPP1_to_chapalaPP2) #PP1 and PP2 both receive water from Chapala and distribute it
        delivered_PP2 = chapala_flow * chapalaPP1_to_chapalaPP2
        delivered_PP3 = calderon_flow + zapotillo_flow #Calderon and Chapala flow towards the PP3
        delivered_Pozos = pozos_flow
        delivered_Toluquilla = toluquilla_flow

        delivered_outputs = {"delivered_PP1": delivered_PP1,"delivered_PP2": delivered_PP2,"delivered_PP3": delivered_PP3,"delivered_Toluquilla": delivered_Toluquilla,"delivered_Pozos": delivered_Pozos}
      
        
        #2. Water potabilized in each supplied area 
        potabilized_PP1 = (delivered_PP1 * (1-loss_potabilisation))
        potabilized_PP2 = (delivered_PP2 * (1-loss_potabilisation))
        potabilized_PP3 = (delivered_PP3 * (1-loss_potabilisation))
        potabilized_Toluquilla = (delivered_Toluquilla * (1-loss_potabilisation))
        potabilized_Pozos = delivered_Pozos #the flow from the wells doesn't present any loss as it is only chlorinated (not potabilized) before entering the grid

        potabilized_outputs = {"potabilized_PP1": potabilized_PP1,"potabilized_PP2": potabilized_PP2,"potabilized_PP3": potabilized_PP3,"potabilized_Toluquilla": potabilized_Toluquilla,"potabilized_Pozos": potabilized_Pozos}
        
        #ADDITIONAL FLOWS
        additional_flow_PP1 = potabilized_PP1 * -0.12
        additional_flow_PP2 = 0
        additional_flow_PP3 = potabilized_Pozos * 0.2
        additional_flow_Toluquilla = (potabilized_PP1 * 0.12 + potabilized_Pozos * 0.02)
        additional_flow_Pozos = potabilized_Pozos * -0.22

        aqp_flow_PP1 = aqp4_Toluquilla_to_PP1
        aqp_flow_PP2 = - aqp1_PP2_to_PP3
        aqp_flow_PP3 = (aqp1_PP2_to_PP3 - aqp2_PP3_to_Pozos)
        aqp_flow_Toluquilla =  aqp3_Pozos_to_Toluquilla - aqp4_Toluquilla_to_PP1
        aqp_flow_Pozos = (aqp2_PP3_to_Pozos - aqp3_Pozos_to_Toluquilla)

        additional_flow_outputs = {"additional_flow_PP1": additional_flow_PP1,"additional_flow_PP2": additional_flow_PP2,"additional_flow_PP3": additional_flow_PP3,"additional_flow_Toluquilla": additional_flow_Toluquilla,"additional_flow_Pozos": additional_flow_Pozos}
        
        #SUPPLIED to users in each area - losses are considered after the aquapheric flows
        supplied_PP1 = (potabilized_PP1 + additional_flow_PP1 + aqp_flow_PP1) * (1-loss_grid)
        supplied_PP2 = (potabilized_PP2 + additional_flow_PP2 + aqp_flow_PP2) * (1-loss_grid)
        supplied_PP3 = (potabilized_PP3 + additional_flow_PP3 + aqp_flow_PP3) * (1-loss_grid)
        supplied_Toluquilla = (potabilized_Toluquilla + additional_flow_Toluquilla + aqp_flow_Toluquilla) * (1-loss_grid)
        supplied_Pozos = (potabilized_Pozos + additional_flow_Pozos + aqp_flow_Pozos) * (1-loss_grid)

        supplied_outputs = {"supplied_PP1": supplied_PP1,"supplied_PP2": supplied_PP2,"supplied_PP3": supplied_PP3,"supplied_Toluquilla": supplied_Toluquilla,"supplied_Pozos": supplied_Pozos}

        total_supplied = supplied_PP1 + supplied_PP2 + supplied_PP3 + supplied_Toluquilla + supplied_Pozos

        model_outputs = {**supplied_outputs, **demand_outputs, **delivered_outputs} #is this necesary?


        #3. OBJECTIVE FUNCITONS

        #3.1 Individual
        
        #3.1.1 Supplied demand
        supplied_demand_outcomes = {f"supplied_demand_{ZA}":np.round(supplied_demand(model_outputs,ZA),rounding_outcomes) for ZA in ZA_names}

        supplied_demand_deficit_outcomes = {f"supplied_demand_deficit_{ZA}":supplied_demand_deficit(supplied_demand_outcomes,ZA) for ZA in ZA_names}

        #3.1.2 Demand per ca´pita
        supply_percapita_outcomes = {f"supply_percapita_{ZA}":np.round(supply_percapita(model_outputs,population_dict, ZA),0) for ZA in ZA_names}

        #3.1.3 Deficit sq
        deficit_sq_outcomes = {f"deficit_sq_{ZA}":np.round(sq_deficit(model_outputs,ZA),rounding_outcomes) for ZA in ZA_names}



        individual_outcomes = {**supplied_demand_outcomes, **supplied_demand_deficit_outcomes, **supply_percapita_outcomes}

        #3.2 Agreggated
        min_supplied_demand = min(supplied_demand_outcomes.values())
        min_deficit_sq = min(deficit_sq_outcomes.values())

        #3.2.2 GINI
        if equity_calc == "GINI":
              if equity_indicator == "supply_percapita": objectives = list(supply_percapita_outcomes.values())
              elif equity_indicator == "supplied_demand": objectives = list(supplied_demand_outcomes.values())
              n = len(objectives)
              sorted_objectives = np.sort(objectives)
              diffs = np.abs(np.subtract.outer(sorted_objectives, sorted_objectives)).flatten()
              # 1 - ... needed so that all principles have a maximization direction
              equity_result = (np.sum(diffs) / (2.0 * n * np.sum(sorted_objectives)))



        aggregated_outcomes = {"min_supplied_demand":min_supplied_demand, "min_deficit_sq":min_deficit_sq, f"{equity_indicator}_{equity_calc}":equity_result}

        
      
        return {**delivered_outputs, **model_outputs, **individual_outcomes, **aggregated_outcomes}





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