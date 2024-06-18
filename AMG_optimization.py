import numpy as np
import pandas as pd
from platypus import EpsNSGAII, ProcessPoolEvaluator
from functions_data import full_dataframe, find_compromise
from ema_workbench import (Model, RealParameter, ScalarOutcome,
                           MultiprocessingEvaluator, ema_logging,
                           Scenario, Constraint, optimize)
from AMG_function import AMG_model_function
import random
import time


def run_optimization(experiment_name,
                     model_function=AMG_model_function, 
                     performance_outcomes = {"supplied_demand_deficit":True,"supplied_demand":False,"supply_percapita":False},
                     justice_outcomes = {"supplied_demand_GINI":False,"supply_percapita_GINI":True, "average_supply_percapita":False,
                                         "ZAs_below_142":False,"ZAs_below_128":False, "ZAs_below_100":False, "ZAs_below_50":False},
                     other_outcomes = {"energy_costs":True},
                     n_nfe = 10000, epsilon = 0.01, seed = 1,
                     scenario={"name":"2020_baseline",'chapala_flow' : 6.9, 'calderon_lared_flow' : 1, 'pozos_flow' : 2.3, 'toluquilla_flow' : 0.5},
                     restriction = True,
                     rounding_levers = 3):
    
    scenario = Scenario(**scenario)
    all_outcomes = {**performance_outcomes, **justice_outcomes, **other_outcomes}
    current_formulation = [outcome for outcome, active in all_outcomes.items() if active]


    #Setting the stage
    ema_logging.log_to_stderr(ema_logging.INFO)

    ZA_names = ["PP1", "PP2", "PP3", "Toluquilla", "Pozos"]

    #instantiate the model
    AMG_model = Model("AMGmodel", function = model_function)


    #LEVERS
    #Define a resolution to round levers and avoid finding solutions with unecesary precision considering uncertainties
    resolution = np.round(np.linspace(-1, 1, 10**rounding_levers, rounding_levers)).tolist()
    
    #Currently water can't flow between pozos and Toluquilla due to heigh difference. We call this the restriction.
    if restriction == True: 
        pozos_to_toluquilla = 0 
        resolution_pozos_to_toluquilla = np.round(np.linspace(-1, 0, 10**rounding_levers, rounding_levers)).tolist()
    else: 
        pozos_to_toluquilla = 1
        resolution_pozos_to_toluquilla = resolution


    AMG_model.levers = [RealParameter('aqp1_PP2_to_PP3',-1,1, resolution = resolution, pff= True),
                        RealParameter('aqp2_PP3_to_Pozos',-1,1, resolution = resolution, pff= True),
                        RealParameter('aqp3_Pozos_to_Toluquilla',-1,pozos_to_toluquilla, resolution_pozos_to_toluquilla),
                        RealParameter('aqp4_Toluquilla_to_PP1',-1,1, resolution = resolution, pff= False)]

    #OUTCOMES
    maximizing_outcomes = [*[f'supplied_demand_{ZA}' for ZA in ZA_names if "supplied_demand" in current_formulation],
                        *[f'supply_percapita_{ZA}' for ZA in ZA_names if "supply_percapita" in current_formulation],
                        *[ outcome for outcome in ["average_supply_percapita"] if outcome in current_formulation]]

    minimizing_outcomes = [*[f'supplied_demand_deficit_{ZA}' for ZA in ZA_names if "supplied_demand_deficit" in current_formulation],
                        *[ outcome for outcome in ['supplied_demand_GINI', 'supply_percapita_GINI', 'energy_costs', "ZAs_below_142", "ZAs_below_128","ZAs_below_100","ZAs_below_50"] if outcome in current_formulation]]

    info_outcomes = [*[f"supplied_{ZA}" for ZA in ZA_names]]

    AMG_model.outcomes = [ScalarOutcome(scalar_outcome, kind=ScalarOutcome.MAXIMIZE) for scalar_outcome in maximizing_outcomes] + [
        ScalarOutcome(minimizing_outcome, kind=ScalarOutcome.MINIMIZE) for minimizing_outcome in minimizing_outcomes] + [
        ScalarOutcome(info_outcome, kind=ScalarOutcome.INFO) for info_outcome in info_outcomes]


    #Optimization
    ema_logging.log_to_stderr(ema_logging.INFO)

    #calculate outcomes but substracting the info outcomes for the epsilons in the optimizaiton
    n_outcomes = len(AMG_model.outcomes)-len(info_outcomes)

    #Constrain
    non_negative_outcomes = [f"supplied_{ZA}" for ZA in ZA_names]
    constraints = [Constraint("non_negative_constrain", outcome_names= outcome, 
                            function=lambda x: max(0, -x)) for outcome in non_negative_outcomes]

    random.seed(seed)
    np.random.seed(seed)

    #start a timer
    start_time = time.time()

    with MultiprocessingEvaluator(AMG_model) as evaluator:
        results = evaluator.optimize(
            nfe=n_nfe, searchover="levers", epsilons=[epsilon] * n_outcomes, constraints = constraints, reference=scenario
        )

    # End timing
    end_time = time.time()

    # Compute the duration
    duration = round(end_time - start_time,0)
    
    
    results.head()
    return results, duration


'''    AMG_model.levers = [RealParameter('aqp1_PP2_to_PP3',-1,1, resolution=resolution, pff = True),
                        RealParameter('aqp2_PP3_to_Pozos',-1,1 , resolution=resolution, pff = True),
                        RealParameter('aqp3_Pozos_to_Toluquilla',-1,pozos_to_toluquilla, resolution=resolution_pozos_to_toluquilla, pff = True),
                        RealParameter('aqp4_Toluquilla_to_PP1',-1,1, resolution=resolution, pff = True)]'''