"""
Created on 2024

@author: Ariel Goldin
Based on: ciullo, 2018
"""

from ema_workbench import (
    Model,
    CategoricalParameter,
    ArrayOutcome,
    ScalarOutcome,
    IntegerParameter,
    RealParameter,
)
from AMG_function import AMG_model  

import numpy as np

ZA_names = ["PP1", "PP2", "PP3", "Toluquilla", "Pozos"]

def supplied_demand (outcomes, ZA):
    supplied_demand_value = outcomes[0][f"supplied_{ZA}"]/outcomes[1][f"demand_{ZA}"]

    return supplied_demand_value




def get_model_for_problem_formulation(problem_formulation_id):
    """Convenience function to prepare AMG_model in a way it can be input in the EMA-workbench.
    Specify uncertainties, levers, and outcomes of interest.

    Parameters
    ----------
    problem_formulation_id : int {0, ..., 5}
                             problem formulations differ with respect to the objectives
                             0: Optimize only 5 supplied demand objectives
                             1: Obtimize

    Notes
    -----


    """
    # Load the model:
    function = AMG_model()
    # workbench model:
    AMG_model = Model("AMG_model", function=function)

    ZA_names = ["PP1", "PP2", "PP3", "Toluquilla", "Pozos"]
    model_vars = ["demand", "delivered", "potabilized", "supplied"]

    # 1.1 UNCERTAINTIES AND LEVERS


    #define a multiplier for all the water flows based on the average 2020 flow that represents drought and possible compensation by other sources
    low_flow=0 #due to meteorological drought or other reasons such as failure, all sources, except disagregated wells, could potentially drop to 0
    high_flow=1.3 #we consider the possibility that if a certain source underperforms others could compensate. Later it should be set to the maximum legal or hidraulical

    #Specify unceertainties
    AMG_model.uncertainties = {'chapala_flow',:[low_flow,6.3*high_flow],
                            RealParameter('calderon_flow',[low_flow,1*high_flow]),
                            RealParameter('zapotillo_flow',[low_flow,1*high_flow]),
                            RealParameter('pozos_flow',[low_flow,0.5*high_flow]),
                            RealParameter('toluquilla_flow',[low_flow,1*high_flow])}

    Real_uncert = {"Bmax": [30, 350], "pfail": [0, 1]}  # m and [.]
    # breach growth rate [m/day]
    cat_uncert_loc = {"Brate": (1.0, 1.5, 10)}

    '''system_uncert = {'private_intakes_1':[445436*lowrange,445436*highrange], RealParameter('private_intakes_2',125446*lowrange,125446*highrange), RealParameter('private_intakes_3',147032*lowrange,147032*highrange), RealParameter('private_intakes_4',128862*lowrange,128862*highrange), RealParameter('private_intakes_5',199613*lowrange,199613*highrange),
                     RealParameter('public_consumption',129*lowrange,129*highrange),
                     RealParameter('crowding_factor',3.55*lowrange,3.55*highrange),
                     RealParameter('node_division',0.35*lowrange,0.35*highrange),
                     RealParameter('loss_grid',0.3*lowrange,0.3*highrange), RealParameter('loss_potabilisation',0.05*lowrange,0.05*highrange),
                     RealParameter('rainfall',0,80*1.5), RealParameter('harvest_coeficient',0.75*lowrange,0.75*highrange), RealParameter('average_household_area',80*lowrange,80*highrange)}'''


    # load uncertainties and levers in AMG_model:
    AMG_model.uncertainties = uncertainties
    AMG_model.levers = levers

    # Problem formulations:
    # Outcomes are all costs, thus they have to minimized:
    direction = ScalarOutcome.MINIMIZE

    # 2-objective PF:
    if problem_formulation_id == 0:
        
        AMG_model.outcomes = []

        for ZA in ZA_names:
            AMG_model.outcomes.append(
                ScalarOutcome(f"Supplied_demand_"{ZA},
                              
                              )
            )
            ScalarOutcome()
        ]




       

    

    else:
        raise TypeError("unknown identifier")

    return AMG_model, function.planning_steps


if __name__ == "__main__":
    get_model_for_problem_formulation(3)
