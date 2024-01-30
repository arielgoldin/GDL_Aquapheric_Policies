import math
import numpy as np
import pandas as pd

'''import sysprivate_intakes_1

supply_areas=pd.read_csv("data/supply_areas.csv")
'''


def AMG_model(private_intakes_1=445436, private_intakes_2=125446, private_intakes_3=147032, private_intakes_4=128862, private_intakes_5=199613
              ,public_consumption=129 #l/day/inhabitant
              ,crowding_factor = 3.5 #people per household
              ,chapala_flow = 6.9, calderon_flow = 1, zapotillo_flow = 1, pozos_flow = 2.3, toluquilla_flow = 0.5
              ,node_division = 0.35
              ,loss_grid = 0.35 # in %
              ,loss_potabilisation = 0.07 # losses in potabilization and transport in %
              , mfc_flow_1 = 0, mfc_flow_2 = 0, mfc_flow_3 = 0, mfc_flow_4 = 0, mfc_flow_5 = 0
              ,rainfall = 80 #mm/month
              ,harvest_coeficient = 0.75 #proportion of water that falls on rooftop that can be harvested
              ,average_household_area = 80 #m2 
              ,rfh_proportion_1 = 0.1, rfh_proportion_2 = 0.3, rfh_proportion_3 = 1.1, rfh_proportion_4 = 0.3, rfh_proportion_5 = 0.1 #proportion of the houses that have rfc /100 
              ):
        
        #1. DEMAND

        #1.1 Household demand
        conversion_m3s = 1/(24*60*60*1000) #converts l/day into m3/s
        demand1 = private_intakes_1 * crowding_factor * public_consumption * conversion_m3s
        demand2 = private_intakes_2 * crowding_factor * public_consumption * conversion_m3s
        demand3 = private_intakes_3 * crowding_factor * public_consumption * conversion_m3s
        demand4 = private_intakes_4 * crowding_factor * public_consumption * conversion_m3s
        demand5 = private_intakes_5 * crowding_factor * public_consumption * conversion_m3s
        total_demand = demand1 + demand2 + demand3 + demand4 + demand5
      
        
        #2. SUPPLY
        supply_2 = (chapala_flow * node_division * (1-loss_potabilisation) - mfc_flow_1) * (1-loss_grid)
        supply_3 = ((calderon_flow + zapotillo_flow ) * (1-loss_potabilisation) + mfc_flow_1 - mfc_flow_2) * (1- loss_grid) 
        supply_5 = (pozos_flow  + mfc_flow_2 - mfc_flow_3 ) * (1-loss_grid)
        supply_4 = (toluquilla_flow * (1-loss_potabilisation) + mfc_flow_3 - mfc_flow_4) * (1-loss_grid)
        supply_1 = (chapala_flow * (1-node_division) * (1-loss_potabilisation) + mfc_flow_4) * (1-loss_grid)
        total_supply = supply_1 + supply_2 + supply_3 + supply_4 + supply_5


        #3. OBJECTIVE FUNCITONS
        supplied_demand_1 = supply_1/demand1
        supplied_demand_2 = supply_2/demand2
        supplied_demand_3 = supply_3/demand3
        supplied_demand_4 = supply_4/demand4
        supplied_demand_5 = supply_5/demand5

        #dumb objective func because numpy gives error here
        min_supplied_demand = supplied_demand_4
        
        #can't get numpy to work here
        #min_supplied_demand = np.min([supplied_demand_1,supplied_demand_2,supplied_demand_3, supplied_demand_4, supplied_demand_5])



        return {"Total demand":total_demand, "Total supply": total_supply, "supplied_demand_1": supplied_demand_1,"supplied_demand_2": supplied_demand_2,
                "supplied_demand_3": supplied_demand_3,"supplied_demand_4": supplied_demand_4,"supplied_demand_5": supplied_demand_5
                "min_supplied_demand":min_supplied_demand}



def get_MFC_flows():
        return demand1, demand2, demand3, demand4, demand5, mfc_flow_1, mfc_flow_2, mfc_flow_3, mfc_flow_4, mfc_flow_5





'''print({"Supplied demand SA1 =": supplied_demand_1,"Supplied demand SA2 =": supplied_demand_2,"Supplied demand SA3 =": supplied_demand_3,"Supplied demand SA4 =": supplied_demand_4,"Supplied demand SA5 =": supplied_demand_5}
)'''

'''              ,cost_loss_a = 500 #million per percentage
              ,cost_loss_b = 50 #million
              ,cost_reuse_a = 5000 #million per m3/day
              ,cost_reuse_b = 100 #million
              ,cost_rfc = 0.005 #million per m2
                            ,reduction_loss_grid = 0 
              ,reduction_loss_potabilisation = 0
                            ,rfh_houses_1 = 0, rfh_houses_2 = 0, rfh_houses_3 = 0, rfh_houses_4 = 0
                            ,water_reuse = 0'''

'''#1.1 Rainfall capture
conversion_m3s = 30.4 * 24 * 60 / 1000 #converts mm/month to m/s
rfh_1_flow = (rfh_proportion_1 * private_intakes_1) * average_household_area * harvest_coeficient * rainfall * conversion_m3s
rfh_2_flow = (rfh_proportion_2 * private_intakes_2) * average_household_area * harvest_coeficient * rainfall * conversion_m3s
rfh_3_flow = (rfh_proportion_3 * private_intakes_3) * average_household_area * harvest_coeficient * rainfall * conversion_m3s
rfh_4_flow = (rfh_proportion_4 * private_intakes_4) * average_household_area * harvest_coeficient * rainfall * conversion_m3s
rfh_5_flow = (rfh_proportion_5 * private_intakes_5) * average_household_area * harvest_coeficient * rainfall * conversion_m3s'''