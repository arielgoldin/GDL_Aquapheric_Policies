# GDL_Aquapheric_Policies
 Master THesis Project

Formulation 1. No uncertainties except flows
1. Define ranges of the flow under drought based on the average 2020 flow. Currently set to 0 and 150% of the average flow for all sources. Later it could be defined for each source based on capacity or legal limits.
2. Define the supplied demand ratio of each supply area + GINI as objectives
3. Create Original RBFs and set them as levers (inputs: water flows5, outputs: AqP flows4)
4. Filter the scenarios based on a drought condition (one or more sources have to be under 80%, the total flow cannot be larger than total demand).
5. Conduct an optimization (n policies, x scenarios (water flows))
6. This will produce a dataframe with n policies that generate n sets of objectives for each scenario.
7. Calculate an indicator of performance for each policy and each objective across all the scenarios. (average for now)
8. Visualize

Unresolved issues
How to apply a filter to the experiments?
How to avoid negative flows in the operation policy?
