# Master THesis Project - Engineering and Policy Analysis at TU Delft
## Ariel Goldin Marcovich
### SUpervisors: Jazmin Zatarain, Neelke Doorn, Edmundo Molina Pérez

### 1. Drought Index
Three notebooks to design the assumption based urban dorught indicator
#### 1.1 Drought Data generator
Generator of space filling database of possible flow values for the four sources.
#### 1.2 Scenario exploration
An analysis based on Logistic-regression scenario discovery to separate the risky and safe drought states.
#### 1.3 Drought index
An algorithm to calculate the current drought state using the logistic regresion.

### 2. Optimization formulations
Notebook to compaire the capacity to provide global maximas or best perferming policies of three different problem formulations for three representative drought scenarios.

### 3. Convergence
An assessment of the convergence of the optimziation algorithm to select reliable hyperparameters

### 4. Decision support
An exploration of the decision support framework that can generate any optimization and visualize policies for any combination of objectives.

** Abstract **
Due to worsening drought and other socio-environmental challenges, the city of Guadalajara, Mexico, is facing increasing challenges in supplying enough water to its five million inhabitants. The city’s water supply system is compartmentalized, meaning that each one of its four main sources supplies a specific area of the city. Thus, if one source underperforms, one area of the city will not have enough water and the other sources cannot compensate. This situation happened in 2021 when one of its sources reached critical levels leaving around 500,000 inhabitants without water supply for over 3 months. To combat this vulnerability the city built the Aquapheric: a circular aqueduct that interconnects the supply areas and can pump up to 1m3/s in both directions in each segment independently. However, the government has not developed a distribution policy for this infrastructure. 
This project proposes a participatory and simulation-based process for designing a Decision Support Tool that could serve as the basis for a distribution policy that reflects the city’s interpretation of justice using Multi-Objective Optimization methods. To identify and specify justice objectives, a participatory workshop was conducted with over 30 members of the local government and academia. Different objectives, restrictions and assumptions were identified that were used to conduct an exploratory analysis. This analysis enabled the identification of optimization settings that enable a standardized unsupervised optimization that is able to find optimal policies for a wide combination of objectives for any drought scenario. The final deliverable is an interactive tool that runs an optimization for the current drought conditions and enables operators or policymakers to select a policy based on its performance in relation to the selected combination of objectives and an assumption-based socio-economic drought intensity indicator.
The theoretical discussions in this research focus on how deep uncertainty in decision-making can be tackled by offering tools based on simple models, built with participatory processes and that enable learning and flexibility as opposed to robustness. 


