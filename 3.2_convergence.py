import os
import numpy as np
import pandas as pd
from platypus import EpsNSGAII, ProcessPoolEvaluator
from functions_data import full_dataframe, find_compromise
from ema_workbench import (Model, RealParameter, ScalarOutcome,
                           MultiprocessingEvaluator, ema_logging,
                           Scenario, Constraint, optimize)

from ema_workbench.em_framework.optimization import (ArchiveLogger, EpsilonProgress)
from ema_workbench import HypervolumeMetric
from ema_workbench.em_framework.optimization import to_problem

from AMG_function import AMG_model_function
import random
import time
import logging
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Problem formulation
    performance_outcomes = {"supplied_demand_deficit": True, "supplied_demand": False, "supply_percapita": False}
    justice_outcomes = {"supplied_demand_GINI": False, "supply_percapita_GINI": True, "supply_percapita_average": False,
                        "ZAs_below_142": False, "ZAs_below_128": False, "ZAs_below_100": False, "ZAs_below_50": False}
    other_outcomes = {"energy_costs": False}
    ZA_names = ["PP1", "PP2", "PP3", "Toluquilla", "Pozos"]

    restriction = True

    hypervolume_df = pd.DataFrame()

    # Setting up logging once
    ema_logging.log_to_stderr(ema_logging.INFO)

    # Read scenario file once
    representative_scenarios_df = pd.read_csv("results/representative_scenarios.csv", index_col=0)

    # for seed in [1]:

    seed = 12
    # Model Parameters
    n_nfe = 1000
    epsilon = 0.04

    # Scenario setting
    flows = ["chapala_flow", "calderon_lared_flow", "pozos_flow", "toluquilla_flow"]
    scenario_name = "2021_drought"
    scenario = representative_scenarios_df.loc[scenario_name, flows + ["name"]].to_dict()

    all_outcomes = {**performance_outcomes, **justice_outcomes, **other_outcomes}
    current_formulation = [outcome for outcome, active in all_outcomes.items() if active]

    experiment_name = "-".join(current_formulation) + f" {scenario['name']}-{n_nfe}-{epsilon}-{seed}"
    print(experiment_name)

    # Instantiate the model
    AMG_model = Model("AMGmodel", function=AMG_model_function)

    # LEVERS
    pozos_to_toluquilla = 0 if restriction else 1

    AMG_model.levers = [RealParameter('aqp1_PP2_to_PP3', -1, 1),
                        RealParameter('aqp2_PP3_to_Pozos', -1, 1),
                        RealParameter('aqp3_Pozos_to_Toluquilla', -1, pozos_to_toluquilla),
                        RealParameter('aqp4_Toluquilla_to_PP1', -1, 1)]

    # OUTCOMES
    maximizing_outcomes = [f'supplied_demand_{ZA}' for ZA in ZA_names if "supplied_demand" in current_formulation] + \
                          [f'supply_percapita_{ZA}' for ZA in ZA_names if "supply_percapita" in current_formulation]

    minimizing_outcomes = [f'supplied_demand_deficit_{ZA}' for ZA in ZA_names if "supplied_demand_deficit" in current_formulation] + \
                          [outcome for outcome in ['supplied_demand_GINI', 'supply_percapita_GINI', 'energy_costs',
                                                  "ZAs_below_142", "ZAs_below_128", "ZAs_below_100", "ZAs_below_50"]
                           if outcome in current_formulation]

    info_outcomes = [f"supplied_{ZA}" for ZA in ZA_names]

    AMG_model.outcomes = [ScalarOutcome(outcome, kind=ScalarOutcome.MAXIMIZE) for outcome in maximizing_outcomes] + \
                         [ScalarOutcome(outcome, kind=ScalarOutcome.MINIMIZE) for outcome in minimizing_outcomes] + \
                         [ScalarOutcome(outcome, kind=ScalarOutcome.INFO) for outcome in info_outcomes]

    # Optimization
    logging.info(f"Starting optimization for {experiment_name} with n_nfe={n_nfe}")

    n_outcomes = len(AMG_model.outcomes) - len(info_outcomes)

    # Constraints
    non_negative_outcomes = [f"supplied_{ZA}" for ZA in ZA_names]
    constraints = [Constraint("non_negative_constrain", outcome_names=outcome,
                              function=lambda x: max(0, -x)) for outcome in non_negative_outcomes]

    random.seed(seed)
    np.random.seed(seed)

    start_time = time.time()

    convergence_metrics = [
        ArchiveLogger(f"./archives", [lever.name for lever in AMG_model.levers],
                      [outcome.name for outcome in AMG_model.outcomes if outcome not in ['supplied_PP1', 'supplied_PP2', 'supplied_PP3', 'supplied_Toluquilla', 'supplied_Pozos']], base_filename="logger.tar.gz"),
        EpsilonProgress(),
    ]

    # Ensure the 'tmp' directory exists
    tmp_dir = "./archives/tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    with MultiprocessingEvaluator(AMG_model) as evaluator:
        results, convergence = evaluator.optimize(
            nfe=n_nfe, searchover="levers", epsilons=[epsilon] * n_outcomes, constraints=constraints, reference=Scenario(**scenario),
            convergence=convergence_metrics
        )

    end_time = time.time()
    duration = round(end_time - start_time, 0)
    results["experiment_name"] = experiment_name
    results["seed"] = seed
    results["scenario"] = scenario_name

    archives = ArchiveLogger.load_archives(f"archives/logger.tar.gz")

    # Removing some unnecessary columns from the reference set
    reference_set = results.drop(columns=["experiment_name", "seed", "scenario"])
    cols = reference_set.columns

    # Ensuring that the archives and the results reference dfs have the same cols
    for key in archives.keys():
        archives[key] = archives[key][cols]

    problem = to_problem(AMG_model, searchover="levers")

    hv = HypervolumeMetric(reference_set, problem)
    print(f"starting hv calculation {hv}")
    hypervolume = [(nfe, hv.calculate(archive)) for nfe, archive in archives.items()]
    print("hv calculation finished")
    hypervolume.sort(key=lambda x: x[0])
    hypervolume = pd.DataFrame(hypervolume)
    hypervolume["seed"] = seed

    hypervolume_df = pd.concat([hypervolume, hypervolume_df], axis=0)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 6))

    # Plot epsilon progress
    for seed in hypervolume_df['seed'].unique():
        data_seed = hypervolume_df[hypervolume_df['seed'] == seed]
        ax1.plot(data_seed[0], data_seed[1], label=f'Seed {seed}')

    ax1.set_title('Epsilon Progress')
    ax1.set_xlabel('Number of Evaluations')
    ax1.set_ylabel('Epsilon Progress')
    ax1.legend()

    # Plot hypervolume
    for seed in hypervolume_df['seed'].unique():
        data_seed = hypervolume_df[hypervolume_df['seed'] == seed]
        ax2.plot(data_seed[0], data_seed[1], label=f'Seed {seed}')

    ax2.set_title('Hypervolume')
    ax2.set_xlabel('Number of Evaluations')
    ax2.set_ylabel('Hypervolume')
    ax2.legend()

    plt.savefig("convergence_test.jpg", )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
