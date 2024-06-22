import statsmodels.api as sm
import numpy as np
import pandas as pd

# Dictionary to store logistic regression results for different thresholds
log_reg_results = {}
dataframes = {}
sufficientarian_thresholds = [50, 100, 142]

# Load data and fit logistic regression models for each threshold
def initialize_models():
    global log_reg_results, dataframes
    for threshold in sufficientarian_thresholds:
        full_df = pd.read_csv(f"1_drought_index/current_range/ZAs_below_{threshold}-1000scenarios-1000nfe epsilon0.8-restricted.csv")
        log_reg_df = full_df
        
        variable_of_interest = f"ZAs_below_{threshold}"
        log_reg_df["scenario_of_interest"] = log_reg_df[variable_of_interest] > 0

        log_reg_df = log_reg_df[['chapala_flow', 'calderon_lared_flow', 'pozos_flow', 'toluquilla_flow', "scenario_of_interest"]]
        dataframes[threshold] = log_reg_df

        log_reg_df.loc[:,"intercept"] = np.ones(np.shape(log_reg_df)[0])
        predictors = ['chapala_flow', 'calderon_lared_flow', 'pozos_flow', 'toluquilla_flow', 'intercept']

        logit = sm.Logit(log_reg_df["scenario_of_interest"], log_reg_df[predictors])
        log_reg_results[threshold] = logit.fit(disp=0)

# Function to predict the likelihood of falling below a given threshold
def predict_scenario(scenario, threshold):
    scenario['intercept'] = 1
    predictors = ['chapala_flow', 'calderon_lared_flow', 'pozos_flow', 'toluquilla_flow', 'intercept']
    scenario_array = np.array([scenario[col] for col in predictors])
    log_odds = np.dot(log_reg_results[threshold].params, scenario_array)
    probability = 1 / (1 + np.exp(-log_odds))
    return probability

# Function to determine the drought state and likelihood based on flow values
def get_drought_state(scenario, reliability_threshold=0.05):
    for threshold in sufficientarian_thresholds:
        likelihood = round(predict_scenario(scenario, threshold),3)
        
        if likelihood>= reliability_threshold:
            if threshold == sufficientarian_thresholds[0]: 
                drought_state = "HIGH"
                break
            elif threshold == sufficientarian_thresholds[1]: 
                drought_state = "MID"
                break
            elif threshold == sufficientarian_thresholds[2]:
                
                drought_state = "LOW"
                break
        if threshold == sufficientarian_thresholds[2]: drought_state = "NO"

    return drought_state
        

# Initialize the models on import
initialize_models()


