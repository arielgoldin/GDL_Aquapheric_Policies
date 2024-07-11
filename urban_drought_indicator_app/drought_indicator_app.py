from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import statsmodels.api as sm

app = Flask(__name__)

# Initialize dictionaries to store results and dataframes
log_reg_results = {}
dataframes = {}

# Fit logistic regression models for each threshold
for threshold in [50, 100, 142]:
    full_df = pd.read_csv(f"../1_drought_index\current_range\ZAs_below_{threshold}-1000scenarios-1000nfe epsilon0.8-restricted.csv")
    log_reg_df = full_df

    variable_of_interest = f"ZAs_below_{threshold}"
    log_reg_df["scenario_of_interest"] = log_reg_df[variable_of_interest] > 0

    log_reg_df = log_reg_df[['chapala_flow', 'calderon_lared_flow', 'pozos_flow', 'toluquilla_flow', "scenario_of_interest"]]
    
    dataframes[threshold] = log_reg_df

    log_reg_df["intercept"] = np.ones(np.shape(log_reg_df)[0])
    predictors = ['chapala_flow', 'calderon_lared_flow', 'pozos_flow', 'toluquilla_flow', 'intercept']

    logit = sm.Logit(log_reg_df["scenario_of_interest"], log_reg_df[predictors])
    log_reg_results[threshold] = logit.fit()

# Prediction function
def predict_scenario(scenario, threshold):
    scenario['intercept'] = 1
    scenario_array = np.array([scenario[col] for col in predictors])
    log_odds = np.dot(log_reg_results[threshold].params, scenario_array)
    probability = 1 / (1 + np.exp(-log_odds))
    return probability

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        chapala_flow = float(request.form['chapala_flow'])
        calderon_lared_flow = float(request.form['calderon_lared_flow'])
        pozos_flow = float(request.form['pozos_flow'])
        toluquilla_flow = float(request.form['toluquilla_flow'])
        
        new_scenario = {
            'chapala_flow': chapala_flow,
            'calderon_lared_flow': calderon_lared_flow,
            'pozos_flow': pozos_flow,
            'toluquilla_flow': toluquilla_flow,
            'intercept': 1
        }
        
        probability_threshold = 0.05
        sufficientarian_thresholds = [50, 100, 142]
        results = []

        for threshold in sufficientarian_thresholds:
            likelihood = predict_scenario(new_scenario, threshold)
            results.append((threshold, likelihood))

        drought_state = "NO drought"
        for threshold, likelihood in results:
            if likelihood >= probability_threshold:
                if threshold == 50: 
                    drought_state = "HIGH drought"
                    break
                elif threshold == 100: 
                    drought_state = "MID drought"
                    break
                elif threshold == 142:
                    drought_state = "LOW drought"
                    break

        return render_template('index.html', results=results, drought_state=drought_state, scenario=new_scenario)
    return render_template('index.html', scenario=None)

if __name__ == '__main__':
    app.run(debug=True)
