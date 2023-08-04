## import of needed libraries
import matplotlib.pyplot as plt
import pandas as pd

## the following function shall enable to plot the feature importance for each prediction
def subplot_feature_importances(data1, title1, saveplot):
    # sort data by importance
    data1 = data1.sort_values(by="Importance", ascending=True)
    # start plotting a barchart with importance of each predictor
    plt.figure(figsize=(7, 6))
    plt.barh(data1["Feature"], data1["Importance"], color="blue")
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title(title1)
    # display and save plot
    plt.tight_layout()
    plt.savefig(saveplot)

## Load data for plotting
data_importance1 = pd.read_csv('feature_importances.csv')
data_importance2 = pd.read_csv('sy_feature_importances.csv')
data_importance3 = pd.read_csv('feature_importances_cf_hospitalized.csv')
data_importance4 = pd.read_csv('feature_importances_cf_icu.csv')
data_importance5 = pd.read_csv('feature_importances_cf_dead.csv')

## Plot data for different predictors
subplot_feature_importances(data_importance1,  "Risk Factors for dying due to covid (Medical Conditions)", 'plt_feature_importance_mc')
subplot_feature_importances(data_importance2, "Risk Factors for dying due to covid (Symptoms)", 'plt_feature_importance_sy')
subplot_feature_importances(data_importance3,  "Risk Factors for being hospitalized due to covid (casefile)", 'plt_feature_importance_cf_hosp')
subplot_feature_importances(data_importance4, "Risk Factors for being in ICU due to covid (casefile)", 'plt_feature_importance_cf_icu')
subplot_feature_importances(data_importance5, "Risk Factors for dying due to covid (casefile)", 'plt_feature_importance_cf_dead')
