import matplotlib.pyplot as plt
import pandas as pd
import os


# Function to plot the feature importance for each prediction
def subplot_feature_importances(data1, title1, saveplot):

    # Sort data by importance
    data1 = data1.sort_values(by="Importance", ascending=True)

    # Start plotting a barchart with importance of each predictor
    plt.figure(figsize=(7, 6))
    plt.barh(data1["Feature"], data1["Importance"], color="blue")
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title(title1)

    plt.tight_layout()
    plot_path = os.path.join('Plots', saveplot)
    plt.savefig(plot_path)

# Load and save plots in corresponding folder

fi_folder = 'FeatureImportances'

file_paths = [
    os.path.join(fi_folder, 'feature_importances_mc.csv'),
    os.path.join(fi_folder, 'feature_importances_sy.csv'),
    os.path.join(fi_folder, 'feature_importances_cf_hospitalized.csv'),
    os.path.join(fi_folder, 'feature_importances_cf_icu.csv'),
    os.path.join(fi_folder, 'feature_importances_cf_dead.csv')
]

data_importance_list = []
for file_path in file_paths:
    data_importance_list.append(pd.read_csv(file_path))

data_importance1 = data_importance_list[0]
data_importance2 = data_importance_list[1]
data_importance3 = data_importance_list[2]
data_importance4 = data_importance_list[3]
data_importance5 = data_importance_list[4]

subplot_feature_importances(data_importance1, "Risk Factors for dying due to covid (Medical Conditions)", 'plt_feature_importance_mc')
subplot_feature_importances(data_importance2, "Risk Factors for dying due to covid (Symptoms)", 'plt_feature_importance_sy')
subplot_feature_importances(data_importance3, "Risk Factors for being hospitalized due to covid (casefile)", 'plt_feature_importance_cf_hosp')
subplot_feature_importances(data_importance4, "Risk Factors for being in ICU due to covid (casefile)", 'plt_feature_importance_cf_icu')
subplot_feature_importances(data_importance5, "Risk Factors for dying due to covid (casefile)", 'plt_feature_importance_cf_dead')