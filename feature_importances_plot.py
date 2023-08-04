import matplotlib.pyplot as plt
import pandas as pd

def subplot_feature_importances(data1, title1):

    data1 = data1.sort_values(by="Importance", ascending=True)

    plt.figure(figsize=(14, 6))

    plt.barh(data1["Feature"], data1["Importance"], color="blue")
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title(title1)

    plt.tight_layout()
    plt.show()

data_importance = pd.read_csv('feature_importances_casefile_hospitalized.csv')
data_importance2 = pd.read_csv('feature_importances2.csv')

subplot_feature_importances(data_importance, data_importance2, "Risk Factors for Dying from covid (Medical Conditions)", "Risk Factors for Dying from covid (Symptoms)")
