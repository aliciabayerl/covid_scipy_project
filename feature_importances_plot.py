import matplotlib.pyplot as plt
import pandas as pd

def subplot_feature_importances(data1, data2, title1, title2):

    data1 = data1.sort_values(by="Importance", ascending=True)
    data2 = data2.sort_values(by="Importance", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].barh(data1["Feature"], data1["Importance"], color="blue")
    axes[0].set_xlabel("Feature Importance")
    axes[0].set_ylabel("Features")
    axes[0].set_title(title1)

    axes[1].barh(data2["Feature"], data2["Importance"], color="blue")
    axes[1].set_xlabel("Feature Importance")
    axes[1].set_ylabel("Features")
    axes[1].set_title(title2)

    plt.tight_layout()
    plt.show()

data_importance = pd.read_csv('feature_importances.csv')
data_importance2 = pd.read_csv('feature_importances2.csv')

subplot_feature_importances(data_importance, data_importance2, "Risk Factors for Dying from covid (Medical Conditions)", "Risk Factors for Dying from covid (Symptoms)")

