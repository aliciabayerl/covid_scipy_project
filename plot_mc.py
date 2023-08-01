import matplotlib.pyplot as plt
import pandas as pd


data_importance = pd.read_csv('feature_importances.csv')

data_importance = data_importance.sort_values(by="Importance", ascending=False)

# Plotting the bar chart
plt.figure(figsize=(10, 6))
plt.barh(data_importance["Feature"], data_importance["Importance"], color="blue")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Risk Factors for Dying from covid")
plt.show()
