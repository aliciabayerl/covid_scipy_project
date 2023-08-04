import pandas as pd
import matplotlib.pyplot as plt
from randomForest import random_forest


data = pd.read_csv('mc_cleaned_dataset.csv')

X = data.drop(columns=['deceased'])
y = data['deceased']

random_forest_model, feature_importances, _, _, _ = random_forest(X, y)

# Select the three features to analyze
selected_features = ['age_categories', 'anemia_confirmed', 'bmi_cat']

# Plot the number of deceased individuals for each category of the three selected features based on feature importances
plt.figure(figsize=(14, 6))

for i, feature in enumerate(selected_features):
    plt.subplot(1, 3, i + 1)
    data.groupby([feature, 'deceased']).size().unstack().plot(kind='bar', stacked=True, ax=plt.gca())
    plt.xlabel(feature)
    plt.ylabel('Number of Deceased')
    plt.title(f'Number of Deceased by {feature}')

plt.tight_layout()
plt.show()