import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('MC_cleaned_dataset.csv')

grouped_data = df.groupby('age_categories')['deceased'].mean()

plt.figure(figsize=(8, 6))
grouped_data.plot(kind='bar', color='skyblue')
plt.xlabel('Age Category')
plt.ylabel('Proportion of Deceased')
plt.title('Proportion of Deceased Cases in each Age Group')
plt.xticks(rotation=0)
plt.show()
