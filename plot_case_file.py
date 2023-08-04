import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Daten einlesen
df_plot = pd.read_csv('casefile_breinigt.csv', sep=',')
df_plot = df_plot.drop(columns='Unnamed: 0')
titles=["Age of participants", "Sex of participants", "Number of exposed participants", "Number of participants that are hospitalized", "Number of participants that were treated in ICU", "Number of deceased participants", "Number of participants with existing underlying conditions"]
labeling=[['18-44', '<18', '45-64', '65+'], ['male', 'female'], ['exposed', 'not exposed'], ['hospitalized', 'not hospitalized'], ['ICU-treatment', 'no ICU-treatment'], ['deceased', 'not deceased'], ['existing conditions', 'no existing conditions']]

def sympt_plotten2(df2):
    plt.figure(figsize=(16, 12)).tight_layout()
    plt.subplots_adjust(hspace=0.8, wspace=0.5)
    for i in range(df2.shape[1]):
        if i ==0:
            values = [np.count_nonzero(df2.iloc[:, i] == 1), np.count_nonzero(df2.iloc[:, i] == 0), np.count_nonzero(df2.iloc[:,i]==2), np.count_nonzero(df2.iloc[:,i]==3)]  #
        else:
            values= [np.count_nonzero(df2.iloc[:,i]==1), np.count_nonzero(df2.iloc[:,i]==0)] #, np.count_nonzero(df2.iloc[:,i]==2), np.count_nonzero(df2.iloc[:,i]==3)
        labels=labeling[i]
        X_axis = np.arange(len(labels))
        plt.subplot(3,5,i+1)
        plt.bar(X_axis, values)
        plt.xticks(X_axis, labels, fontsize=5)
        plt.ylabel("Number of samples", fontsize=7)
        plt.title(titles[i], fontsize=7)
        print(i)
    #plt.show()
sympt_plotten2(df_plot)
plt.show()