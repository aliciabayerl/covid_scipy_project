


## Visualsierung der Daten

## Bibliothekten einladen
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Daten einlesen
df = pd.read_csv('Symptome_breinigt.csv', sep=',')
df=df.drop(columns=['Unnamed: 0'])
titles=["Number of anemic participants", "Number of deceased participants", "Number of participants with fever", "Number of participants with high bloodpressure", "Number of participants with hypothermia", "Number of participants with low oxygen level", "Number of participants with respiratory distress", "Number of participants with abdominal pain", "Number of participants with appetite loss", "Number of participants with chest pain", "Number of participants with chills", "Number of participants with cough", "Number of participants with diarrhea", "Number of participants with fatigue", "Number of participants with headache", "Number of participants with nausea", "Number of participants with runny nose", "Number of participants with sore throat", "Number of participants with loss of taste or smell"]
labeling=[['anemic', 'not anemic'], ['deceased', 'not deceased'], ['fever', 'no fever'], ['high bloodpressure', 'no high bloodpressure'], ['hypothermia', 'no hypothermia'], ['low oxygen level', 'normal  oxygen level'], ['respiratory distress', 'no respiratory distress'], ['abdominal pain', 'no abdominal pain'], ['appetite loss', 'no appetite loss'], ['chest pain', 'no chest pain'], ['chills', 'no chills'],['cough', 'no cough'], ['diarrhea', 'no diarrhea'], ['fatigue', 'no fatigue'], ['headache', 'no headache'], ['nausea', 'no nausea'], ['runny nose', 'no runny nose'], ['sore throat', 'no sore throat'], ['loss of taste or smell', 'no loss of taste or smell']]
def sympt_plotten(df):
    plt.figure(figsize=(16, 12)).tight_layout()
    plt.subplots_adjust(hspace=0.8, wspace=0.5)
    for i in range(df.shape[1]):
        values= [np.count_nonzero(df.iloc[:,i]==1), np.count_nonzero(df.iloc[:,i]==0)]
        labels=labeling[i]
        X_axis = np.arange(len(labels))
        plt.subplot(4,5,i+1)
        #plt.figure(figsize=(15,6))
        plt.bar(X_axis, values)
        plt.xticks(X_axis, labels, fontsize=6)
        plt.ylabel("Number of samples", fontsize=7)
        plt.title(titles[i], fontsize=7)
        print(i)
    #plt.show()
sympt_plotten(df)
plt.show()

