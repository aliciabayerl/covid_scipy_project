


## Visualsierung der Daten

## Bibliothekten einladen
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Daten einlesen
df = pd.read_csv('sy_cleaned_dataset.csv', sep=',')
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
        plt.xticks(X_axis, labels, fontsize=5)
        plt.ylabel("Number of samples", fontsize=7)
        plt.title(titles[i], fontsize=7)
        print(i)
    #plt.show()
sympt_plotten(df)
plt.show()

df2= pd.read_csv('MC_cleaned_dataset.csv', sep=',')
df2= df2.drop('anemia_confirmed', axis=1)
print(df2)
print(df2.columns)

titles=["Age of participants", "BMI of participants", "Number of deceased participants", "Number of participants that were hospitalized before", "Number of participants with asthma", "Number of participants with cardiac diseases", "Number of participants with diabetes", "Number of participants with HIV", "Number of participants with hypertension", "Number of participants with pulmonary diseases", "Sex of participants", "Number of participants that smoke", "Number of participants with uncontrolled diabetes", "Exposure risk of participants"]
labeling=[['18-44', '<18', '45-64', '65+'], ['normal', 'underw.', 'overw.', 'obesity'], ['deceased', 'not deceased'], ['hospitalized before', 'not hospitalized before'], ['asthma', 'no asthma'], ['cardiac diseases', 'no cardiac diseases'], ['diabetes', 'no diabetes'], ['HIV pain', 'no HIV'], ['hypertension', 'no hypertension'], ['pulmonary diseases', 'no pulmonary diseases'], ['female', 'male'],['smoking', 'not smoking'], ['uncontrolled diabetes', 'no uncontrolled diabetes'], ['exposure risk', 'no exposure risk']]
def sympt_plotten2(df):
    plt.figure(figsize=(16, 12)).tight_layout()
    plt.subplots_adjust(hspace=0.8, wspace=0.5)
    for i in range(df2.shape[1]):
        if i ==0 or i== 1:
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
sympt_plotten2(df2)
plt.show()

