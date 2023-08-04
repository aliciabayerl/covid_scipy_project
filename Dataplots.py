
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('sy_cleaned_dataset.csv', sep=',')
df2 = pd.read_csv('MC_cleaned_dataset.csv', sep=',')
df2 = df2.drop('anemia_confirmed', axis=1)

titles_df1 = ["Number of anemic participants", "Number of deceased participants", "Number of participants with fever", "Number of participants with high bloodpressure", "Number of participants with hypothermia", "Number of participants with low oxygen level", "Number of participants with respiratory distress", "Number of participants with abdominal pain", "Number of participants with appetite loss", "Number of participants with chest pain", "Number of participants with chills", "Number of participants with cough", "Number of participants with diarrhea", "Number of participants with fatigue", "Number of participants with headache", "Number of participants with nausea", "Number of participants with runny nose", "Number of participants with sore throat", "Number of participants with loss of taste or smell"]
labeling_df1 = [['anemic', 'not anemic'], ['deceased', 'not deceased'], ['fever', 'no fever'], ['high bloodpressure', 'no high bloodpressure'], ['hypothermia', 'no hypothermia'], ['low oxygen level', 'normal oxygen level'], ['respiratory distress', 'no respiratory distress'], ['abdominal pain', 'no abdominal pain'], ['appetite loss', 'no appetite loss'], ['chest pain', 'no chest pain'], ['chills', 'no chills'], ['cough', 'no cough'], ['diarrhea', 'no diarrhea'], ['fatigue', 'no fatigue'], ['headache', 'no headache'], ['nausea', 'no nausea'], ['runny nose', 'no runny nose'], ['sore throat', 'no sore throat'], ['loss of taste or smell', 'no loss of taste or smell']]

titles_df2 = ["Age of participants", "BMI of participants", "Number of deceased participants", "Number of participants that were hospitalized before", "Number of participants with asthma", "Number of participants with cardiac diseases", "Number of participants with diabetes", "Number of participants with HIV", "Number of participants with hypertension", "Number of participants with pulmonary diseases", "Sex of participants", "Number of participants that smoke", "Number of participants with uncontrolled diabetes", "Exposure risk of participants"]
labeling_df2 = [['18-44', '<18', '45-64', '65+'], ['normal', 'underw.', 'overw.', 'obesity'], ['deceased', 'not deceased'], ['hospitalized before', 'not hospitalized before'], ['asthma', 'no asthma'], ['cardiac diseases', 'no cardiac diseases'], ['diabetes', 'no diabetes'], ['HIV pain', 'no HIV'], ['hypertension', 'no hypertension'], ['pulmonary diseases', 'no pulmonary diseases'], ['female', 'male'], ['smoking', 'not smoking'], ['uncontrolled diabetes', 'no uncontrolled diabetes'], ['exposure risk', 'no exposure risk']]

def plot_stacked_bar(df, titles, labeling):
    num_plots = len(titles)
    num_cols = 4
    num_rows = (num_plots - 1) // num_cols + 1
    plt.figure(figsize=(16, 4 * num_rows))
    
    for i in range(num_plots):
        ax = plt.subplot(num_rows, num_cols, i + 1)
        data = df.groupby([df.iloc[:, i], 'deceased']).size().unstack()
        data.plot(kind='bar', stacked=True, ax=ax, color=['orange', 'blue'])
        ax.set_xticklabels(labeling[i], fontsize=7, rotation=0)
        ax.set_ylabel("Number of samples", fontsize=7)
        ax.set_title(titles[i], fontsize=7)
    
    plt.tight_layout()
    plt.show()

plot_stacked_bar(df, titles_df1, labeling_df1)
plot_stacked_bar(df2, titles_df2, labeling_df2)
