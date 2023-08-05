## Import of needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Specify the folder name
datasets_folder = 'Datasets'

# Combine the folder name with the file names
file_paths = [
    os.path.join(datasets_folder, 'sy_cleaned_dataset.csv'),
    os.path.join(datasets_folder, 'MC_cleaned_dataset.csv'),
    os.path.join(datasets_folder, 'cf_cleaned_dataset.csv')
]

# Read the cleaned datasets
df = pd.read_csv(file_paths[0], sep=',')
df2 = pd.read_csv(file_paths[1], sep=',')
df_plot = pd.read_csv(file_paths[2], sep=',')

# Drop the 'anemia_confirmed' column from df2
df2 = df2.drop('anemia_confirmed', axis=1)

## Definition of Plot titles and lables for all plots
# casefile
titles_cf=["Age of participants", "Sex of participants", "Number of exposed participants", "Number of participants that are hospitalized", "Number of participants that were treated in ICU", "Number of deceased participants", "Number of participants with existing underlying conditions"]
labeling_cf=[['<18', '18-49', '50-64', '65+'], ['female', 'male'], ['not exposed', 'exposed'], ['not hospitalized', 'hospitalized'], ['no ICU-treatment', 'ICU-treatment'], ['not deceased', 'deceased'], ['no existing conditions', 'existing conditions']]
# symptoms
titles_df1 = ["Number of anemic participants", "Number of deceased participants", "Number of participants with fever", "Number of participants with high bloodpressure", "Number of participants with hypothermia", "Number of participants with low oxygen level", "Number of participants with respiratory distress", "Number of participants with abdominal pain", "Number of participants with appetite loss", "Number of participants with chest pain", "Number of participants with chills", "Number of participants with cough", "Number of participants with diarrhea", "Number of participants with fatigue", "Number of participants with headache", "Number of participants with nausea", "Number of participants with runny nose", "Number of participants with sore throat", "Number of participants with loss of taste or smell"]
labeling_df1 = [['not anemic', 'anemic'], ['not deceased', 'deceased'], ['no fever', 'fever'], ['no high bloodpressure', 'high bloodpressure',], ['no hypothermia', 'hypothermia'], ['normal oxygen level', 'low oxygen level'], ['no respiratory distress','respiratory distress'], ['no abdominal pain', 'abdominal pain'], ['no appetite loss', 'appetite loss'], ['no chest pain', 'chest pain'], ['no chills', 'chills'], ['no cough', 'cough'], ['no diarrhea', 'diarrhea'], ['no fatigue', 'fatigue'], ['no headache', 'headache'], ['no nausea', 'nausea'], ['no runny nose', 'runny nose'], ['no sore throat', 'sore throat'], ['no loss of taste or smell', 'loss of taste or smell']]
# medical conditions
titles_df2 = ["Age of participants", "BMI of participants", "Number of deceased participants", "Number of participants that were hospitalized before", "Number of participants with asthma", "Number of participants with cardiac diseases", "Number of participants with diabetes", "Number of participants with HIV", "Number of participants with hypertension", "Number of participants with pulmonary diseases", "Sex of participants", "Number of participants that smoke", "Number of participants with uncontrolled diabetes", "Exposure risk of participants"]
labeling_df2 = [['<18','18-44', '45-64', '65+'], ['underw.', 'normal', 'overw.', 'obesity'], ['not deceased','deceased'], ['not hospitalized before', 'hospitalized before'], ['no asthma','asthma'], ['no cardiac diseases','cardiac diseases'], ['no diabetes','diabetes'], ['no HIV','HIV pain'], ['no hypertension','hypertension'], ['no pulmonary diseases','pulmonary diseases'], ['male','female'], ['not smoking','smoking'], ['no uncontr. diabetes','uncontrolled diabetes'], ['no exposure risk','exposure risk']]

## for each used column there sholud be a visulisation of the data distribution and it should show how the predicted variable is distributed ej how many people of each age group have deceased
def plot_stacked_bar(df, titles, labeling, grouping, saveplot, legendnames):
    # define of the number of plots
    num_plots = len(titles)
    # define the number of cols and rows
    num_cols = 4
    num_rows = (num_plots - 1) // num_cols + 1
    plt.figure(figsize=(16, 4 * num_rows))
    # definition of the lables in legend
    mylabels= legendnames
    
    for i in range(num_plots):
        # define new subplots with certain spacing
        ax = plt.subplot(num_rows, num_cols, i + 1)
        plt.subplots_adjust(hspace=0.8, wspace=0.5)
        # group by grouping attribute to show the data as stacked bars
        data = df.groupby([df.iloc[:, i], grouping]).size().unstack()
        # plot the data as bars
        data.plot(kind='bar', stacked=True, ax=ax, color=['orange', 'blue'])
        # set axis tick-labels/ labels / a title / lables for the legend
        ax.set_xticklabels(labeling[i], fontsize=7, rotation=0)
        ax.set_ylabel("Number of samples", fontsize=7)
        plt.legend(labels=mylabels, fontsize=6)
        ax.set_title(titles[i], fontsize=7)
    
    # save the figures (with all belonging plots) as a .png-file
    plot_path = os.path.join('Plots', saveplot)
    plt.savefig(plot_path)

# used the predefined function to plot all predicted variables in dependence of their different predictor variables
plot_stacked_bar(df, titles_df1, labeling_df1, 'deceased', "plt_symptoms", ['alive','deceased'])
plot_stacked_bar(df2, titles_df2, labeling_df2, 'deceased', "plt_medical_cond", ['alive','deceased'])
plot_stacked_bar(df_plot, titles_cf, labeling_cf, 'hosp_yn', "plt_casefile_hospital",['not hosp.','hospitalised'])
plot_stacked_bar(df_plot, titles_cf, labeling_cf, 'icu_yn', "plt_casefile_icu", ['not in ICU','in ICU'])
plot_stacked_bar(df_plot, titles_cf, labeling_cf, 'death_yn',"plt_casefile_death", ['alive','deceased'])
