#Scipy_Project
## Test Bereinigung Daten
import pandas as pd
import numpy as np
df = pd.read_csv('D:/Dokumente_Festplatte/Master_Geoinformatik/Semester2/scientificpython/Abschlussaufgabe/df_riskfactormanuscript_forhdx.csv', sep=';')
df1=df #Arbeitsversion erstellen
df1
df1 = df1.drop(columns=['age_categories', 'age_years', "anemia", "anyinfectious","bmi_adult","bmi_cat", "bmi_obese", "country.x", "ever_hospitalized", "obs_appearance","symptoms_sob.x","exposure_carecovidpatient", "exposure_contactcovidcase", "exposure_hcw", "exposure_visithcf", "exposure_workingoutsidehome", "history_asthma", "history_cardiac", "history_chronic_cat", "history_diabetes", "history_hiv", "history_hypertension", "history_pulmonary", "history_tb", "form.case..case_id", "Region_collapsed", "Region_manuscript", "sex", "smoke", "Studysite_manuscript", "suspected_malaria", "symptoms_any", "test_reason", "uncontrolled_diabetes8"])
#del df['Gehalt']
df1=df1.drop(columns=['symptoms_jointpain.x', 'symptoms_wheezing.x'])
df1
df1['covidcasestatus_new'].unique()
df1['deceased'].unique()
#df1[df1 ['covidcasestatus_new'] == 'Suspect- no valid test']
df1['deceased'].unique()
df1
df1.drop (df1 [df1 ['covidcasestatus_new'] == 'Suspect- no valid test'].index, inplace= True)
df1=df1.dropna(subset = ['deceased'])
print(df['anemic_yn'].unique())
print(df['fever'].unique())
print(df['highbloodpressure_enrollment_13080'].unique())
print(df['hypothermia_enrollment'].unique())
print(df['low_oxygen94_enrollment'].unique())
print(df['respiratorydistress'].unique())
print(df['symptoms_abdominalpain.x'].unique())

print(df['symptoms_appetite'].unique())
print('test')
print(df['symptoms_chestpain.x'].unique())
print(df['symptoms_chills.x'].unique())
print(df['symptoms_cough.x'].unique())
print(df['symptoms_diarrhea.x'].unique())
print(df['symptoms_fatigue.x'].unique())
print('test')
print(df['symptoms_headache.x'].unique())

print(df['symptoms_nausea.x'].unique())
print(df['symptoms_runnynose.x'].unique())
print(df['symptoms_sorethroat.x'].unique())
print(df['symptoms_tasteorsmell'].unique())

df1['anemic_yn'] = df1['anemic_yn'].fillna('no')
# da bei fever nur ein Na Wert vorliegt kann dieser gelöscht werden ohne zu viele Daten zu verlieren
df1=df1.dropna(subset = ['fever'])
#df1['anemic_yn'] = df1['anemic_yn'].replace(NaN, 'no')
df1['highbloodpressure_enrollment_13080'] = df1['highbloodpressure_enrollment_13080'].fillna('no')
df1['highbloodpressure_enrollment_13080'] = df1['highbloodpressure_enrollment_13080'].replace('High blood pressure- over 130/80', 'high')
# da bei hypothermia nur ein Na Wert vorliegt kann dieser gelöscht werden ohne zu viele Daten zu verlieren
df1=df1.dropna(subset = ['hypothermia_enrollment'])
df1['low_oxygen94_enrollment'] = df1['low_oxygen94_enrollment'].fillna('normal')
df1['low_oxygen94_enrollment'] = df1['low_oxygen94_enrollment'].replace('Normal oxygen level- above or equal to 94', 'normal')
#chestpain nur ein na Wert deshalb entfernen
df1=df1.dropna(subset = ['symptoms_chestpain.x'])
#headache nur ein na Wert deshalb entfernen
df1=df1.dropna(subset = ['symptoms_headache.x'])
#runny nose nur ein na Wert deshalb entfernen
df1=df1.dropna(subset = ['symptoms_runnynose.x'])
#symptoms_sorethroat nur ein na Wert deshalb entfernen
df1=df1.dropna(subset = ['symptoms_sorethroat.x'])
#symptoms_tasteorsmell nur vier na Werte deshalb entfernen
df1=df1.dropna(subset = ['symptoms_tasteorsmell'])

df1.head()
df1.rename(columns={'anemic_yn':'anemic','covidcasestatus_new':'covid','highbloodpressure_enrollment_13080':'highbloodpressure','hypothermia_enrollment':'hypothermia','low_oxygen94_enrollment':'low_oxygen', 'symptoms_abdominalpain.x':'abdominalpain', 'symptoms_appetite':'appetite', 'symptoms_chestpain.x':'chestpain','symptoms_chills.x':'chills', 'symptoms_cough.x':'cough', 'symptoms_diarrhea.x':'diarrhea', 'symptoms_fatigue.x':'fatigue', 'symptoms_headache.x':'headache', 'symptoms_nausea.x':'nausea','symptoms_runnynose.x':'runnynose', 'symptoms_sorethroat.x':'sorethroat','symptoms_tasteorsmell':'tasteorsmell' },inplace=True)
df1.head()



