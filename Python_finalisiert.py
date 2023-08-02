#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Test Bereinigung Daten


# In[66]:


## Bibliothekten einladen
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import export_graphviz
from IPython.display import Image


# In[23]:


## Daten einlesen
df = pd.read_csv('D:/Dokumente_Festplatte/Master_Geoinformatik/Semester2/scientificpython/Abschlussaufgabe/df_riskfactormanuscript_forhdx.csv', sep=';')
df1=df #Arbeitsversion erstellen


# In[24]:


## Spalten löschen
df1 = df1.drop(columns=['age_categories', 'age_years', "anemia", "anyinfectious","bmi_adult","bmi_cat", "bmi_obese", "country.x", "ever_hospitalized", "obs_appearance","symptoms_sob.x","exposure_carecovidpatient", "exposure_contactcovidcase", "exposure_hcw", "exposure_visithcf", "exposure_workingoutsidehome", "history_asthma", "history_cardiac", "history_chronic_cat", "history_diabetes", "history_hiv", "history_hypertension", "history_pulmonary", "history_tb", "form.case..case_id", "Region_collapsed", "Region_manuscript", "sex", "smoke", "Studysite_manuscript", "suspected_malaria", "symptoms_any", "test_reason", "uncontrolled_diabetes8"])
df1=df1.drop(columns=['symptoms_jointpain.x', 'symptoms_wheezing.x'])


# In[25]:


## Überprüfen der unique values
## hier dürfen keine Na-Werte geduldet werden
df1['covidcasestatus_new'].unique()
df1['deceased'].unique()
df1.drop (df1 [df1 ['covidcasestatus_new'] == 'Suspect- no valid test'].index, inplace= True)
df1=df1.dropna(subset = ['deceased'])


# In[26]:


## Betrachtung der restlichen unique Werte
print(df['anemic_yn'].unique())
print(df['fever'].unique())
print(df['highbloodpressure_enrollment_13080'].unique())
print(df['hypothermia_enrollment'].unique())
print(df['low_oxygen94_enrollment'].unique())
print(df['respiratorydistress'].unique())
print(df['symptoms_abdominalpain.x'].unique())
print(df['symptoms_appetite'].unique())
print(df['symptoms_chestpain.x'].unique())
print(df['symptoms_chills.x'].unique())
print(df['symptoms_cough.x'].unique())
print(df['symptoms_diarrhea.x'].unique())
print(df['symptoms_fatigue.x'].unique())
print(df['symptoms_headache.x'].unique())
print(df['symptoms_nausea.x'].unique())
print(df['symptoms_runnynose.x'].unique())
print(df['symptoms_sorethroat.x'].unique())
print(df['symptoms_tasteorsmell'].unique())


# In[27]:


## für diese Spalten leigen sehr viele Na-Werte vor -> die entsprechenden Werte werden mit no ausgefüllt, da keine bestätigte 
## Anämie, , o.ä. vorliegt
df1['anemic_yn'] = df1['anemic_yn'].fillna('no')
## zusätzlich wird der Wert high bloodpressure durch yes ersetzt 
df1['highbloodpressure_enrollment_13080'] = df1['highbloodpressure_enrollment_13080'].fillna('no')
df1['highbloodpressure_enrollment_13080'] = df1['highbloodpressure_enrollment_13080'].replace('High blood pressure- over 130/80', 'yes')
## zusätzlich wird der Wert low oxygen durch yes ersetzt 
df1['low_oxygen94_enrollment'] = df1['low_oxygen94_enrollment'].fillna('no')
df1['low_oxygen94_enrollment'] = df1['low_oxygen94_enrollment'].replace('Normal oxygen level- above or equal to 94', 'no')

## da nur einzelne Na-Werte vorliegen z.B. bei Fieber, Hypothermia, Chestpain, headache, runny nose, sorethroat, taste or smell symptoms kann dieser einzelne Na-Wert gelöscht werden,
## ohne zu viele Daten zu verlieren
df1=df1.dropna(subset = ['fever'])
df1=df1.dropna(subset = ['hypothermia_enrollment'])
df1=df1.dropna(subset = ['symptoms_chestpain.x'])
df1=df1.dropna(subset = ['symptoms_headache.x'])
df1=df1.dropna(subset = ['symptoms_runnynose.x'])
df1=df1.dropna(subset = ['symptoms_sorethroat.x'])
df1=df1.dropna(subset = ['symptoms_tasteorsmell'])


# In[28]:


## Umbenneung der Spalten
df1.rename(columns={'anemic_yn':'anemic','covidcasestatus_new':'covid','highbloodpressure_enrollment_13080':'highbloodpressure','hypothermia_enrollment':'hypothermia','low_oxygen94_enrollment':'low_oxygen', 'symptoms_abdominalpain.x':'abdominalpain', 'symptoms_appetite':'appetite', 'symptoms_chestpain.x':'chestpain','symptoms_chills.x':'chills', 'symptoms_cough.x':'cough', 'symptoms_diarrhea.x':'diarrhea', 'symptoms_fatigue.x':'fatigue', 'symptoms_headache.x':'headache', 'symptoms_nausea.x':'nausea','symptoms_runnynose.x':'runnynose', 'symptoms_sorethroat.x':'sorethroat','symptoms_tasteorsmell':'tasteorsmell' },inplace=True)


# In[38]:


df1_prediction=df1.iloc[:,[1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
df1_prediction[df1_prediction == 'no'] = 0
df1_prediction[df1_prediction == 'yes'] = 1
df1_prediction.deceased[df1_prediction.deceased == 'deceased'] = 1 


# In[39]:


## Export Csv
gfg_csv_data = df1_prediction.to_csv('D:/Dokumente_Festplatte/Master_Geoinformatik/Semester2/scientificpython/Abschlussaufgabe/Symptome_breinigt.csv', index = True)


# In[ ]:


## Visualisierung der Datenverteilung


# In[ ]:


## Überwachte Klassifikation


# In[108]:


## Vorbedingungen
spaltenwahl= [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
df1_predictors=df1_prediction.iloc[:, spaltenwahl]
df1_predictorsnames=list(df1_predictors.columns)
df1_predictors=df1_predictors.to_numpy()
df1_predictors = df1_predictors.astype('float64')

abhaengig=df1_prediction.deceased
abhaengigunique=abhaengig.unique()
abhaengig=abhaengig.to_numpy()
abhaengig = abhaengig.astype('float64')


# In[120]:


## Entscheidungsbaum
model = DecisionTreeClassifier()
X, y = df1_predictors, abhaengig
model.fit(X, y)


# In[110]:


## Visualisierung
plt.figure(figsize=(16,10))
plot_tree(model, filled=True, proportion=True);


# In[ ]:


## Random Forest


# In[111]:


## Aufbauen eines RF-Modells
rf = RandomForestClassifier()
rffitx=rf.fit(X, y)


# In[112]:


## Durchführung einer Kreuzvalidierung (wegen der begrenzten Anzahl an Testsamples)
scores = np.zeros(1)
## Modell trainieren
rf =RandomForestClassifier() 
## Score mit Kreuzvalidierung berechnen
score = cross_val_score(rf, X, y, cv = 5)
scores = (np.mean(score))
print(str(np.mean(scores)) + " accuracy") # "with a standard deviation of " + str(np.std(scores)))    # print('max_score:', np.max(scores))


# In[113]:


## Feature-Bedeutung
importances = rffitx.feature_importances_
importances=np.array(importances)
importances=importances.reshape(1,18)
importancepd=pd.DataFrame(importances)
df1_predictorsnamesdf=np.array(df1_predictorsnames)
df1_predictorsnamesdf=df1_predictorsnamesdf.reshape(1,18)
importance_Tab=pd.DataFrame(np.concatenate((df1_predictorsnamesdf, importances)))
abhaengigunique=['0','1']
#abhaengigunique=list(np.unique(abhaengig))


# In[114]:


## Visualisierung eines der RF Bäume
fig = plt.figure(figsize=(15, 10))
plot_tree(rffitx.estimators_[0], 
          feature_names=df1_predictorsnames,
          class_names=abhaengigunique, 
          filled=True, rounded=True)
plt.show()


# In[ ]:





# In[ ]:


## Unüberwachtes Lernen
# K Means


# In[129]:


## Bibliotheken importieren
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


# In[138]:


## Initialisieren des Modells
# for two classes
kmeans = KMeans(n_clusters=2)                             
# fitting the data point inside the model
kmeans.fit(X)
# predecting the clusters (gives for eery datapoint the class it belongs to)
y_kmeans = kmeans.predict(X)


# In[133]:


## Genauigkeitsberechnung
knnscore = accuracy_score(y,y_kmeans)
knnscore

