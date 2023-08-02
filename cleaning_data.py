import pandas as pd
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('df_riskfactormanuscript.csv')
df1 = pd.read_csv('df_riskfactormanuscript.csv')

columns_to_delete = [
    "fever", "low_oxygen94_enrollment", "age_years", "anemia", "anyinfectious", "bmi_adult", "bmi_obese", "country.x", "covidcasestatus_new", "exposure_workingoutsidehome", "form.case..case_id", "obs_appearance", "Region_collapsed", "Region_manuscript",
    "respiratorydistress", "history_tb", "history_chronic_cat", "highbloodpressure_enrollment_13080", "hypothermia_enrollment", "Studysite_manuscript", "suspected_malaria",
    "symptoms_abdominalpain.x", "symptoms_any", "symptoms_appetite", "symptoms_chestpain.x",
    "symptoms_chills.x", "symptoms_cough.x", "symptoms_diarrhea.x", "symptoms_fatigue.x",
    "symptoms_headache.x", "symptoms_jointpain.x", "symptoms_nausea.x", "symptoms_runnynose.x",
    "symptoms_sob.x", "symptoms_sorethroat.x", "symptoms_tasteorsmell", "symptoms_wheezing.x",
    "test_reason"
]

columns_to_delete2 = [
    "age_categories", "age_years", "anemia", "anyinfectious","bmi_adult","bmi_cat", "bmi_obese", "country.x", "ever_hospitalized", "obs_appearance","symptoms_sob.x","exposure_carecovidpatient", "exposure_contactcovidcase", "exposure_hcw", "exposure_visithcf",
    "exposure_workingoutsidehome", "history_asthma", "history_cardiac", "history_chronic_cat", "history_diabetes",
    "history_hiv", "history_hypertension", "history_pulmonary", "history_tb", "form.case..case_id", "Region_collapsed",
    "Region_manuscript", "sex", "smoke", "Studysite_manuscript", "suspected_malaria", "symptoms_any", "test_reason", "uncontrolled_diabetes8", "symptoms_jointpain.x", "symptoms_wheezing.x"
]

# Rename Columns
data.rename(columns={'uncontrolled_diabetes8': 'uncontrolled_diabetes'}, inplace=True)
data.rename(columns={'anemic_yn': 'anemia_confirmed'}, inplace=True)
df1.rename(columns={'anemic_yn':'anemic','covidcasestatus_new':'covid','highbloodpressure_enrollment_13080':'highbloodpressure','hypothermia_enrollment':'hypothermia','low_oxygen94_enrollment':'low_oxygen', 'symptoms_abdominalpain.x':'abdominalpain', 'symptoms_appetite':'appetite', 'symptoms_chestpain.x':'chestpain','symptoms_chills.x':'chills', 'symptoms_cough.x':'cough', 'symptoms_diarrhea.x':'diarrhea', 'symptoms_fatigue.x':'fatigue', 'symptoms_headache.x':'headache', 'symptoms_nausea.x':'nausea','symptoms_runnynose.x':'runnynose', 'symptoms_sorethroat.x':'sorethroat','symptoms_tasteorsmell':'tasteorsmell' },inplace=True)


columns_to_fill_no = [
    'anemia_confirmed', 'ever_hospitalized', 'history_cardiac', 'uncontrolled_diabetes', 'history_asthma',
    'history_hiv', 'history_hypertension', 'history_pulmonary', 'smoke'
]

columns_to_fill_no2 = [
    'anemic', 'highbloodpressure','low_oxygen'
]


exposure_columns = [
    "exposure_carecovidpatient", "exposure_contactcovidcase", "exposure_hcw", "exposure_visithcf"
]

columns_to_delete_na2 = [ 
    'fever', 'hypothermia_enrollment', 'symptoms_chestpain.x', 'symptoms_headache.x', 'symptoms_runnynose.x', 'symptoms_sorethroat.x',
    'symptoms_tasteorsmell'
]

age_range_mapping = {
    "< 18": 0,
    "18-44": 1,
    "45-64": 2,
    "65+": 3,
}

bmi_mapping = {
    "underweight": 0,
    "normal weight": 1,
    "overweight": 2,
    "obesity": 3,
}

# Replace Data or Rename

    # Medical conditions
data['exposure_risk'] = data[exposure_columns].apply(lambda row: 'yes' if 'yes' in row.values else 'no', axis=1)
data[columns_to_fill_no] = data[columns_to_fill_no].fillna('no')
data['ever_hospitalized'] = data['ever_hospitalized'].replace({
    "Never hospitalized (Outpatient managed)": "no",
    "Ever hospitalized": "yes"
})
data['bmi_cat'] = data["bmi_cat"].fillna('normal weight')
data['deceased'] = data['deceased'].replace("deceased", 'yes')

    # Symptoms
df1[columns_to_fill_no2] = df1[columns_to_fill_no2].fillna('no')
df1['highbloodpressure'] = df1['highbloodpressure'].replace('High blood pressure- over 130/80', 'yes')
df1['low_oxygen'] = df1['low_oxygen'].replace('Normal oxygen level- above or equal to 94', 'no')
df1['deceased'] = df1['deceased'].replace("deceased", 'yes')
df1['covid'] = df1['covid'].replace("confirmed (rtpcr)", 'yes')

# Mapping 

    # Medical Conditions
data['age_categories'] = data['age_categories'].map(age_range_mapping)
data['bmi_cat'] = data['bmi_cat'].map(bmi_mapping)


# Delete columns or rows 

    # Medical Conditions
data = data.drop(columns=exposure_columns)
data = data.drop(columns=columns_to_delete)
data = data.dropna(subset=['deceased'], axis=0)
data.drop(data.columns[0], axis=1, inplace=True)

    # Symptoms
df1 = df1.drop(columns=columns_to_delete2)
#df1['covid'].unique()
#df1['deceased'].unique()
df1.drop (df1 [df1 ['covid'] == 'Suspect- no valid test'].index, inplace= True)
df1=df1.dropna(subset = ['deceased'])
df1.drop(df1.columns[0], axis=1, inplace=True)
df1 = df1.drop(columns='covid')

# Transform into numerical values
data = data.replace({"yes": 1, "no": 0})
data = data.replace({"male": 0, "female": 1})

df1 = df1.replace({"yes": 1, "no": 0})

pd.set_option('display.max_columns', None)

data.to_csv('mc_cleaned_dataset.csv', index=False)
df1.to_csv('sy_cleaned_dataset.csv', index=False)