import pandas as pd

data = pd.read_csv('df_riskfactormanuscript.csv')

columns_to_delete = [
    "fever", "age_years", "anemia", "anyinfectious", "bmi_adult", "bmi_cat", "country.x", "covidcasestatus_new", "exposure_workingoutsidehome", "form.case..case_id", "obs_appearance", "Region_collapsed", "Region_manuscript",
    "respiratorydistress", "history_chronic_cat", "sex", "smoke", "Studysite_manuscript", "suspected_malaria",
    "symptoms_abdominalpain.x", "symptoms_any", "symptoms_appetite", "symptoms_chestpain.x",
    "symptoms_chills.x", "symptoms_cough.x", "symptoms_diarrhea.x", "symptoms_fatigue.x",
    "symptoms_headache.x", "symptoms_jointpain.x", "symptoms_nausea.x", "symptoms_runnynose.x",
    "symptoms_sob.x", "symptoms_sorethroat.x", "symptoms_tasteorsmell", "symptoms_wheezing.x",
    "test_reason"
]

exposure_columns = [
    "exposure_carecovidpatient", "exposure_contactcovidcase", "exposure_hcw", "exposure_visithcf"
]

data['exposure_risk'] = data[exposure_columns].apply(lambda row: 'yes' if 'yes' in row.values else 'no', axis=1)

data = data.drop(columns=exposure_columns)
data = data.drop(columns=columns_to_delete)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print(data)