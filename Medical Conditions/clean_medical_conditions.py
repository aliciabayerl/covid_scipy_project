import pandas as pd
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('df_riskfactormanuscript.csv')

columns_to_delete = [
    "fever", "age_years", "anemia", "anyinfectious", "bmi_adult", "bmi_obese", "country.x", "covidcasestatus_new", "exposure_workingoutsidehome", "form.case..case_id", "obs_appearance", "Region_collapsed", "Region_manuscript",
    "respiratorydistress", "history_tb", "history_chronic_cat", "highbloodpressure_enrollment_13080", "hypothermia_enrollment", "Studysite_manuscript", "suspected_malaria",
    "symptoms_abdominalpain.x", "symptoms_any", "symptoms_appetite", "symptoms_chestpain.x",
    "symptoms_chills.x", "symptoms_cough.x", "symptoms_diarrhea.x", "symptoms_fatigue.x",
    "symptoms_headache.x", "symptoms_jointpain.x", "symptoms_nausea.x", "symptoms_runnynose.x",
    "symptoms_sob.x", "symptoms_sorethroat.x", "symptoms_tasteorsmell", "symptoms_wheezing.x",
    "test_reason"
]

data.rename(columns={'low_oxygen94_enrollment': 'low_oxygen_level'}, inplace=True)
data.rename(columns={'uncontrolled_diabetes8': 'uncontrolled_diabetes'}, inplace=True)

columns_to_fill_no = [
    'anemia_confirmed', 'ever_hospitalized', 'history_cardiac', 'uncontrolled_diabetes', 'history_asthma',
    'history_hiv', 'history_hypertension', 'history_pulmonary', 'low_oxygen_level', 'smoke'
]


exposure_columns = [
    "exposure_carecovidpatient", "exposure_contactcovidcase", "exposure_hcw", "exposure_visithcf"
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



# Rename and Replace
data.rename(columns={'anemic_yn': 'anemia_confirmed'}, inplace=True)
data['exposure_risk'] = data[exposure_columns].apply(lambda row: 'yes' if 'yes' in row.values else 'no', axis=1)
data[columns_to_fill_no] = data[columns_to_fill_no].fillna('no')
data['ever_hospitalized'] = data['ever_hospitalized'].replace({
    "Never hospitalized (Outpatient managed)": "no",
    "Ever hospitalized": "yes"
})
data['low_oxygen_level'] = data['low_oxygen_level'].replace('Normal oxygen level- above or equal to 94', 'no')
data['bmi_cat'] = data["bmi_cat"].fillna('normal weight')
data['deceased'] = data['deceased'].replace("deceased", 'yes')

# Mapping
data['age_categories'] = data['age_categories'].map(age_range_mapping)
data['bmi_cat'] = data['bmi_cat'].map(bmi_mapping)

# Delete columns or rows 

data = data.drop(columns=exposure_columns)
data = data.drop(columns=columns_to_delete)
data = data.dropna(subset=['deceased'], axis=0)
data.drop(data.columns[0], axis=1, inplace=True)

# Transform into numerical values
data = data.replace({"yes": 1, "no": 0})
data = data.replace({"male": 0, "female": 1})

pd.set_option('display.max_columns', None)

print(data.head(15))

data.to_csv('modified_dataset.csv', index=False)
