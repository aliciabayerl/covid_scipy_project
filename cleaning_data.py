import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

## Load Data
## Information: COVID-19_Case_Surveillance.csv cannot be uploaded to git_hub due to its size of 13 Mio KB
## please foolow the download link, download the file yourself and put it in the directory you are using
## due to the same fact we downsized our data by using only the first 100000 row of the file

current_dir = os.getcwd()
datasets_folder = 'Datasets'  

data = pd.read_csv(os.path.join(datasets_folder, 'df_riskfactormanuscript.csv'))
df1 = pd.read_csv(os.path.join(datasets_folder, 'df_riskfactormanuscript.csv'))
reader = pd.read_csv("COVID-19_Case_Surveillance.csv", iterator=True, sep=',')
df = reader.get_chunk(100000)

## Predefinition of the columns that should be deleted
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
columns_to_delete_cf = [
    "case_month", "res_state", "state_fips_code", "res_county", "county_fips_code", "ethnicity", "case_positive_specimen_interval",
    "case_onset_interval", "case_positive_specimen_interval", "case_onset_interval", "process", "symptom_status","race", "process"
]

## Rename Columns if necissary
data.rename(columns={'uncontrolled_diabetes8': 'uncontrolled_diabetes', 'anemic_yn': 'anemia_confirmed'}, inplace=True)
df1.rename(columns={'anemic_yn':'anemic','covidcasestatus_new':'covid','highbloodpressure_enrollment_13080':'highbloodpressure','hypothermia_enrollment':'hypothermia','low_oxygen94_enrollment':'low_oxygen', 'symptoms_abdominalpain.x':'abdominalpain', 'symptoms_appetite':'appetite', 'symptoms_chestpain.x':'chestpain','symptoms_chills.x':'chills', 'symptoms_cough.x':'cough', 'symptoms_diarrhea.x':'diarrhea', 'symptoms_fatigue.x':'fatigue', 'symptoms_headache.x':'headache', 'symptoms_nausea.x':'nausea','symptoms_runnynose.x':'runnynose', 'symptoms_sorethroat.x':'sorethroat','symptoms_tasteorsmell':'tasteorsmell' },inplace=True)

## Columns for that na values should be mapped to 'no' because there is no certainty that the people have a certain disease or other features
columns_to_fill_no = [
    'anemia_confirmed', 'ever_hospitalized', 'history_cardiac', 'uncontrolled_diabetes', 'history_asthma',
    'history_hiv', 'history_hypertension', 'history_pulmonary', 'smoke'
]

columns_to_fill_no2 = [
    'anemic', 'highbloodpressure','low_oxygen'
]

# Predefine exposure columns that shall be merged
exposure_columns = [
    "exposure_carecovidpatient", "exposure_contactcovidcase", "exposure_hcw", "exposure_visithcf"
]

# Predefine columns that shall be deleted
columns_to_delete_na2 = [ 
    'fever', 'hypothermia', 'chestpain', 'headache', 'runnynose', 'sorethroat',
    'tasteorsmell'
]

## Predefinition of the integer values to which the current categories should be mapped
age_range_mapping_cf = {
    "0 - 17 years": 0,
    "18 to 49 years": 1,
    "50 to 64 years": 2,
    "65+ years": 3,
}
conditions_mapping_cf = {
    "Yes": 1,
    "No": 0
}
sex_mapping_cf = {
    "Female": 0,
    "Male": 1
}
exposure_mapping_cf = {
    "Missing": 0,
    "Unknown": 0,
    "Yes": 1
}
mapping_cf = {
    "No": 0,
    "Yes": 1
}
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

## Replace Data, fill missing values and rename, drop columns not needed or with small number of missing values
# Medical condition

data['exposure_risk'] = data[exposure_columns].apply(lambda row: 'yes' if 'yes' in row.values else 'no', axis=1)

data[columns_to_fill_no] = data[columns_to_fill_no].fillna('no')

data['ever_hospitalized'] = data['ever_hospitalized'].replace({
    "Never hospitalized (Outpatient managed)": "no",
    "Ever hospitalized": "yes"
})
data['bmi_cat'] = data["bmi_cat"].fillna('normal weight')
data['deceased'] = data['deceased'].replace("deceased", 'yes')

# Symptoms

df1 = df1.dropna(subset = columns_to_delete_na2)
df1[columns_to_fill_no2] = df1[columns_to_fill_no2].fillna('no')
df1['highbloodpressure'] = df1['highbloodpressure'].replace('High blood pressure- over 130/80', 'yes')
df1['low_oxygen'] = df1['low_oxygen'].replace('Normal oxygen level- above or equal to 94', 'no')
df1['deceased'] = df1['deceased'].replace("deceased", 'yes')
df1['covid'] = df1['covid'].replace("confirmed (rtpcr)", 'yes')

# Casefile

df.drop(df[df['current_status'] == 'Probable Case'].index, inplace= True)

# Drop unknown / missing data in the columns that should be predicted (hospitalized, ICU, death) and drop Nan-Values in these columns
df.drop(df[df['hosp_yn'] == 'Unknown'].index, inplace= True)
df.drop(df[df['hosp_yn'] == 'Missing'].index, inplace= True)
df.drop(df[df['icu_yn'] == 'Unknown'].index, inplace= True)
df.drop(df[df['icu_yn'] == 'Missing'].index, inplace= True)
df=df.dropna(subset = ['death_yn'])
df.drop(df[df['death_yn'] == 'Unknown'].index, inplace= True)
df.drop(df[df['death_yn'] == 'Missing'].index, inplace= True)

# Addtionally drop Nan-Values in predictor rows, because the used data source is big enough to drop them all andstill have enough rows remaining
df=df.dropna(subset = ['sex'])
df=df.dropna(subset = ['age_group'])

# in the column underlying_conditions there are to many Nan values to drop them all due to the fact that only if there are underlying conditions it is indicated, therefore all Nan-rows should be filled with no
df['underlying_conditions_yn'] = df['underlying_conditions_yn'].fillna('No')



## Mapping
## the strings are mapped to defined integer values to use them as variables in the models
## therfore the predefined mapping lists are used

# Medical Conditions
data['age_categories'] = data['age_categories'].map(age_range_mapping)
data['bmi_cat'] = data['bmi_cat'].map(bmi_mapping)

# Casefile
df['age_group'] = df['age_group'].map(age_range_mapping_cf)
df['exposure_yn'] = df['exposure_yn'].map(exposure_mapping_cf)
df['sex'] = df['sex'].map(sex_mapping_cf)
df['hosp_yn'] = df['hosp_yn'].map(mapping_cf)
df['icu_yn'] = df['icu_yn'].map(mapping_cf)
df['death_yn'] = df['death_yn'].map(mapping_cf)
df['underlying_conditions_yn'] = df['underlying_conditions_yn'].map(conditions_mapping_cf)


## Delete columns or rows

# Medical Conditions - Deletion of not needed column, ids and missing value rows in the deceased column
data = data.drop(columns=exposure_columns)
data = data.drop(columns=columns_to_delete)
data = data.dropna(subset=['deceased'], axis=0)
data.drop(data.columns[0], axis=1, inplace=True)

# Symptoms - Deletion of unnecessary columns, missing value rows, unconfirmed covid cases, ids
df1 = df1.drop(columns=columns_to_delete2)
df1.drop(df1[df1['covid'] == 'Suspect- no valid test'].index, inplace=True)
df1.dropna(subset=['deceased'], axis=0, inplace=True)
df1.drop(columns='covid', inplace=True)
df1.drop(df1.columns[0], axis=1, inplace=True)

# Casefile - Deletion of unnecessary columns
df = df.drop(columns='current_status')
df = df.drop(columns=columns_to_delete_cf)


## Transform into numerical values
## certain strings are transformed to numerical values

# Medical Conditions
data = data.replace({"yes": 1, "no": 0})
data = data.replace({"male": 0, "female": 1})

# Symptoms
df1 = df1.replace({"yes": 1, "no": 0})

# Export the cleaned data to new datasets
pd.set_option('display.max_columns', None)

file_path = os.path.join(datasets_folder, 'mc_cleaned_dataset.csv')
data.to_csv(file_path, index=False)

file_path = os.path.join(datasets_folder, 'sy_cleaned_dataset.csv')
df1.to_csv(file_path, index=False)

file_path = os.path.join(datasets_folder, 'cf_cleaned_dataset.csv')
df.to_csv(file_path, index=False)
