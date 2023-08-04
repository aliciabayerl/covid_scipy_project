import pandas as pd
reader = pd.read_csv("COVID-19_Case_Surveillance.csv", iterator=True, sep=',')
df = reader.get_chunk(100000) #45000000 rf dauert dann sehr lange

columns_to_delete = [
    "case_month", "res_state", "state_fips_code", "res_county", "county_fips_code", "ethnicity", "case_positive_specimen_interval",
    "case_onset_interval", "case_positive_specimen_interval", "case_onset_interval", "process", "symptom_status","race", "process"
]
df = df.drop(columns=columns_to_delete)
df.columns


# TODO: Histogram machen
age_range_mapping = {
    "0 - 17 years": 0,
    "18 to 49 years": 1,
    "50 to 64 years": 2,
    "65+ years": 3,
}
conditions_mapping = {
    "Yes": 1,
    "No": 0
}

sex_mapping = {
    "Female": 0,
    "Male": 1
}
exposure_mapping = {
    "Missing": 0,
    "Unknown": 0,
    "Yes": 1
}
mapping = {
    "No": 0,
    "Yes": 1
}
df.drop(df[df['current_status'] == 'Probable Case'].index, inplace= True)
df['age_group'] = df['age_group'].map(age_range_mapping)
df['exposure_yn'] = df['exposure_yn'].map(exposure_mapping)
df['sex'] = df['sex'].map(sex_mapping)
df.drop(df[df['hosp_yn'] == 'Unknown'].index, inplace= True)
df.drop(df[df['hosp_yn'] == 'Missing'].index, inplace= True)
df['hosp_yn'] = df['hosp_yn'].map(mapping)
df.drop(df[df['icu_yn'] == 'Unknown'].index, inplace= True)
df.drop(df[df['icu_yn'] == 'Missing'].index, inplace= True)
df['icu_yn'] = df['icu_yn'].map(mapping)
df=df.dropna(subset = ['death_yn'])
df.drop(df[df['death_yn'] == 'Unknown'].index, inplace= True)
df.drop(df[df['death_yn'] == 'Missing'].index, inplace= True)
df['death_yn'] = df['death_yn'].map(mapping)
#NA aus sex entfernen
df=df.dropna(subset = ['sex'])
df=df.dropna(subset = ['age_group'])
df['underlying_conditions_yn'] = df['underlying_conditions_yn'].fillna('No')
df['underlying_conditions_yn'] = df['underlying_conditions_yn'].map(conditions_mapping)
df.current_status.unique()
df = df.drop(columns='current_status')

gfg_csv_data = df.to_csv('D:/Dokumente_Festplatte/Master_Geoinformatik/Semester2/scientificpython/Abschlussaufgabe/covid_scipy_project/casefile_breinigt.csv', index = True)

