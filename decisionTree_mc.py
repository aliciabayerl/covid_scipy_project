import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
import graphviz

data = pd.read_csv('modified_dataset.csv')

features = ['age_categories', 'ever_hospitalized', 'history_diabetes', 'bmi_cat',
            'history_hypertension', 'exposure_risk', 'sex', 'anemia_confirmed', 'history_cardiac',
            'uncontrolled_diabetes', 'history_hiv', 'smoke', 'history_pulmonary']

target = 'deceased'

X = data[features]
y = data[target]

#Perform label encoding for categorical variables 
label_encoder = LabelEncoder()
for column in X.select_dtypes(include='object').columns:
    X[column] = label_encoder.fit_transform(X[column])

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create the Decision Tree classifier
decision_tree = DecisionTreeClassifier(max_depth=6,random_state=42)

#Fit the model to the training data
decision_tree.fit(X_train, y_train)


#Visualize the decision tree using graphviz
dot_data = export_graphviz(decision_tree, out_file=None, 
                           feature_names=features, class_names=['Not Deceased', 'Deceased'], 
                           filled=True, rounded=True, special_characters=True)


graph = graphviz.Source(dot_data)

graph.render("decision_tree")  

#the decision tree suggests that individuals with higher oxygen levels and younger age are more likely to survive covid
#having no history of diabetes and not being hospitalized before also increase the chance of survival 
#lower exposure risk improves survival odds, higher oxygen levels and younger age are critical factors for predicting survival,
#while exposure risk and sex may also play a role

