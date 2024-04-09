import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

# Load the dataset
data = pd.read_csv('ML_exam/tennisdata.csv')

# Convert textual data into numerical data and separate features and target variable
features = data.apply(LabelEncoder().fit_transform)
target_column = features.pop('PlayTennis')

# Create, train, and visualize the decision tree model
visualization = export_text(
    DecisionTreeClassifier(criterion='entropy').fit(features, target_column),
    feature_names=list(features), class_names=['No', 'Yes'])

print(visualization)