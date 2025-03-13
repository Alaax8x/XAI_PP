# Importing the Library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import dice_ml
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import resample
from sklearn import metrics

# Reading the Data
data = pd.read_csv('Data/bankloan.csv')
print(data.head())

# Analyzing the Data
print(data.info())
print(data.describe())
data.drop('ID', axis='columns', inplace=True)
print(data.head())

# Balancing Data
minority_class = data[data['Personal.Loan'] == 1]
majority_class = data[data['Personal.Loan'] == 0]

# Up sample the minority class
minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)

# Combine the upsampled minority class with the majority class
balanced_data = pd.concat([majority_class, minority_upsampled])
print(balanced_data["Personal.Loan"].value_counts())

# Splitting and Training the Data
target = balanced_data["Personal.Loan"]
train_dataset, test_dataset, y_train, y_test = train_test_split(balanced_data,
                                                                target,
                                                                test_size=0.25,
                                                                random_state=0,
                                                                stratify=target)
x_train = train_dataset.drop('Personal.Loan', axis=1)
x_test = test_dataset.drop('Personal.Loan', axis=1)

# dice_ml.Data
d = dice_ml.Data(dataframe=train_dataset,
                 continuous_features=['Age', 'Experience', 'ZIP.Code',
                                      'CCAvg', 'Mortgage', 'Income', 'Family'],
                 outcome_name='Personal.Loan')

# Preparing the pipeline for the model
numerical = ['Age', 'Experience', 'ZIP.Code', 'CCAvg', 'Mortgage', 'Income', 'Family']
categorical = x_train.columns.difference(numerical)

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

transformations = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', transformations),
                      ('classifier', RandomForestClassifier())])
model = clf.fit(x_train, y_train, )

# Model evaluation
model_predict = model.predict(x_test)
print(classification_report(y_test, model_predict))

# Create a confusion matrix using sklearn
conf_matrix = metrics.confusion_matrix(y_test, model_predict, labels=[1, 0])

# Plot the confusion matrix using seaborn
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='PiYG', xticklabels=['Approved','Declined'], yticklabels=['Approved','Declined'])
plt.xlabel('Predicted Preference', fontsize=12)
plt.ylabel('Actual Preference', fontsize=12)
plt.show()

# Using sklearn backend
m = dice_ml.Model(model=model, backend="sklearn")

# Feature importance
importances = model.named_steps['classifier'].feature_importances_
print(importances)

# To display all the outcome
import pandas as pd; pd.set_option('display.max_rows', 1000); pd.set_option('display.max_columns', 1000); pd.set_option('display.width', 1000)

# Making an instance
new_query_instance  = {'Age':45,'Experience': 5,'Income': 40,'ZIP.Code': 94574,'Family': 1,'CCAvg': 6.1,'Education': 1,
                   'Mortgage': 160,'Securities.Account': 1,'CD.Account': 1,'Online': 1,'CreditCard': 1}

features_to_vary  = ['Experience','Income','CCAvg','Education',
                     'Securities.Account','CD.Account','Online','CreditCard']
# pd.DataFrame([new_query_instance]

# Using method=random for generating CFs
exp = dice_ml.Dice(d, m, method="random")

# Generate CFs
e1 = exp.generate_counterfactuals(pd.DataFrame([new_query_instance]),
                                  total_CFs=2,
                                  desired_class="opposite",
                                  features_to_vary= features_to_vary)
e1.visualize_as_dataframe(show_only_changes=True)