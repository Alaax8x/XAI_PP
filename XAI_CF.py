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

# Reading the Data
data = pd.read_csv('Data/bankloan.csv')
print(data.head())

# Analyzing the Data
print(data.info())
print(data.describe())
data.drop('ID', axis='columns', inplace=True)
print(data.head())

# # Visualizing the Data
# # Histogram Representation
# data.hist(figsize=(20, 15), color='blue')
# plt.show()
#
# # Box Plot between Personal Loan v/s Income
# plt.figure(figsize=(5, 5))
# sns.boxplot(x='Personal.Loan', y='Income', data=data)
# plt.title('Income vs Personal Loan')
# plt.xlabel('Personal Loan')
# plt.ylabel('Income')
# plt.show()
#
# # Counting Loans
# loan_counts = data['Personal.Loan'].value_counts()
# print(loan_counts)
#
# # Count plot between approved and not approved loans
# plt.figure(figsize=(8, 6))
# loan_counts.plot(kind='bar', color=['blue', 'orange'])
# plt.title('Count of Approved and Not Approved Loans')
# plt.xlabel('Loan Status')
# plt.ylabel('Count')
# plt.xticks([0, 1], ['Not Approved', 'Approved'], rotation=0)
# plt.grid(axis='y')
# plt.show()

# Splitting and Training the Data
target = data["Personal.Loan"]
train_dataset, test_dataset, y_train, y_test = train_test_split(data,
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
model = clf.fit(x_train, y_train)

# Using sklearn backend
m = dice_ml.Model(model=model, backend="sklearn")

# Feature importance
importances = model.named_steps['classifier'].feature_importances_
print(importances)

# get MAD
# mads = d.get_mads(normalized=True)
# print(mads)
# create feature weights
# feature_weights = {}
# for feature in mads:
#     feature_weights[feature] = round(1/mads[feature], 2)
# print(feature_weights)

# max_weight = 10  # Set an upper limit
# for feature in feature_weights:
#     if feature_weights[feature] == float('inf'):
#         feature_weights[feature] = max_weight
# print(feature_weights)
# features_weights = {"Age": 10, "Mortgage": 5, "Income": 5, "Family": 10}


# To display all the outcome
import pandas as pd; pd.set_option('display.max_rows', 1000); pd.set_option('display.max_columns', 1000); pd.set_option('display.width', 1000)

# Making an instance
new_query_instance  = {'Age':45,'Experience': 5,'Income': 40,'ZIP.Code': 94574,'Family': 1,'CCAvg': 6.1,'Education': 1,
                   'Mortgage': 160,'Securities.Account': 1,'CD.Account': 1,'Online': 1,'CreditCard': 1}

features_to_vary  = ['Experience','Income','ZIP.Code','CCAvg','Education',
                   'Mortgage','Securities.Account','CD.Account','Online','CreditCard']
# pd.DataFrame([new_query_instance]

# Using method=random for generating CFs
exp = dice_ml.Dice(d, m, method="random")

# Generate CFs
e1 = exp.generate_counterfactuals(pd.DataFrame([new_query_instance]),
                                  total_CFs=5,
                                  desired_class="opposite",
                                  proximity_weight=1.5,
                                  diversity_weight=1.0,
                                  features_to_vary= features_to_vary)
e1.visualize_as_dataframe(show_only_changes=True)