# Step : 1 Importing the Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
import dice_ml  # Importing DiCE for Counterfactual Explanations
from dice_ml import Dice



# Step 2 : Reading the Data
data = pd.read_csv('Data/bankloan.csv')
print(data.head())

# Step 3 : Analyzing the Data
print(data.info())
print(data.describe())
data.drop('ID', axis='columns', inplace=True)
print(data.head())

# Step 4 : Visualizing the Data
# Histogram Representation
data.hist(figsize=(20, 15), color='blue')
plt.show()

# Box Plot between Personal Loan v/s Income
plt.figure(figsize=(5, 5))
sns.boxplot(x='Personal.Loan', y='Income', data=data)
plt.title('Income vs Personal Loan')
plt.xlabel('Personal Loan')
plt.ylabel('Income')
plt.show()

# Counting Loans
loan_counts = data['Personal.Loan'].value_counts()
print(loan_counts)

# Count plot between approved and not approved loans
plt.figure(figsize=(8, 6))
loan_counts.plot(kind='bar', color=['blue', 'orange'])
plt.title('Count of Approved and Not Approved Loans')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.xticks([0, 1], ['Not Approved', 'Approved'], rotation=0)
plt.grid(axis='y')
plt.show()

# Step 5 : Splitting and Training the Data
X = data.drop(columns=['Personal.Loan'])
y = data['Personal.Loan']

x1, x2, y1, y2 = train_test_split(X, y, test_size=0.25, random_state=0)
r = RandomForestRegressor(n_estimators=10, random_state=0)
r.fit(x1, y1)

# Predicting the y
pred = r.predict(x2)
auc_score = roc_auc_score(y2, pred)
print(f"AUC Score: {auc_score}")

mse = mean_squared_error(y2, pred)
print(f"MSE Score: {mse}")

# Step 6: Counterfactual Explanations using DiCE
# Converting data into a format suitable for DiCE
d = dice_ml.Data(dataframe=data, continuous_features=['Income', 'CCAvg'], outcome_name='Personal.Loan')

# Defining the model for DiCE
m = dice_ml.Model(model=r, backend="sklearn")

# Creating a DiCE explainer
dice_exp = Dice(d, m)

# Generating counterfactual examples
query_instance = x2.iloc[0]  # Selecting a sample instance
dice_exp.generate_counterfactuals(query_instance, total_CFs=2, desired_class=1).visualize_as_dataframe()
