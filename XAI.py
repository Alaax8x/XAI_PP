# Step : 1 Importing the Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error,confusion_matrix,classification_report

# Step 2 : Reading the Data
data = pd.read_csv('Data/bankloan.csv')
print(data.head())

# Step 3 : Analyzing the Data
print(data.info())

print(data.describe())
data.drop('ID',axis = 'columns', inplace = True)
print(data.head())

# Step 4 : visualizing the Data
#Histogram Representation
data.hist(figsize=(20,15),color = 'blue')
# plt.show()

#Box Plot between Personal Loan v/s Income
# plt.figure(figsize=(5,5))
# sns.boxplot(x='Personal.Loan',y='Income',data = data)
# plt.title('Income vs Personal Loan')
# plt.xlabel('Personal Loan')
# plt.ylabel('Income')
# plt.show()

#Counting Loans
loan_counts = data['Personal.Loan'].value_counts()
print(loan_counts)

#Count plot between approved and not approved loans
# plt.figure(figsize=(8, 6))
# loan_counts.plot(kind='bar', color=['blue', 'orange'])
# plt.title('Count of Approved and Not Approved Loans')
# plt.xlabel('Loan Status')
# plt.ylabel('Count')
# plt.xticks([0, 1], ['Not Approved', 'Approved'], rotation=0)
# plt.grid(axis='y')
# plt.show()

#Box plot of Creditcard v/s Income
# plt.figure(figsize=(8, 6))
# sns.boxplot(x='CreditCard', y='Income', data=data)
# plt.title('Income vs CreditCard')
# plt.xlabel('CreditCard')
# plt.ylabel('Income')
# plt.show()

#Box plot of Online v/s CCAvg
# plt.figure(figsize=(8, 6))
# sns.boxplot(x='Online', y='CCAvg', data=data)
# plt.title('CCAvg vs Online')
# plt.xlabel('Online')
# plt.ylabel('CCAvg')
# plt.show()

#Boxplot of Education v/s CCAvg
# plt.figure(figsize=(8, 6))
# sns.boxplot(x='Education', y='CCAvg', data=data)
# plt.title('CCAvg vs Education')
# plt.xlabel('Education')
# plt.ylabel('CCAvg')
# plt.show()

# Step 5 : Splitting and Training the Data
#Seperating the columns
X = data.drop(columns=['Personal.Loan'])
y = data['Personal.Loan']
#Splitting for training and testing
x1,x2,y1,y2 = train_test_split(X,y,test_size=0.25,random_state=0)
clf = RandomForestClassifier(n_estimators=10, random_state=0)
clf.fit(x1, y1)

#predicting the y
pred = clf.predict(x2)
# ROC Accuracy Score
auc_score = roc_auc_score(y2,pred)
print(f"AUC Score:{auc_score}")
# 0.9694979568009341
#Mean Sqared Error
pred_proba = clf.predict_proba(x2)[:, 1]  # Probability of class 1
auc_score = roc_auc_score(y2, pred_proba)
print(f"AUC Score: {auc_score}")

pred_class = clf.predict(x2)
print("Accuracy:", accuracy_score(y2, pred_class))

# Step 6 : implementing Counterfactual explanations
import numpy as np
from scipy.optimize import minimize


def counterfactual_loss(x, x_original, model, desired_class, lambda_param=0.1):  # Reduced lambda
    x = x.reshape(1, -1)
    y_pred = model.predict_proba(x)[0][desired_class]
    prediction_loss = 1 - y_pred
    distance_loss = np.linalg.norm(x - x_original)
    total_loss = (0.9 * prediction_loss) + (0.1 * distance_loss)  # Make prediction loss more dominant
    return total_loss

# Original instance and desired class
x_original = x2.iloc[0].values
fixed_features = ['Age', 'ZIP.Code', 'Experience']  # Features that cannot be changed

desired_class = 1

# Set bounds to original value for fixed features
bounds = []
for i, col in enumerate(x1.columns):
    if col in fixed_features:
        bounds.append((x_original[i], x_original[i]))  # Force no change
    else:
        bounds.append((max(0, np.percentile(x1[col], 10)), np.percentile(x1[col], 90)))

# Optimization
result = minimize(
    counterfactual_loss,
    x0=x_original,
    args=(x_original, clf, desired_class, 0.1),
    bounds=bounds,
    method='trust-constr',  # Better for constrained optimization
    options={'maxiter': 1000, 'disp': True}  # Display optimization progress
)


counterfactual = result.x
print("\nCounterfactual Features:")
print(pd.DataFrame([counterfactual], columns=x2.columns))

print("\nOriginal Prediction Probabilities:", clf.predict_proba([x_original]))
print("Counterfactual Prediction Probabilities:", clf.predict_proba([counterfactual]))