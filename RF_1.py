# Step : 1 Importing the Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error,confusion_matrix,classification_report
from sklearn.metrics import roc_auc_score


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

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
plt.show()

#Box Plot between Personal Loan v/s Income
plt.figure(figsize=(5,5))
sns.boxplot(x='Personal.Loan',y='Income',data = data)
plt.title('Income vs Personal Loan')
plt.xlabel('Personal Loan')
plt.ylabel('Income')
plt.show()

#Counting Loans
loan_counts = data['Personal.Loan'].value_counts()
print(loan_counts)

#Count plot between approved and not approved loans
plt.figure(figsize=(8, 6))
loan_counts.plot(kind='bar', color=['blue', 'orange'])
plt.title('Count of Approved and Not Approved Loans')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.xticks([0, 1], ['Not Approved', 'Approved'], rotation=0)
plt.grid(axis='y')
plt.show()

#Box plot of Creditcard v/s Income
plt.figure(figsize=(8, 6))
sns.boxplot(x='CreditCard', y='Income', data=data)
plt.title('Income vs CreditCard')
plt.xlabel('CreditCard')
plt.ylabel('Income')
plt.show()

#Box plot of Online v/s CCAvg
plt.figure(figsize=(8, 6))
sns.boxplot(x='Online', y='CCAvg', data=data)
plt.title('CCAvg vs Online')
plt.xlabel('Online')
plt.ylabel('CCAvg')
plt.show()

#Boxplot of Education v/s CCAvg
plt.figure(figsize=(8, 6))
sns.boxplot(x='Education', y='CCAvg', data=data)
plt.title('CCAvg vs Education')
plt.xlabel('Education')
plt.ylabel('CCAvg')
plt.show()

# Step 5 : Splitting and Training the Data
#Seperating the columns
X = data.drop(columns=['Personal.Loan'])
y = data['Personal.Loan']
#Splitting for training and testing
x1,x2,y1,y2 = train_test_split(X,y,test_size=0.25,random_state=0)
r = RandomForestClassifier(n_estimators = 10, random_state = 0)
r.fit(x1,y1)

# RandomForestRegressor
RandomForestRegressor(n_estimators=10, random_state=0)
#predicting the y
pred = r.predict(x2)
# ROC Accuracy Score
auc_score = roc_auc_score(y2,pred)
print(f"AUC Score:{auc_score}")
# 0.9694979568009341
#Mean Sqared Error
mse = mean_squared_error(y2,pred)
print(f"MSE Score:{mse}")
# 0.013360000000000002

# Building GUI interface:
def PersonalLoan(Age, Experience, Income, ZIPCode, Family, CCAvg, Education, Mortgage, SecuritiesAccount, CDAccount, Online,
           CreditCard):

    # Map "Yes"/"No" inputs to numerical values
    SecuritiesAccount = 1 if SecuritiesAccount == "Yes" else 0
    CDAccount = 1 if CDAccount == "Yes" else 0
    Online = 1 if Online == "Yes" else 0
    CreditCard = 1 if CreditCard == "Yes" else 0

    # Convert all inputs to numpy array
    x = np.array([Age, Experience, Income, ZIPCode, Family, CCAvg, Education, Mortgage, SecuritiesAccount, CDAccount, Online,
           CreditCard], dtype=float).reshape(1, -1)

    # Make prediction
    prediction = r.predict(x)

    # Convert 0/1 to "Declined"/"Approved"
    result = "Approved" if prediction[0] == 1 else "Declined"
    # Custom message based on prediction
    message = (f"""Your loan application is **{result}**. Thank you for using our service.
               \nThe reason you got this result:
                \n \"*the explanation*\"""")
    return result, message


# Create the Gradio app
with gr.Blocks() as app:
    # Add a message at the top
    gr.Markdown("""
    # Loan Approval Predictor  
    **Important:**  
    Please provide accurate and truthful information to get the best prediction.  
    This tool is for **educational purposes only** and does not guarantee actual loan approval.
    """)

    # Define the input fields
    inputs = [
        gr.Number(label="Age", info='How old are you?'),
        gr.Number(label="Experience", info='How many loan experience do you have?'),
        gr.Slider(minimum=0, maximum=500000, label="Income", info='How much is your income annually?'),
        gr.Number(label="ZIPCode", info='What year is your ZIP Code?'),
        gr.Number(value=2, label="Family", info='How many family members do you have?'),
        gr.Slider(minimum=0, maximum=10, label="CCAvg", info='What is your CCAvg?'),
        gr.Radio(choices=["1", "2", "3"], label="Education", info='Select your education level (1: Undergraduate, 2: Graduate, 3: Advanced/Professional).'),
        gr.Number(label="Mortgage", info='How much is your Mortgage?'),
        gr.Radio(choices=["Yes", "No"], label="Securities Account", info='Do you have a Securities Account?'),
        gr.Radio(choices=["Yes", "No"], label="CD Account", info='Do you have a CD Account?'),
        gr.Radio(choices=["Yes", "No"], label="Online Banking", info='Do you have an Online Banking Account?'),
        gr.Radio(choices=["Yes", "No"], label="Credit Card", info='Do you have a Credit Card Account?')
    ]

    # Define the outputs
    result_output = gr.Textbox(label="The Loan Approval Prediction:")
    message_output = gr.Markdown()

    # Add a submit button and connect inputs/outputs
    submit_btn = gr.Button("Submit")
    submit_btn.click(PersonalLoan, inputs=inputs, outputs=[result_output, message_output])

# Launch the app
app.launch(share=True)
