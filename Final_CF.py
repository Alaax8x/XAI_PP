## Importing the Library ##
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import dice_ml
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import resample
from sklearn import metrics
import category_encoders as ce
from sklearn.model_selection import cross_val_score


## Reading the Data ##
data = pd.read_csv('Data/bankloan.csv')
print(data.head())

## Analyzing the Data ##
print(data.info())
print(data['Experience'].describe())
data.drop('ID', axis='columns', inplace=True)
print(data.head())


## Balancing Data ##
minority_class = data[data['Personal.Loan'] == 1]
majority_class = data[data['Personal.Loan'] == 0]

# Up sample the minority class
minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)

# Combine the upsampled minority class with the majority class
balanced_data = pd.concat([majority_class, minority_upsampled])
print(balanced_data["Personal.Loan"].value_counts())

## Splitting and Training the Data ##
target = balanced_data["Personal.Loan"]
balanced_dataX = balanced_data.drop('Personal.Loan', axis=1)

x_train, x_test, y_train, y_test = train_test_split(balanced_dataX,
                                                                target,
                                                                test_size=0.25,
                                                                random_state=0,
                                                                stratify=target,
                                                                shuffle=True)

# dice_ml.Data
d = dice_ml.Data(dataframe=balanced_data,
                 continuous_features=['Age', 'Experience', 'ZIP.Code',
                                      'CCAvg', 'Mortgage', 'Income', 'Family'],
                 outcome_name='Personal.Loan')

## Encoding data ##
numerical = ['Age', 'Experience', 'ZIP.Code', 'CCAvg', 'Mortgage', 'Income', 'Family']
categorical = x_train.columns.difference(numerical)

encoder = ce.OrdinalEncoder(cols= categorical)

# encode the train and test data
x_train = encoder.fit_transform(x_train)
x_test = encoder.transform(x_test)

##  Building the Model ##

# fit the model
model = RandomForestClassifier(max_depth=5,random_state=0)
model.fit(x_train, y_train)

## Model evaluation ##
model_predict = model.predict(x_test)

# Creating an evaluation report
print(classification_report(y_test, model_predict))

# Create a confusion matrix using sklearn
conf_matrix = metrics.confusion_matrix(y_test, model_predict, labels=[1, 0])

# Plot the confusion matrix using seaborn
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='PiYG',
            xticklabels=['Approved','Declined'], yticklabels=['Approved','Declined'])
plt.xlabel('Predicted Preference', fontsize=12)
plt.ylabel('Actual Preference', fontsize=12)
plt.show()

# Extract feature importance and sort them
importances = model.feature_importances_
feature_names = x_train.columns
sorted_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

# Convert to lists for plotting
features, importance_values = zip(*sorted_features)

# Create the plot
plt.figure(figsize=(10, 6))
sns.barplot(x=importance_values, y=features, palette="viridis")

# Add labels and title
plt.xlabel("Feature Importance Score", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.title("Feature Importance in Loan Approval Model", fontsize=14)
plt.show()


# Using sklearn backend
m = dice_ml.Model(model=model, backend="sklearn")

# To display all the outcome
import pandas as pd; pd.set_option('display.max_rows', 1000); pd.set_option('display.max_columns', 1000); pd.set_option('display.width', 1000)

## Making a new instance ##

features_to_vary  = ['Experience','Income','CCAvg','Education',
                     'Securities.Account','CD.Account','Online','CreditCard']

new_query_instance  = {'Age':45,'Experience': 5,'Income': 40,'ZIP.Code': 94574,'Family': 1,'CCAvg': 6.1,'Education': 1,
                   'Mortgage': 160,'Securities.Account': 1,'CD.Account': 1,'Online': 1,'CreditCard': 1}

## Generating counterfactual ##
# Using method=random for generating CFs
exp = dice_ml.Dice(d, m, method="genetic")

# Generate CFs
e1 = exp.generate_counterfactuals(pd.DataFrame([new_query_instance]),
                                  total_CFs=2,
                                  desired_class="opposite",
                                  features_to_vary= features_to_vary,
                                  permitted_range={'Experience': [0, 50]},
                                  proximity_weight= 1.0,
                                  diversity_weight= 1.0)
e1.visualize_as_dataframe(show_only_changes=True)

## Checking for overfitiing ##
scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
print("Cross-validation accuracy:", scores.mean())