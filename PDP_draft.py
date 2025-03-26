## Importing the Library ##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import resample
from sklearn import metrics
import category_encoders as ce


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

import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay


pdp_age = PartialDependenceDisplay.from_estimator(model, x_train, features=[2, 5, 6])
plt.show()


import os
from sklearn.tree import export_graphviz
import six
import pydot

col = ['Age','Experience','Income','ZIP.Code','Family','CCAvg','Education',
                   'Mortgage','Securities.Account','CD.Account', 'Online', 'CreditCard']

#modelname.feature_importance_
y = model.feature_importances_
#plot
fig, ax = plt.subplots()
width = 0.4 # the width of the bars
ind = np.arange(len(y)) # the x locations for the groups
ax.barh(ind, y, width, color="green")
ax.set_yticks(ind+width/10)
ax.set_yticklabels(col, minor=False)
plt.title('Feature importance in RandomForest Classifier')
plt.xlabel('Relative importance')
plt.ylabel('feature')
plt.figure(figsize=(5,5))
fig.set_size_inches(6.5, 4.5, forward=True)
plt.show()

from sklearn import tree
# dotfile = six.StringIO()
# i_tree = 0
# for tree_in_forest in model.estimators_:
#     export_graphviz(tree_in_forest,out_file='tree.dot',
#     feature_names=col,
#     filled=True,
#     rounded=True)
#     (graph,) = pydot.graph_from_dot_file('tree.dot')
#     name = 'tree' + str(i_tree)
#     graph.write_png(name+  '.png')
#     os.system('dot -Tpng tree.dot -o tree.png')
#     i_tree +=1

