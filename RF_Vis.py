from itertools import count
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import category_encoders as ce
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
import graphviz
from supertree import SuperTree


# Load Data
data = pd.read_csv('data/bankloan.csv')
data.drop('ID', axis='columns', inplace=True)
data.drop('ZIP.Code', axis='columns', inplace=True)

# Balance the Data
minority_class = data[data['Personal.Loan'] == 1]
majority_class = data[data['Personal.Loan'] == 0]
minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
balanced_data = pd.concat([majority_class, minority_upsampled])

# Split Data
target = balanced_data["Personal.Loan"]
balanced_dataX = balanced_data.drop('Personal.Loan', axis=1)
x_train, x_test, y_train, y_test = train_test_split(
    balanced_dataX, target, test_size=0.25, random_state=0, stratify=target, shuffle=True
)

# Encode Categorical Data
numerical = ['Age', 'Experience', 'CCAvg', 'Mortgage', 'Income', 'Family']
categorical = x_train.columns.difference(numerical)
encoder = ce.OrdinalEncoder(cols=categorical)
x_train = encoder.fit_transform(x_train)
x_test = encoder.transform(x_test)

# Train Model
model = RandomForestClassifier(max_depth=5, random_state=0)
model.fit(x_train, y_train)
## Model evaluation ##
model_predict = model.predict(x_test)
print(classification_report(y_test, model_predict))


## Visualisation ##

features = balanced_dataX.columns
target = 'Personal.Loan'
class_names = ['Declined', 'Approved']
print(count(len(features)))
print(class_names)
# Plot the tree using the plot_tree function from sklearn
tree = model.estimators_[0]


############# Plot_tree
plt.figure(figsize=(20,10))  # Set figure size to make the tree more readable
plot_tree(tree,
          feature_names=features,  # Use the feature names from the dataset
          class_names=class_names,
          filled=True,              # Fill nodes with colors for better visualization
          rounded=True)             # Rounded edges for nodes
plt.title("Decision Tree from the Random Forest: Plot_tree")
plt.show()

############ graphviz
dot_data = export_graphviz(tree, out_file=None, feature_names=features, class_names=class_names, filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("RF_Vis/Dtree/decision_tree1", format= "png")  # Saves the tree as a file

############# Supertree
y_train = y_train.reset_index(drop=True)  # Reset index, drop old index
st = SuperTree(
    model,  # Extract a single tree
    x_train,
    y_train,
    list(features),  # Convert Index to a normal list,
    class_names
)

st.save_html("RF_Vis/Dtree/random_forest_tree.html")
