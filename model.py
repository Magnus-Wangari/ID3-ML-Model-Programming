from IPython.display import Image
import pydotplus
import math
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import tree, metrics
import networkx as nx
import matplotlib.pyplot as plt
import operator
from sklearn.model_selection import train_test_split




data=pd.read_csv('data.csv',names=['engine','turbo','fueleco','weight','fast'])
#data.head()
data.info() #view details about imported data

data['fast'],class_names = pd.factorize(data['fast'])
print(class_names)
print(data['fast'].unique())


#factorize data to interger format
data['engine'],_ = pd.factorize(data['engine'])
data['turbo'],_ = pd.factorize(data['turbo'])
data['weight'],_ = pd.factorize(data['weight'])
data['fueleco'],_ = pd.factorize(data['fueleco'])

#print the intergers optional
print(data)
#print(data['engine'].unique())
#print(data['turbo'].unique())
#print(data['weight'].unique())
#print(data['fueleco'].unique())
#data.head()

data.info() #check the data info should be converted to intergers 

X = data.iloc[:,:-1]
y = data.iloc[:,-1]


# split data randomly into 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# train the decision tree
dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
dtree.fit(X_train, y_train)

# use the model to make predictions with the test data
y_pred = dtree.predict(X_test)

# how did our model perform?
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))

print(y_pred)
import graphviz


feature_names = X.columns
dot_data = tree.export_graphviz(dtree, out_file=None, filled=True, rounded=True,
                                feature_names=feature_names,  
                                class_names=class_names)
graph = graphviz.Source(dot_data) 
#graph.format='png' 
print(graph)

#confusion matrix

cm=confusion_matrix(y_test ,y_pred)
print(cm)

#confusionmatrix with more details

cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(cm)


#just added another evaluation called classification report 

#from sklearn.metrics import classification_report

#print(classification_report(y_test, y_pred))