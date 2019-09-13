# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 16:27:34 2019

@author: yeeya
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import mglearn
import graphviz
import numpy as np
from sklearn.tree.export import export_graphviz
import matplotlib.pyplot as plt
%matplotlib inline


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)


## LOGISTIC REGRESSION
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

##DESICION TREE

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)

print('Accuracy on the training subset: {:.3f}'.format(tree.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(tree.score(X_test, y_test)))

# Since the score is 1 the model is overfitting.. Now we set max depth to 4 so that it doesn't overfit

tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)

print('Accuracy on the training subset: {:.3f}'.format(tree.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(tree.score(X_test, y_test)))


export_graphviz(tree, out_file='cancertree.dot', class_names=['malignant', 'benign'], feature_names=cancer.feature_names,
               impurity=False, filled=True)

n_features = cancer.data.shape[1]
plt.barh(range(n_features), tree.feature_importances_, align='center')
plt.yticks(np.arange(n_features), cancer.feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()

## RANDOM FOREST

forest = RandomForestClassifier(n_estimators=100,random_state=0)
forest.fit(X_train,y_train)
print('Accuracy on the training subset: {:.3f}'.format(forest.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(forest.score(X_test, y_test)))

# Again, it's overfitting.. 

n_features = cancer.data.shape[1]
plt.barh(range(n_features), forest.feature_importances_, align='center')
plt.yticks(np.arange(n_features), cancer.feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()

# Now more features play a role (nonzero importance) in the decision making!

# Visualization of each 100 trees:
i_tree = 0
for tree_in_forest in forest.estimators_:
    with open('tree_' + str(i_tree) + '.dot', 'w') as cancertree2:
        cancertree2 = export_graphviz(tree_in_forest, out_file = 'cancertree2.dot')
    i_tree = i_tree + 1
    
