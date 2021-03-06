
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
import pandas as pd



iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf.fit(iris.data, iris.target)

dot_data = tree.export_graphviz(clf, out_file = None)
graph = graphviz.Source(dot_data)
graph.render('iris')


print(clf.predict(iris.data[:1, :]))

