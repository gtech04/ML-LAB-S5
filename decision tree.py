import pandas as pd 
from sklearn import datasets
from sklearn.tree import Decision TreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
iris = datasets.load_iris()
X= pd. pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target, columns=['target'])
X_ train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = Decision TreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: (accuracy}")
print(classification_reporty_test, y_pred))
print/confusion_matrixy_test, y_pred))
plt. figure(figsize=(12, 8))
plot tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree Visualization")
plt.show()
