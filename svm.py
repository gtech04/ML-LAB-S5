import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,
confusion_matrix
from sklearn.svm import SVC
iris = datasets.load_iris()
X = iris.data
y = iris.target
mask = y < 2
X = X[mask]
y = y[mask]
X = X[:, [2, 3]]
y_labels = np.where(y == 0, 'Iris-setosa', 'Iris-versicolor')
x_train, x_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.3,
random_state=42)
svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(x_train, y_train)
y_pred_svm = svm_classifier.predict(x_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
report_svm = classification_report(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm * 100:.2f}%")
print("\nSVM Classification Report:\n", report_svm)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Greens',
xticklabels=np.unique(y_labels), yticklabels=np.unique(y_labels))
plt.title('SVM Confusion Matrix')plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
def plot_svm_hyperplane(clf, X, y):
plt.figure(figsize=(10, 6))
labels = np.unique(y)
colors = ['red', 'green']
for i, label in enumerate(labels):
plt.scatter(X[y == label, 0], X[y == label, 1], color=colors[i], label=label,
edgecolors='k')
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 200)
yy = np.linspace(ylim[0], ylim[1], 200)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
plt.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'])
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
s=100, linewidth=1, facecolors='none', edgecolors='k',
label='Support Vectors')
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("SVM Hyperplane, Margins, and Support Vectors")
plt.legend()
plt.grid(True)
plt.show()
plot_svm_hyperplane(svm_classifier, x_train, y_trai

