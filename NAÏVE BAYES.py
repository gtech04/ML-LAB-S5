import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('iris.data.csv', header=None)
df = df.drop(columns=['Id']) 
print(df.shape)
print(df.head(10))
print(df)

df.info()

df.isnull().sum()

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred = nb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print (f'Accuracy: {accuracy * 100 :.2f}%' )
print(f'\nClassification Report:\n{report}')
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
xticklabels=['Setosa', 'Versicolor', 'Virginica'],
yticklabels=['Setosa', 'Versicolor', 'Virginica' ])
plt.title('Confusion Matrix' )
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
