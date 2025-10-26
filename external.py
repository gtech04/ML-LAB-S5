#factorials
def fact(a):
    if a == 1:
        return a
    return fact(a - 1) * a
n = int(input("Enter a number: ")) 
print(fact(n))
#prime
n=int(input("Enter a number:")) 
if n<=1:
    print("Not a prime number")
else:
    flag=0
    for i in range(2,n-1):
        if n%i==0:
            flag=1
            break
    if flag==0:
        print("Prime number")
    else:
        print("Not a prime number")
#union inter
  def union_intersection(lst1,lst2):     
  union=list(set(lst1)|set(lst2))     
  intersection=list(set(lst1)&set(lst2))     
  return union,intersection 
nums1=[1,2,3,4,5]
nums2=[3,4,5,6,7,8]
print("Original lists:")
print(nums1)
print(nums2)
result=union_intersection(nums1,nums2)
print("\nUnion od said two lists:") 
print(result[0])
print("\nIntersection of said two lists:") 
print(result[1])   
#occurance
def word_counter(str):
    counts=dict()
    words=str.split()
    for word in words:
        if word in counts:
            counts[word]+=1
        else:
            counts[word]=1
    return counts
print(word_counter('the quick brown fox jumps over the lazy dog'))
#multi
def multiply_matrices(matrix1, matrix2):    
    rows1 = len(matrix1)
    cols1 = len(matrix1[0])
    rows2 = len(matrix2)
    cols2 = len(matrix2[0])
    if cols1 != rows2:
        return "Matrix multiplication not possible"
    result = [[0 for _ in range(cols2)] for _ in range(rows1)]
    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    return result
matrix1 = [
    [1, 2, 3],
    [4, 5, 6]
]
matrix2 = [
    [7, 8],
    [9, 10],
    [11, 12]
]
result_matrix = multiply_matrices(matrix1, matrix2)
if isinstance(result_matrix, str):
    print(result_matrix)
else:
    for row in result_matrix:
        print(row)
#frequency 
      file = open("gfg.txt", "r")
frequent_word = ""
frequency = 0
words = []
for line in file:
    line_word = line.lower().replace(',', '').replace('.', '').split()
    for w in line_word:
        words.append(w)
for i in range(len(words)):
    count = 1
    for j in range(i + 1, len(words)):
        if words[i] == words[j]:
            count += 1
    if count > frequency:
        frequency = count
        frequent_word = words[i]

print("Most repeated word: " + frequent_word)
print("Frequency: " + str(frequency))
file.close()

#single reg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("/content/Boston.csv")
x = df[['zn']] #change the zn & indus according to any column heading of the csv file
y = df['indus']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²): {r2:.2f}")
print(f"Accuracy = {model.score(x_test, y_test) * 100:.2f}%")
print(f"Regression Equation: MEDV = {model.intercept_:.2f} + {model.coef_[0]:.2f} * RM")

plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='red', label='Predicted Values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal Fit')
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.show()

#multi reg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("/content/Boston.csv")
x = df[['zn', 'chas']] #change the zn & indus, chas according to any column heading of the csv file
y = df['indus']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²) Score: {r2:.2f}")
print("Accuracy =", model.score(x_test, y_test) * 100)

plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='red', label='Predicted Values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal Fit')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()

#poly reg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv(r"/content/Boston.csv")
x = df[['chas']]
y = df['indus']#change the zn & indus according to any column heading of the csv file
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
poly = PolynomialFeatures(degree=3)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)
model = LinearRegression()
model.fit(x_train_poly, y_train)
y_pred = model.predict(x_test_poly)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²) Score: {r2:.2f}")

plt.figure(figsize=(10,6))
x_range = np.linspace(x_test.min(), x_test.max(), 100).reshape(-1,1)
x_range_poly = poly.transform(x_range)
y_range = model.predict(x_range_poly)
plt.scatter(x_test, y_test, color='blue', label='Actual Values')
plt.plot(x_range, y_range, color='red', label='Polynomial Regression')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Polynomial Regression')
plt.legend()
plt.show()

#naive
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/content/iris (1).csv', header=None)
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

#decision
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target, columns=['target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree Visualization")
plt.show()

#svm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC

iris = datasets.load_iris()
X = iris.data
y = iris.target
mask = y < 2
X = X[mask]
y = y[mask]
X = X[:, [2, 3]]
y_labels = np.where(y == 0, 'Iris-setosa', 'Iris-versicolor')

x_train, x_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.3, random_state=42)
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
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

def plot_svm_hyperplane(clf, X, y):
    plt.figure(figsize=(10, 6))
    labels = np.unique(y)
    colors = ['red', 'green']
    for i, label in enumerate(labels):
        plt.scatter(X[y == label, 0], X[y == label, 1],
                    color=colors[i], label=label, edgecolors='k')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 200)
    yy = np.linspace(ylim[0], ylim[1], 200)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    plt.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
                linestyles=['--', '-', '--'])
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=100, linewidth=1, facecolors='none', edgecolors='k',
                label='Support Vectors')
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.title("SVM Hyperplane, Margins, and Support Vectors")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_svm_hyperplane(svm_classifier, x_train, y_train)

#kmean
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

X, y = load_iris(return_X_y=True)
kmeans = KMeans(n_clusters = 3, random_state = 2) 
kmeans.fit(X)
pred = kmeans.fit_predict(X)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(X[:,0],X[:,1],c = pred, cmap=cm.Accent) 
plt.grid(True)
for center in kmeans.cluster_centers_: center = center[:2]
plt.scatter(center[0],center[1],marker = '^',c = 'red') 
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")

plt.subplot(1,2,2) 
plt.scatter(X[:,2],X[:,3],c = pred, cmap=cm.Accent) 
plt.grid(True)
for center in kmeans.cluster_centers_:center = center[2:4]
plt.scatter(center[0],center[1],marker = '^',c = 'red') 
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)") 
plt.show()

#ann
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential  
from keras.layers import Dense, Dropout, LeakyReLU, PReLU, ELU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dataset = pd.read_csv('/content/Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1]
y = dataset.iloc[:, -1]

geography = pd.get_dummies(X["Geography"], drop_first=True)
gender = pd.get_dummies(X["Gender"], drop_first=True)
X = pd.concat([X, geography, gender], axis=1)
X.drop(['Geography', 'Gender'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = Sequential()
classifier.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu', input_dim=11))
classifier.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu'))
classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))

classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model_history = classifier.fit(X_train, y_train, validation_split=0.33, batch_size=10, epochs=100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = accuracy_score(y_test, y_pred)
print('The accuracy of the model is', accuracy)

cl_report = classification_report(y_test, y_pred)
print(cl_report)

print(model_history.history.keys())

plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
