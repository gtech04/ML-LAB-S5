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

dataset = pd.read_csv('Churn_Modelling.csv')
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

#k-mean

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans

data = pd.read_csv("/content/cluster_data.csv")

X = data.values

kmeans = KMeans(n_clusters=3, random_state=2)
pred = kmeans.fit_predict(X)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=pred, cmap=cm.Accent)
plt.grid(True)
for center in kmeans.cluster_centers_:
    plt.scatter(center[0], center[1], marker='^', c='red')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

if X.shape[1] >= 4:
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 2], X[:, 3], c=pred, cmap=cm.Accent)
    plt.grid(True)
    for center in kmeans.cluster_centers_:
        plt.scatter(center[2], center[3], marker='^', c='red')
    plt.xlabel("Feature 3")
    plt.ylabel("Feature 4")

plt.show()

#decisiom id3

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

data = pd.read_csv("/content/diabetes_dataset.csv")

target_column = "Age"   

X = data.drop(columns=[target_column])
y = data[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=[str(c) for c in y.unique()], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

#naive

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

#regresion (single)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
df=pd.read_csv("boston.csv")
x=df[['RM']]
y=df['MEDV']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"Mean Squared Error(MSE):{mse:.2f}")
print(f"Mean Absolute Error(MAE):{mae:.2f}")
print(f"R-squared (R**2) Score: {r2:.2f}")
print("Accuracy=",model.score(x_test,y_test)*100)
plt.figure(figsize=(10,6))
plt.scatter(y_test,y_test,color='blue',label='Actual Values')
plt.scatter(y_test,y_pred,color='red',label='Predicted values')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'k--',lw=2) plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted values') plt.legend()
plt.show()

#regresion (multi)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
df=pd.read_csv("boston.csv")
x=df[['RM','AGE']]
y=df['MEDV']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"Mean Squared Error(MSE):{mse:.2f}")
print(f"Mean Absolute Error(MAE):{mae:.2f}")
print(f"R-squared (R**2) Score: {r2:.2f}")
print("Accuracy=",model.score(x_test,y_test)*100)
plt.figure(figsize=(10,6))
plt.scatter(y_test,y_test,color='blue',label='Actual Values')
plt.scatter(y_test,y_pred,color='red',label='Predicted values')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'k--',lw=2) plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted values') plt.legend()
plt.show()

#regresion (poly)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
df=pd.read_csv(r"boston.csv")
x=df[['RM']]
y=df['MEDV']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
poly=PolynomialFeatures(degree=3)
x_train_poly=poly.fit_transform(x_train)
x_test_poly=poly.transform(x_test)
model=LinearRegression()
model.fit(x_train_poly,y_train)
y_pred=model.predict(x_test_poly)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"Mean Squared Error(MSE): {mse:.2f}")
print(f"Mean Absolute Error(MAE): {mae:.2f}")
print(f"R-square(R**2) score: {r2:.2f}")
plt.figure(figsize=(10,6))
x_range=np.linspace(x_test.min(),x_test.max(),100).reshape(-1,1) x_range_poly=poly.transform(x_range)
y_range=model.predict(x_range_poly)
plt.scatter(x_test,y_test,color='blue',label='Actual values')
plt.scatter(x_range,y_range,color='red',label='Polynomial Regression') plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Polynomial Regression') plt.legend()
plt.show()

#factorial

def fact(a):
    if a==1:
        return a
    return fact(a-1)*a
n=int(input("Enter a number:")) 
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

#union_intersection

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

#word_occurance

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

#multiply_matrices

def multiply_matrices(matrix1,matrix2):    
    rows1=len(matrix1)
    cols1=len(matrix1[0])
    rows2=len(matrix2)
    cols2=len(matrix2[0])
    if cols1!=rows2:
        return "Matrix multiplication not possible"     
      result=[[0 for _ in range(cols2)]
              for _ in range(rows1)]
    for i in range (rows1):
        for j in range (cols2):
            for k in range (cols1):
                result[i][j]+=matrix1[i][k]*matrix2[k][j]     
              return result
matrix1=[
    [1,2,3],
    [4,5,6]
]
matrix2=[
    [7,8],
    [9,10],
    [11,12]
]
result_matrix=multiply_matrices(matrix1,matrix2) 
if isinstance (result_matrix,str):
    print(result_matrix)
else:
    for row in result_matrix:         
      print(row)

#frequency

file=open("gfg.txt","r")
frequent_word=""
frequency=0
words=[]
for line in file:  
MIline_word=line.lower().replace(',',").replace('.',").split(" ")
    for w in line_word:
        words.append(w)
for i in range(0,len(words)):
    count=1;
    for j in range(i+1,len(words)):
        if(words[i]==words[j]):
            count=count+1
    if(count>frequency):
        frequency=count
        frequent_word=words[i]
print("Most repeated word:"+frequent_word) 
print("Frequency:"+str(frequency))
file.close()
