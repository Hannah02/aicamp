#imports
import numpy as np
import pandas as pd
import seaborn as sns

from datetime import datetime

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
#loading in data for avocado prediction
avocado = pd.read_csv('avocado.csv')
avocado.head(5)
#cleaning our data
#selecting avocado type

avo_type = 'organic'
avocado = avocado[avocado.type == avo_type]

#avocado['Date']= pd.to_datetime(avocado['Date'], format='%Y-%m-%d')

avocado = avocado.dropna(axis=0)

avocado['Year'] = [d.split('-')[0] for d in avocado['Date']]
avocado['Month'] = [d.split('-')[1] for d in avocado['Date']]
avocado['Day'] = [d.split('-')[2] for d in avocado['Date']]

avocado = avocado[['Year', 'Month', 'Day', 'AveragePrice', 'region', 'Total Bags']]

print(avocado.head(5))
#avocado = avocado.drop(columns='region')

avocado['region_id'] = avocado['region'].factorize()[0]
avocado.head()
avocado.head()
#Splitting Data using Train_Test_Split
X = avocado.drop(columns=['AveragePrice', 'region'])
y = avocado['AveragePrice'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
X_train[:5]
y_train
print X_train.shape
#One hot encoding to convert region to numbers
#targets = pd.Series(y_train)
#one_hot = pd.get_dummies(targets, sparse=True)
#y_train = np.asarray(one_hot)

#One hot encoding to convert region to numbers
#targets = pd.Series(y_test)
#one_hot = pd.get_dummies(targets, sparse=True)
#y_test = np.asarray(one_hot)




model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)
predictions=model.predict(X_test)
#Using a RandomForestRegressor to train the model

#LinearRegression
#DecisionTreeRegressor
#KNeighborsRegressor
#DecisionTreeRegressor, RandomForestRegressor -> GridSearch

#Regression Models -> Scikit Learn, Keras?

clf = make_pipeline(StandardScaler(), RandomForestRegressor(random_state=1, n_estimators=50))

scores = cross_val_score(clf, X_train, y_train, cv=3, n_jobs=-1)
#print(f"{round(np.mean(scores),3)*100}%")
print(round(np.mean(scores),3)*100)

clf.fit(X_train,y_train)

print(mean_squared_error(y_pred=clf.predict(X_test), y_true=y_test))

fig = plt.figure()
sns.lineplot(x = 'Day', y = 'AveragePrice', data = avocado)