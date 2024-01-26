import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
dataset = pd.read_csv('Raisin_Dataset.csv', header = None)
dataset.columns = ['Area', 'MajorAxisLength','MinorAxisLength','Eccentricity','ConvexArea', 'Extent', 'Perimeter', 'Class']
factor = pd.factorize(dataset['Class'])
dataset.Class = factor[0]
definitions = factor[1]
X = dataset.iloc[:,0:7].values
y = dataset.iloc[:,7].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 21)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
reversefactor = dict(zip(range(3),definitions))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)
joblib.dump(classifier, 'randomforestmodel.pkl')
