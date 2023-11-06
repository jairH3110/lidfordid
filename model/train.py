from os import PathLike
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump
import pandas as pd
import pathlib

df = pd.read_csv(pathlib.Path('../data/l4d2_player_stats_final2.csv'))
# Separar las características (X) y la variable objetivo (y)
y = df['Playtime']
X = df.drop('Playtime', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

print ('Training model.. ')
clf = RandomForestRegressor(n_estimators = 10,
                            max_depth=2,
                            random_state=0)
clf.fit(X_train, y_train)
print ('Saving model..')

dump(clf, pathlib.Path('../model/lidfordid-disease-v1.joblib'))
