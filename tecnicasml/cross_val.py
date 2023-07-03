import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import (
    cross_val_score, KFold
)
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':

    dataset = pd.read_csv('./data/felicidad.csv')

    X = dataset.drop(['country', 'score'], axis=1)
    y = dataset['score']

    mse_values = []

    model = DecisionTreeRegressor()
    score = cross_val_score(model, X,y, cv=3, scoring='neg_mean_squared_error')
    print(np.abs(np.mean(score)))

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for train, test in kf.split(dataset):
        x_train = pd.DataFrame(columns=list(X),index=range(len(train)))
        x_test = pd.DataFrame(columns=list(X),index=range(len(test)))
        y_train = pd.DataFrame(columns=['score'],index=range(len(train)))
        y_test = pd.DataFrame(columns=['score'],index=range(len(test)))
        for i in range(len(train)):
            x_train.iloc[i] = X.iloc[train[i]]
            y_train.iloc[i] = y.iloc[train[i]]
        for j in range(len(test)):
            x_test.iloc[j] = X.iloc[test[j]]
            y_test.iloc[j] = y.iloc[test[j]]
        model = DecisionTreeRegressor().fit(x_train,y_train)
        predict = model.predict(x_test)
        mse_values.append(mean_squared_error(y_test,predict))
        
    print("Los tres MSE fueron: ",mse_values)
    print("El MSE promedio fue: ", np.mean(mse_values))