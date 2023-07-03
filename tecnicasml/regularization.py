import pandas as pd
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    dataset = pd.read_csv('./data/felicidad.csv')
    print(dataset.describe())

    #se divide el dt, en variables y en la variable a predecir
    X = dataset[['gdp', 'family', 'lifexp', 'freedom', 'corruption', 'generosity', 'dystopia']]
    y = dataset[['score']]

    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    model_linear = LinearRegression().fit(X_train, y_train)
    y_predict_linear = model_linear.predict(X_test)
    
    model_lasso = Lasso(alpha=0.02).fit(X_train, y_train)
    y_predict_lasso = model_lasso.predict(X_test)

    modelRdige = Ridge(alpha=1).fit(X_train, y_train)
    y_predict_Ridge = modelRdige.predict(X_test)

    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print('LINEAR LOSS: ', linear_loss)

    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print('LASSO LOSS: ', lasso_loss)

    ridge_loss = mean_squared_error(y_test, y_predict_Ridge)
    print('RIDGE LOSS: ', ridge_loss)

    print("="*32)
    print("Coeficientes lasso")
    print(model_lasso.coef_)

    print("="*32)
    print("Coeficientes Ridge")
    print(modelRdige.coef_)