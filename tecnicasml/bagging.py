import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    dt_heart = pd.read_csv('./data/heart.csv')
    print(dt_heart['target'].describe())

    X = dt_heart.drop(['target'], axis=1)
    y = dt_heart['target']

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.35)

    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_pred = knn_class.predict(X_test)
    print('='*64)
    print(accuracy_score(knn_pred, y_test))

    bag_class = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=50).fit(X_train,y_train)
    bag_pred = bag_class.predict(X_test)
    print('='*64)
    print(accuracy_score(bag_pred, y_test))

   
    #Utilizando estimadores base diversos, se agrega el randomforest
   
    knn = KNeighborsClassifier()
    rf = RandomForestClassifier()

    bagging_knn = BaggingClassifier(base_estimator=knn, n_estimators=50)
    bagging_rf = BaggingClassifier(base_estimator=rf, n_estimators=50)

    voting = VotingClassifier(estimators=[('bagging_knn', bagging_knn), ('baggging_rf', bagging_rf)]).fit(X_train, y_train)
    

    voting_pred = voting.predict(X_test)
    print('='*64)
    print('Utilizando estimadores base diversos:',  accuracy_score(voting_pred, y_test))

    
    
    #Ajuste de hiperparámetros del estimador base

    from sklearn.model_selection import GridSearchCV

    bagging_grid = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=50)

    # Definir los hiperparámetros a ajustar

    param_grid = {
        'base_estimator__n_neighbors': [3,5,7],
        'base_estimator__weights': ['uniform', 'distance'],
        'base_estimator__metric': ['euclidean', 'manhattan']
    }

    #Realizar la busqueda de cuadricula
    grid_search = GridSearchCV(bagging_grid, param_grid, cv=5).fit(X_train,y_train)

    #Obtener le mejor clasificador
    best_begging = grid_search.best_estimator_
    print(best_begging)

    #Se realiza la prediccion
    best_bagging_predict = best_begging.predict(X_test)

    #Se evalua la prediccion
    print('Utilizando por votacion seria: ', accuracy_score(best_bagging_predict, y_test))