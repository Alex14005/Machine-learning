#Se importan librerias generales
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

#Se importan los modulos especificos
#Se importa PCA y IPCA
from sklearn.decomposition import KernelPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df_heart = pd.read_csv('./data/heart.csv')

    print(df_heart.head(5))

    #Se dividen entre features y la variable a predecir
    df_features = df_heart.drop(['target'], axis=1)
    df_target = df_heart['target']

    #Se transforman los datos y se ajusta el modelo
    df_features = StandardScaler().fit_transform(df_features)
    
    #Se preparan los datos como train y test
    X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.3, random_state=42)

    #Se crea kpca
    kpca = KernelPCA(n_components=4, kernel='poly')
    kpca.fit(X_train)

    df_train = kpca.transform(X_train)
    df_test = kpca.transform(X_test)

    logistic = LogisticRegression(solver='lbfgs')

    logistic.fit(df_train, y_train)
    print("SCORES KPCA: ", logistic.score(df_test, y_test))