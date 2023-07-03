#Se importan librerias generales
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

#Se importan los modulos especificos
#Se importa PCA y IPCA
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

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

    print(X_train.shape)
    print(y_train.shape)

    #Se crea PCA la variable pro defecto seria(N_components = min)n_muestras, n_features
    pca = PCA(n_components=3)
    #Se entrena PCA para que se ajuste a los datos que tenemos
    pca.fit(X_train)
    #El IPCA tiene un parametro mas (batch_size), porque no manda a entrenar los datos al mismo tiempo y ese parametro es para que midas el tamaño de bloque a entrenar
    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)

    #Se mide la varianza
    #plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    #plt.show()

    logistic = LogisticRegression(solver='lbfgs')

    #Utilizando PCA
    df_train = pca.transform(X_train)
    df_test = pca.transform(X_test)
    logistic.fit(df_train, y_train)
    print("Score PCA: ", logistic.score(df_test, y_test))

    #Utilizando IPCA
    df_train = ipca.transform(X_train)
    df_test = ipca.transform(X_test)
    logistic.fit(df_train, y_train)
    print("SCORES IPCA: ", logistic.score(df_test, y_test) )