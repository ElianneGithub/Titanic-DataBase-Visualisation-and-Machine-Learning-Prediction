#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 21:51:04 2020

@author: joelda
"""

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from subprocess import check_output

titanic = pd.read_csv("titanic.csv")

titanic = titanic.drop(['Cabin','Ticket','Name'],axis=1)
titanic = titanic.dropna()

# Transformons les données non numérique en données numériques


titanic['Embarked'] = titanic['Embarked'].map( {'S': 1, 'C': 2, 'Q': 3} ).astype(int)
titanic['Sex'] = titanic['Sex'].map( {'male': 1, 'female': 2} ).astype(int)

# Important tous les modèles dont nous auront besoin

from sklearn.linear_model import LogisticRegression  
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier  
from sklearn import svm  
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier 

from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

""" Nous allons utiliser la classification pour construire des modèles """

""" Regardons d'abord grâce à la visualization suivante les relations de correlations entre les differentes données de manière globale"""

plt.figure(figsize=(7,4)) 
sns.heatmap(titanic.corr(),annot=True,cmap='cubehelix_r') 
plt.show()

# Pour commencer nos modèles nous allons d'abord séparer la dataset en deux : train et test, l'attribut test_size=0.3 permet de séparer les données autour d'un ratio de train = 70% et test = 30%

train, test = train_test_split(titanic, test_size = 0.3)

X_train = train.drop("Survived", axis=1) # Nous prenons les colones d'informations dont nous avons besoin
Y_train = train["Survived"] # Nous détermions le Output
X_test  = test.drop("PassengerId", axis=1).copy() #  Nous prenons les colones d'informations dons nous avons besoin pour le test

X_train, X_valeur, Y_train, Y_valeur = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)

print(X_train.shape, Y_train.shape, X_valeur.shape, Y_valeur.shape, X_test.shape)

# Construisons notre modèle de Régression Logistique

Modele_Regression = LogisticRegression(solver='lbfgs') #on sélectionne l'algorithme

Modele_Regression.fit(X_train, Y_train) # on entraîne l'algorithme avec la dataset crée X_train et son Output y_train

precision_Regression = round(Modele_Regression.score(X_valeur, Y_valeur) * 100, 2) # Puis on transmet le dataset testé à l'algorithme entraîné et on obtient le pourcentage de précision

print("Le taux de précision du modèle Régression Logistique est de : ",precision_Regression)

""" Nous allons procéder de la même manière pour tous les autres modèles que nous allons tester """


# Construisons notre modèle de Support Vector Machine (SVM) 

Modele_SVM = SVC(gamma='auto') 

Modele_SVM.fit(X_train, Y_train) 

precision_SVM = round(Modele_SVM.score(X_valeur, Y_valeur) * 100, 2) 

print("Le taux de précision du modèle Support Vector Machine (SVM) est de : ",precision_SVM)


# Construisons notre modèle de K-Nearest Neighbours

Modele_KNN = KNeighborsClassifier(n_neighbors = 3)

Modele_KNN.fit(X_train, Y_train)

precision_KNN = round(Modele_KNN.score(X_valeur, Y_valeur) * 100, 2)

print("Le taux de précision du modèle K-Nearest Neighbours est de : ",precision_KNN)


# Construisons notre modèle Arbre de décision

Modele_AD = DecisionTreeClassifier(max_depth=3)

Modele_AD.fit(X_train, Y_train)

precision_AD = round(Modele_AD.score(X_valeur, Y_valeur) * 100, 2)

print("Le taux de précision du modèle Arbre de décision est de : ",precision_AD)


""" Cherchons maintenant à comparer les résultats des différents modèlles afin de déterminer lequel est le plus efficace
En construisant un dataframe qui va lister chaque modèle et l'associer à son résultat de précision """

modeles = pd.DataFrame({
    'Modele': ['Regression Logistique','Support Vector Machine', 'KNearestNeighbors',
               'Arbre de décision'],
    'Précision': [precision_Regression,precision_SVM,precision_KNN,
               precision_AD]})

modeles.sort_values(by='Précision', ascending=False)

print(modeles)

""" Après avoir utilisé plusieurs modèles, on voit que le modèle de Regression Logistique et le modèle d'Arbre de décision ont les plus précisions les plus élevés et sont donc les plus efficaces"""

""" En dernière étape nous allons enfin tester le modèle de régression logistique sur le dataset et voir les resultats de correlation entre les informations et la survie"""

correlation_RL = pd.DataFrame(titanic.columns.delete(0)) # Nous supprimons la table PassengerId 

correlation_RL.columns = ['Information']

correlation_RL["Correlation"] = pd.Series(Modele_Regression.coef_[0]) 

correlation_RL.sort_values(by='Correlation', ascending=False)

print(correlation_RL)

correlation_RL.plot(x='Information',y='Correlation',kind='barh')

""" Ansi Le graphique nous permet de conclure que d'après le modèle de la Régression Logistique,
le taux de correlation le plus élevé en rapport avec la survie d'un passager est surtout en lien avec son sex,l'endroit d'où il a embarqué mais les autres facteurs jouent aussi un rôle"""


























