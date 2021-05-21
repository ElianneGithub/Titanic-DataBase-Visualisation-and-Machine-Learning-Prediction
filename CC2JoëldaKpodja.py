#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 10:39:42 2020

@author: joelda
"""

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


from subprocess import check_output

print(check_output(["ls", "titanic.csv"]).decode("utf8"))


titanic = pd.read_csv("titanic.csv") # Chargeons la base de donnees


titanic.info()  


print(titanic.describe())

""" Exercice 1 :  Dans un premier temps, nous nettoierons les données:
    
• élimination des colonnes pour lesquelles beaucoup de données sont manquantes, ou peu pertinentes 

• élimination des lignes qui contiennent encore des données manquantes après élimination des colonnes """


# D'apres les informations recoltes grace a titanic.info(), la colone Cabin contient le moins de donnes et donc le plus de cases vides
# la fonction titanic.isnull().sum() nous montre que 177 cases sont vides pour la colonne de l age et 687 cases sont vides pour la colone Cabin et seulement 2 pour la colonne Embarked

print(titanic.isnull().sum())

# Donc on enleve la colonnes Cabin dont 77% des donnes sont manquantes en plus de la colonne Ticket n'est pas très pertinent 


titanic = titanic.drop(['Cabin','Ticket'],axis=1)

print(titanic.head())

# Nous allons ensuite proceder a l elimination des lignes contiennant encore des données manquantes

titanic = titanic.dropna()

print(titanic.isnull().sum())

# Nous sommes a present certains que toutes les lignes ne contiennent pas d'informations manquantes et que les colonnes non pertinantes ont ete elimines

""" Exercice 2 : Dans un second temps, nous proposerons différents types de visualisation qui nous semblent appropriés,
l’objectif étant de repérer visuellement des motifs de corrélation entre les différentes informations. """


# Nous allons dans un premier temps nous interesser à la demographie de la population des voyageurs donc faire des visualisations  pertinentes autour de leurs sexe, age et de leur parentalite


""" les histogrammes suivant nous permettent de constater que la majorité des passagers se situent entre l'age de 20 et 40 ans """


sns.distplot(titanic['Age'],color='darkgreen')
plt.show()

sns.countplot(x='Age', data=titanic)
plt.show()



""" les histogrammes suivant nous permettent de constater que la majorité des passagers ont voyagé en 3e classe et la majorité on embarqué à l'embarquement S 
et que la majorité des passagers ont payé un tarif se situant entre 0 et 50 et que très peu ont payé un tarif excédant 100.
Les tarifs des classes 1 ont un tarif plus cher, suivis par les classes 2 et la classe 3 qui contient le plus de passagers est la moins chère.

La plupart des passagers voyageant en premiere et seconde classe sont plus ages """

 

sns.countplot(x='Pclass',data=titanic)
plt.show()

sns.countplot(x='Embarked',data=titanic)
plt.show()

sns.distplot(titanic['Fare'],color='darkblue')
plt.show()

sns.barplot(x='Pclass',y='Fare',data=titanic)
plt.show()

sns.boxplot(data=titanic, x='Age', y='Sex', hue ='Pclass')
plt.show()


""" les histogrammes suivant nous permet de constater que la majorité des passagers sont des hommes qui sont environ deux fois plus nombreux que les femmes.
L'ensemble des passagers ont entre 20 et 40 ans
Nous pouvons clairement voir que la moyenne d'âge des passagers homme est juste en dessous de 35 ans tandis que celle des passagers femme est autour de 28.  

"""

sns.countplot(x='Sex', data=titanic)
plt.show()


sns.violinplot(x='Sex', y='Age', data=titanic)
plt.show()

sns.barplot(x='Sex', y='Age', data=titanic)
plt.show()

sns.boxplot(x='Sex', y='Age', data=titanic)
plt.show()


# Nous allons à present nous concentrer autour des données concernant la survie des passagers en fonction des differents autres facteurs

""" on constate que la majorité des passagers n'ont pas survécu """

sns.countplot(x='Survived',data=titanic)
plt.show()


""" En croissant les donnes sur la survie et la classe, on remarque que la majorité des passagers voyageant dans la classe 3 n'ont pas survécu
tandis que la majorité des voyageurs ayant survécu voyageaient en classe 1   """

sns.countplot(x='Survived',hue='Pclass',data=titanic)
plt.show()



""" En corrélant les donnes sur la survie et le sexe, on remarque que la majorité des passagers n'ayant pas survécu sont des hommes
 tandis que la majorité des passagers ayant survécu sont des femmes.
Il semble que la majorité et des hommes et des femmes qui ont survécu avaient entre 20 et 40 ans
 """

sns.countplot(x='Survived',hue='Sex',data=titanic)
plt.show()

sns.boxplot(data=titanic, x='Age', y='Sex', hue ='Survived')

""" La majorité des passagers ayant survécu ont embarqué en S et en C. Parci ceux qui n'ont pas survécu la majorité ont embarqué en S  """

sns.countplot(x='Survived',hue='Embarked',data=titanic)
plt.show()

""" On remarque que la majorité des personnes ayant payé un tarif entre 0 et 100 n'ont pas survécu
 cependant on voit bien la quasi totalité des personnes ayant payé untarif supérieur à 100 ont survécu """
 
sns.FacetGrid(titanic, col='Survived').map(plt.hist, 'Fare', bins=20)
plt.show()

sns.barplot(x='Survived',y="Fare",data=titanic)
plt.show()

""" il n'ya pas vraiment de corrélation pertinente entre l'âge et la survie mais on peut déceler la majorité des 20-40 ans n'ont pas survécu """

sns.violinplot(x='Survived', y='Age', data=titanic)
plt.show()

sns.FacetGrid(titanic, col='Survived').map(plt.hist, 'Age', bins=20)
plt.show()

""" Nous allons ensuite regarder les donnés de manière plus précise autour de la survie des passagers """

""" Dans la classe 1 : la majorité des personnes ayant survécu ont entre 0 et 60 ans
Dans la classe 3 une majorité de personnes n'ont pas survécu et ils avaient entre 20 et 40 ans
 """

sns.FacetGrid(titanic, col='Survived', row='Pclass', height=2.2, aspect=1.6).map(plt.hist, 'Age', alpha=.5, bins=20).add_legend()
plt.show()
""" En général les personnes ayant entre 1 et 3 parents ou de famille ont le plus survécu """

titanic['Parents'] = titanic['SibSp'] + titanic['Parch']
sns.factorplot('Parents','Survived', data=titanic, aspect = 1 )
plt.show()

""" Il semble que les pesonnes ayant embarqué à C et ayant payé un tarif plus élevé ont le plus survécu    """

sns.FacetGrid(titanic, row='Embarked', size=2.2, aspect=1.6).map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep').add_legend()
plt.show()

sns.FacetGrid(titanic, row='Embarked', col='Survived', size=2.2, aspect=1.6).map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None).add_legend()
plt.show()

""" Enfin regroupons toutes les colonnes de sorte à faire resortir de manière globale les relations entre les differentes informations autour de la survie """

sns.pairplot(titanic.drop("PassengerId", axis=1), hue="Survived", size=3)
plt.show()

""" A present nous allons construire quelques tableau afin de nous donner une idée plus concrète sur le taux de survie en fonction des differentes informations"""

print(titanic[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))

""" les passagers en classe 1 ont le taux de survie le plus élévé"""

print(titanic[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

""" les passagers femme ont le taux de survet le plus élevé"""

print(titanic[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(titanic[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))

""" les passagers ayant le moins de parents ont les taux de survie les plus élevé"""

print(titanic[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))

""" Les passagers ayant embarqué à l'embarquement C ont le taux de survie le plus élevé"""

titanic['Fareplage'] = pd.qcut(titanic['Fare'], 4)
print(titanic[['Fareplage', 'Survived']].groupby(['Fareplage'], as_index=False).mean().sort_values(by='Fareplage', ascending=True))

""" les passagers ayant payé des tarifs se trouvant dans les plage de prix les plus élevés ont le taux de survie le plus élevé """





































