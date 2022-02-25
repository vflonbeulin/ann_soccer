#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Chargement des donnees
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 1:10].values
y = dataset.iloc[:, 10].values

# Traitement des donnees manquantes
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
imputer = imputer.fit(X[:, [5]])  
X[:, [5]] = imputer.transform(X[:, [5]])

# Encoder variables categoriques
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Demarre le reseau de neuronnes
from tensorflow.keras.models import Sequential # initialise le reseau
from tensorflow.keras.layers import Dense # creer les couches du reseau

# Initialisation
classifier = Sequential()

# Ajouter couche d'entrée et la couche cachée
classifier.add(Dense(
    units=6,
    activation='relu',
    kernel_initializer='uniform',
    input_dim=9
))

# Ajouter une 2eme couche cachée
classifier.add(Dense(
    units=6,
    activation='relu',
    kernel_initializer='uniform'
))

#Ajouter la couche de sortie
classifier.add(Dense(
    units=1,
    activation='sigmoid',
    kernel_initializer='uniform'
))

# Compilation
classifier.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

classifier.fit(
    X_train,
    y_train,
    batch_size=10,
    epochs=100
)

# Predictions
y_pred = classifier.predict(X_test)
y_pred = (y_pred > .5)

# Prediction pour un joueur lambda en ligue nationale
# Age : 24
# Taille : 175 (cm)
# Poids : 160 (livres lb)
# Pied favoris : right (right/left)
# Controle ballon : 85 (score de 1 a 100)
# vision : 82
# Endurance : 80
# Passe courte : 68
# Placement : 81

playerArray = np.array([[24,175,160,0,85,82,80,68,81]])
lambdaPlayer_prediction = classifier.predict(sc.transform(playerArray))
lambdaPlayer_prediction = (lambdaPlayer_prediction > .5)

# Prediction pour un joueur se rapprochant de Leo Messi statistiquement
# Age : 27
# Taille : 170 (cm)
# Poids : 160 (livres lb)
# Pied favoris : right (right/left)
# Controle ballon : 97 (score de 1 a 100)
# vision : 90
# Endurance : 74
# Passe courte : 86
# Placement : 92

playerArray = np.array([[27,170,160,0,97,90,74,86,92]])
internationalPlayer_prediction = classifier.predict(sc.transform(playerArray))
internationalPlayer_prediction = (internationalPlayer_prediction > .5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
