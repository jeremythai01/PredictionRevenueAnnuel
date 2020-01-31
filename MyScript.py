import numpy as np
from sklearn import preprocessing


# Instanciation des variables
X, y = [], []
count_class1, count_class2 = 0, 0
max_datapoints = 25000

# Lecture des et importation de 25 000 lignes
with open('income_data.txt', 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line[:-1].split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

# Conversion en matrice NumPy
X = np.array(X)

# Codification des variables catégorielles
label_encoder = [] 
X_encoded = np.empty(X.shape)
for i,item in enumerate(X[0]):
    if item.isdigit(): 
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

# Création des ensembles d'entraînement et de test

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X, y = X_encoded[:, :-1].astype(int), X_encoded[:, -1].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

#Modele regression log 

# Instanciation du modèle
logreg = LogisticRegression(solver = "lbfgs", multi_class = "multinomial")
# Entraînement du modèle
logreg = logreg.fit(X_train, y_train)
# Test du modèle
print("Les probabilités sont : ",logreg.predict_proba(X_test)[0]*100, 
      ", c'est donc une" ,logreg.predict(X_test)[0])
print("Le modèle a une performance de :", logreg.score(X_test, y_test))


#Modele k plus proches voisins 
from sklearn.neighbors import KNeighborsClassifier
for i in range(1, 6):
    # Instanciation du modèle
    knn = KNeighborsClassifier(n_neighbors = i)
    # Entraînement du modèle
    knn = knn.fit(X_train, y_train)
    # Test du modèle
    print("Pour", i, "voisins, les probabilités sont : ",knn.predict_proba(X_test)[0]*100, 
          ", c'est donc une" ,knn.predict(X_test)[0])
    print("Le modèle a une performance de :", knn.score(X_test, y_test))

    #Modele reseaux neurones 
from sklearn.neural_network import MLPClassifier

for i in range(100, 700, 100):
    # Instanciation du modèle
    mlp = MLPClassifier(max_iter = i, random_state = 1234)
    # Entraînement du modèle
    mlp = mlp.fit(X_train, y_train)
    # Test du modèle
    print("Avec", i, "itérations, les probabilités sont : ", mlp.predict_proba(X_test)[0]*100, 
          ", c'est donc une" , mlp.predict(X_test)[0])
    print("Le modèle a une performance de :", mlp.score(X_test, y_test))

#Modele arbre de decision

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Instanciation du modèle
dt = DecisionTreeClassifier()
# Entraînement du modèle
dt = dt.fit(X_train, y_train)
# Test du modèle
print("Les probabilités sont : ", dt.predict_proba(X_test)[0]*100, 
      ", c'est donc une" , dt.predict(X_test)[0])
print("Le modèle a une performance de :", knn.score(X_test, y_test))

#Modele machine a vecteurs de support 

from sklearn.svm import SVC

# Instanciation du modèle
svc = SVC(gamma = "scale", probability = True)
# Entraînement du modèle
svc = svc.fit(X_train, y_train)
# Test du modèle
print("Les probabilités sont : ", svc.predict_proba(X_test)[0]*100, 
      ", c'est donc une" , svc.predict(X_test)[0])
print("Le modèle a une performance de :", svc.score(X_test, y_test))

    