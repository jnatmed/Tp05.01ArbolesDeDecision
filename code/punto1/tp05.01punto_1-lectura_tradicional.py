import pandas as pd 

# Lee el archivo
data = pd.read_csv(r'code\punto1\tabla_punto1.csv') 
# Preview the first 5 lines of the loaded data 
print(data.head())
print(data.shape)


# Separo "a mano" features de target

# Features
feature_names = ['PRONOSTICO','TEMPERATURA','HUMEDAD','VIENTO']
x = data[feature_names]

# Target
y = data.ASADO
print(y)

from sklearn import tree

arbol = tree.DecisionTreeClassifier(criterion='entropy')

arbol = arbol.fit(x, y)

#Importamos la librería
import graphviz
from graphviz import render


tree.export_graphviz(arbol, out_file='tree.dot',
                                feature_names=x,
                                class_names=y,
                                label='all',
                                filled=True, rounded=True,
                                special_characters=True)  

render('dot', 'png', 'tree.dot')