import pandas as pd 

# Lee el archivo
data = pd.read_csv('data/tabla_punto1.csv') 
# Preview the first 5 lines of the loaded data 
print(data.head())
print(data.shape)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
for column_name in data.columns:
  if (data[column_name].dtype == object) & (column_name!='ASADO'):
    data[column_name] = le.fit_transform(data[column_name])

print(data.head())

# Separo "a mano" features de target

# Features
feature_names = list(data.columns)
# Elimino type porque es la clase
feature_names.remove('ASADO')
x = data[feature_names]

# Target
y = data.ASADO
le.fit(data['ASADO'])
target_names=le.classes_
print(target_names)

from sklearn import tree

arbol = tree.DecisionTreeClassifier(criterion='entropy')

arbol = arbol.fit(x, y)

#Importamos la librería
import graphviz
from graphviz import render


tree.export_graphviz(arbol, out_file='tree.dot',
                                feature_names=feature_names,
                                class_names=target_names,
                                label='all',
                                filled=True, rounded=True,
                                special_characters=True)  

render('dot', 'png', 'tree.dot')