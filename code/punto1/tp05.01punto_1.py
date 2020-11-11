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

from sklearn.model_selection import train_test_split
from sklearn import tree

# Separo en 80-20 entrenamiento y testeo
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

# Creo el objeto Decision Tree classifer
arbol_tt = tree.DecisionTreeClassifier()

# Entreno el Decision Tree Classifer
arbol_tt = arbol_tt.fit(X_train,y_train)

#Realizo las predicciones en función del árbol generado
y_pred = arbol_tt.predict(X_test)

from sklearn import metrics #Importar el módulo metrics de scikit-learn

# Vamos a testear el modelo
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))