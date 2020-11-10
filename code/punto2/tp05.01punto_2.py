import pandas as pd 

# Lee el archivo
data = pd.read_csv("code\punto2\wine.data")

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
for column_name in data.columns:
  if (data[column_name].dtype == object) & (column_name!='classIdentifier'):
    data[column_name] = le.fit_transform(data[column_name])

print(data.head())

# Separo "a mano" festures de target

# Features
feature_names = list(data.columns)
# Elimino type porque es la clase
feature_names.remove('classIdentifier')

x = data[feature_names]

# Target
y = data.classIdentifier
le.fit(data['classIdentifier'])
target_names=le.classes_

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

# Vemos un reporte de clasificación de varias métricas
print(metrics.classification_report(y_test, y_pred))

import numpy as np
species = np.array(y_test)
predictions = np.array(y_pred)

metrics.confusion_matrix(species, predictions)

arbol_parametrizado = tree.DecisionTreeClassifier(criterion="entropy", max_depth=7, min_samples_leaf=4)

# Entreno el Decision Tree Classifer con el mismo muestreo generado antes (80-20 %)
arbol_parametrizado = arbol_parametrizado.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = arbol_parametrizado.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.tree import export_graphviz
from io import StringIO  
import pydotplus

# Capturlo la exportación del gráfico del árbol  
dot_data = StringIO()
tree.export_graphviz(arbol_parametrizado, out_file=dot_data,
                                feature_names=feature_names,
                                class_names=['1','2','3'],
                                filled=True, rounded=True,
                                special_characters=True)  

# Con el string del dot lo paso a un gráfico
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

# Genero png y lo descargo
graph.write_png(r'code\punto2\arbol_punto2bTP05.png')