# Tp04HerramientasParaElAnalisisDeInformacion

### 1. A partir del dataset presentado a continuación, y teniendo en cuenta las fórmulas de entropía y ganancia de información calcule y diagrame el árbol de decisión que le permita decidir si comer asado o no en función del clima:

##### Respuesta: el arbol de decision generado es el siguiente

![arbol](code/punto1/tree_punto1.png)


### 2. Trabaje con el dataset de Scikit Learn “wine”:
a. Utilice el metadata que provee la librería, ¿Cuál es el tema que aborda el dataset?

Respuesta: El dataset usa el analisis quimico para determinar el origen de los vinos
haciendo un import del dataset se puede observar su descripcion completa
![descripcion_dataset_wine](code/punto2/descripcion_wine.txt)

b. Genere el árbol de decisión que permita clasificar los diferentes tipos de vino utilizando un muestreo con proporciones de 80% para entrenamiento y 20% para testeo.

![arbol punto 2b](code/punto2/arbol_punto2bTP05.png)

c. Explore la solución dada y las posibles configuraciones para obtener un nuevo árbol que clasifique “mejor”. Documente las conclusiones.

Accuracy : 
70 train/ 30 test => 93% precision
80 train/ 20 test => 94% precision
90 train/ 10 test => 100% orecision

Al parecer una configuracion de 90 train/ 10 test => 100% precision clasifica de forma optima el arbol. Sin embargo esto es obvio al disminuir el conjunto de prueba, es decir, si la cantidad de pruebas disminuye es mucho mas probable que los pocos testeos que hagan sean acertados, en cambio cuantas mas pruebas se hagan, es mas probable que se provoquen fallas. Luego a medida que aumentamos el conjunto de prueba se logra un punto maximo de precision del 94% precision con un conjunto de 80% train y 20% de test.

### 3. Ahora, analice el archivo zoo.csv2:
a. Genere el árbol de decisión que permita inferir el tipo de animal en función de sus características. Explique someramente que resultado se obtiene en términos del árbol y en términos de la eficiencia del mismo.

![arbol punto3](code/punto3/arbol_punto3TP05.png)

• ¿Varía ese resultado si se elimina el atributo “animal”? ¿Por qué?

- se puede observar que el resultado no varia, puede deberse a que la entropia no varia. Se reemplaza la variable "animal" por la variable "legs"

• Cuantos niveles posee el árbol generado? ¿Qué atributos debemos modificar si deseamos realizar una poda del mismo? Modifique esos atributos para que el árbol generado conste de 4 niveles. ¿Afecta la eficiencia de la clasificación esta modificación?

- el arbol generado posee 6 niveles (contando la raiz)
- para lograr podar el arbol, se tiene que modificar el atributo "max_depth=4"

´´´
arbol_parametrizado = tree.DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=4)
´´´

Baja la eficiencia de un 95% a un 90%


### 4. Se provee la base de datos de los pasajeros del famoso barco que se hundiera en su viaje inaugural (archivo titanic-en.csv) con los siguientes atributos y valores posibles:
- Class {"1st","2nd","3rd","crew"}
- Age {"adult","child"}
- Sex {"male","female"}
- Survived {"yes","no"}
Genere el árbol de clasificación, explore la solución dada y las posibles alternativas para obtener un nuevo árbol que clasifique “mejor”.

![arbol punto4](code/punto4/arbol_punto4TP05.png)

Conclusion: Con un 56% de datos de entrenamiento, y 44% de prueba, se obtiene una precision del 79%, la cual es la mejor alternativa que clasifica mejor el arbol de decision.

### 5. Un Banco de Portugal realizó una campaña de marketing en busca de clientes de plazos fijos basada en llamados telefónicos. Se provee el dataset3 real (bank-full.csv) con más 45000 instancias y el detalle (bank-names.txt) de los datos registrados de cada una de las personas contactadas por la entidad bancaria.
a. Realice las tareas necesarias para poder procesar el dataset en Scikit-Learn.

Fue necesario cambiar los tipos de datos a string
´´´
data['balance'] = data['balance'].astype('string')
data['pdays'] = data['pdays'].astype('string')
´´´
Con visual-studio reemplace los ";" por ","

y la fila de "target names" le tuve que sacar la comillas dobles

b. Luego, genere el árbol de decisión, y optimice los resultados, con el objetivo de explicar cuáles son las características más importantes que permiten identificar a una persona que accederá o no al plazo fijo. Documente los resultados.

el arbol resultante es el siguiente

![arbol punto5-b](code/punto5/arbol_punto5TP05.png)

Se puede observar que el atributo "duration" es el mas importante. Asimismo para mejorar la precision del modelo de marchine learning se toma un 50% de entrenamiento y un 50% de prueba, lo que nos da una precision del 89%.

### 6. Guarde los archivos resultantes de las actividades prácticas en una carpeta denominada tp0301-<legajo> que a su vez tenga un directorio por cada uno de los puntos de este trabajo, comprima la carpeta y envíelo al equipo docente.

### Medidas de evaluación para técnicas de clasificación:
En función de la clasificación realizada, complete las siguientes actividades:
    a. Accuracy.
        1. Escoja un modelo y calcule el accuracy del mismo.

Tomando el modelo del punto 2 - wine.data, el arbol que lo clasifica tiene un Accuracy del 94% para un conjunto de entrenamiento del 80% y 20% de prueba. 

        2. ¿Cómo se interpreta la métrica anterior?

Esto significa que el 94% del conjunto de prueba fue bien clasificado.

        3. ¿Qué aporta el accuracy?

Nos permite saber, que tan bien el modelo predictivo describe a los datos.

    b. Recall/Precision.
        1. Ahora, sobre el mismo modelo de a), calcule las métricas recall y precisión para ambos modelos.

![calculo Recall y Precision punto 2 - Wine](code/medidas-evaluacion/recall-precision.png)

        2. ¿Cuál es la diferencia entre ambas?

El recall es la cantidad que el modelo de machine learning sera capaz de clasificar y la precision que tan bien seran clasificara el modelo cada tipo de vino.

        3. ¿Qué aspectos aborda cada una?

En el ejemplo de los vinos, el Recall nuestro de modelo de machine learning es capaz de identificar 93% de los vinos de clase 1, el 100% de clase 2 y el 100% de los vinos de clase 3. En cuanto a la precision, nos dice que el 100% de los vinos de clase 1 que seran bien clasificados, el 94% para los vinos de clase 2 y el 100% de clase 3.

    c. Matiz de confusión: ¿En qué casos el modelo clasifica mal?

    Esta es la matriz del modelo para el data set wine.

        [[14  0  0]
        [ 1 15  0]
        [ 0  0  6]]

    Esto nos dice que de los 15 clases de vino tipo 1, solo 1 la clasifico como tipo 2.
