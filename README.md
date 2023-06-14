# another-mnistPredictor-in-tensorflow-but-you-can-use-it
La predicción de números escritos a mano con Tensorflow. Es ya una tarea clásica para estudiantes de Redes Neuronales.

Vamos a entrenar un modelo de red neuronal utilizando el conjunto de datos MNIST, que consiste en imágenes de dígitos escritos a mano. 
A continuación, te explico cada uno de los pasos de manera sencilla:

**Paso 1: Cargar el dataset MNIST**

Se carga el dataset MNIST que contiene las imágenes de entrenamiento y prueba, así como las etiquetas correspondientes que indican qué dígito representa cada imagen.
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

**Paso 2: Preprocesamiento de los datos**
Se realiza un preprocesamiento de los datos dividiendo los valores de píxeles entre 255 para normalizarlos y asegurarse de que estén en un rango de 0 a 1. 
Esto facilita el procesamiento para la red neuronal.

La escala de grises generalmente se representa en números de 0 a 255. 
En este rango, el valor 0 representa el negro absoluto (sin intensidad de luz) y el valor 255 representa el blanco absoluto (máxima intensidad de luz).
La razón por la cual se utiliza el rango de 0 a 255 es por la representación de 8 bits, donde cada píxel en una imagen en escala de grises se almacena como un valor de 8 bits (1 byte). Con 8 bits, se pueden representar 2^8 = 256 valores distintos, es decir, desde 0 hasta 255. Cada valor representa un nivel de intensidad de luz en la escala de grises.

Paso 3: Crear el modelo de la red neuronal
Se crea el modelo de la red neuronal utilizando la API secuencial de Keras. El modelo consta de una capa de aplanamiento que convierte las imágenes de 2D en un vector de 1D, seguida de tres capas ocultas con activación ReLU y una capa de salida con activación softmax que clasifica los dígitos en las 10 posibles categorías.

Paso 4: Compilar el modelo
Se compila el modelo especificando el optimizador, la función de pérdida y las métricas que se utilizarán durante el entrenamiento. En este caso, se utiliza el optimizador Adam, la función de pérdida sparse_categorical_crossentropy (apropiada para clasificación multiclase) y se mide la precisión.

Paso 5: Entrenar el modelo
Se entrena el modelo utilizando los datos de entrenamiento durante 50 épocas. Durante el entrenamiento, el modelo aprenderá a reconocer y clasificar los dígitos en base a las imágenes proporcionadas.

Paso 6: Evaluar el modelo con el conjunto de prueba
Una vez entrenado el modelo, se evalúa su rendimiento utilizando el conjunto de prueba. Se calcula la pérdida y la precisión del modelo en este conjunto. La precisión indica qué tan bien clasifica el modelo los dígitos en el conjunto de prueba.

Espero que esta explicación te haya ayudado a comprender cada uno de los pasos del código. Si tienes alguna otra pregunta, no dudes en preguntar.





User
por que las capas dense tienen 128 unidades
ChatGPT
Las capas densas con 128 unidades se eligen como una elección de diseño para el modelo de red neuronal en este caso particular. El número de unidades en una capa densa determina la cantidad de neuronas en esa capa y, por lo tanto, la capacidad de representación y aprendizaje del modelo.

En general, una mayor cantidad de unidades en las capas densas permite que el modelo aprenda representaciones más complejas y sofisticadas de los datos de entrada. Sin embargo, agregar más unidades también aumenta la cantidad de parámetros en el modelo, lo que puede llevar a un mayor consumo de recursos computacionales y un mayor riesgo de sobreajuste si no se controla adecuadamente.

En este caso, se ha elegido un valor de 128 unidades para las capas densas basado en consideraciones de rendimiento y capacidad de representación adecuada. Este valor es comúnmente utilizado y ha demostrado ser efectivo en una variedad de tareas de clasificación de imágenes, incluido el reconocimiento de dígitos en el conjunto de datos MNIST.

Es importante tener en cuenta que el número óptimo de unidades en las capas densas puede variar dependiendo del problema específico y de la arquitectura del modelo. Puedes experimentar con diferentes valores para determinar la configuración más adecuada para tu caso particular.
