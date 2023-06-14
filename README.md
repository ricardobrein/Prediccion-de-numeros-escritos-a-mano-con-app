# Another-mnistPredictor-in-tensorflow-but-you-can-use-it

## Esta es una implementacion de uan red neuronal para predecir digitos escritos a mano pero cree una aplicacion de tkinter para poder escribir numeros y que el modelo haga la prediccion.
La predicción de números escritos a mano con Tensorflow. Es ya una tarea clásica para estudiantes de Redes Neuronales.

Vamos a entrenar un modelo de red neuronal utilizando el conjunto de datos MNIST, que consiste en imágenes de dígitos escritos a mano. 
A continuación, te explico cada uno de los pasos de manera sencilla:

### Paso 1: Cargar el dataset MNIST

Se carga el dataset MNIST que contiene las imágenes de entrenamiento y prueba, así como las etiquetas correspondientes que indican qué dígito representa cada imagen.
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

### Paso 2: Preprocesamiento de los datos
Ya que las imagenes MNIST estan en escala de grises y miden 28x28 pixeles, Se realiza un preprocesamiento de los datos dividiendo los valores de píxeles entre 255 para normalizarlos y asegurarse de que estén en un rango de 0 a 1. La escala de grises generalmente se representa en números de 0 a 255.

> Esto facilita el procesamiento para la red neuronal.

 
> En este rango, el valor 0 representa el negro absoluto (sin intensidad de luz) y el valor 255 representa el blanco absoluto (máxima intensidad de luz).
>> La razón por la cual se utiliza el rango de 0 a 255 es por la representación de 8 bits, donde cada píxel en una imagen en escala de grises se almacena como un valor de 8 bits (1 byte). Con 8 bits, se pueden representar 2^8 = 256 valores distintos, es decir, desde 0 hasta 255. Cada valor representa un nivel de intensidad de luz en la escala de grises.

### Paso 3: Crear el modelo de la red neuronal
Creamos el modelo de red neuronal utilizando la API de Keras. El modelo consta de una capa Flatten(): Esta agrega una capa de aplanamiento que convierte las características 2D en un vector 1D. Seguimos con tres capas ocultas Dense(units=128, activation='relu'): completamente conectadas con 128 unidades (neuronas) y función de activación ReLU y por ultimo una capa de salida completamente conectada (Dense) pero con 10 unidades (una para cada dígito) y función de activación softmax. La función softmax asigna probabilidades a cada clase y se utiliza en problemas de clasificación multiclase.

### Paso 4: Compilar el modelo
Se compila el modelo especificando el optimizador, la función de pérdida y las métricas que se utilizarán durante el entrenamiento. En este caso, se utiliza el optimizador [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), la función de pérdida sparse_categorical_crossentropy (apropiada para clasificación multiclase) y se mide la precisión.
![Funcion de perdida](media/loss.jpg)

### Paso 5: Entrenar el modelo
Se entrena el modelo utilizando los datos de entrenamiento durante 50 epochs. Durante el entrenamiento, el modelo aprenderá a reconocer y clasificar los dígitos de el dataset MNIST

### Paso 6: Evaluar el modelo con el conjunto de prueba.

![Grafica de perdida y precision](media/grafico)

Una vez entrenado el modelo, se evalúa su rendimiento utilizando el conjunto de prueba. Se calcula la pérdida y la precisión del modelo en este conjunto. La precisión indica qué tan bien clasifica el modelo los dígitos en el conjunto de prueba.

Espero que esta explicación ayude a comprender mejor, ya que me ha servido bastante para entender los principios de las redes neuronales.

### Algunas dudas que me surgieron e investigué un poco.

**Por que en los libros veia 64, 128 y hasta 1024 unidades en la capa oculta Dense (totalmente conectada)?**

R: En general, una mayor cantidad de unidades en las capas densas permite que el modelo **aprenda representaciones más complejas** y sofisticadas de los datos de entrada. Sin embargo, agregar más unidades también aumenta la cantidad de parámetros en el modelo, lo que puede lleva a un **mayor consumo de recursos** y un mayor **riesgo de sobreajuste.**

En este caso, un valor de 128 debido a que probe varios y con este numero el rendimiento fue adecuado. Este valor es comúnmente utilizado en una variedad de tareas de clasificación de imágenes, incluso el reconocimiento de dígitos MNIST en muchas implementaciones lo veía.

El número óptimo de unidades en las capas densas puede variar dependiendo del problema específico y de la arquitectura del modelo.
