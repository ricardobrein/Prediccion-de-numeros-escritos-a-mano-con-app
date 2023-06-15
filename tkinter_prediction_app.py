import tkinter as tk
from PIL import Image, ImageOps
import numpy as np
import io

# Función para predecir el número dibujado
def predecir_numero():
    imagen_dibujada = lienzo.postscript(colormode='gray')
    imagen = Image.open(io.BytesIO(imagen_dibujada.encode('utf-8')))
    imagen = imagen.resize((28, 28)).convert('L')  # Redimensionar y convertir a escala de grises
    imagen = ImageOps.invert(imagen)  # Invertir los colores de la imagen
    imagen = np.array(imagen) / 255.0  # Convertir el array con NumPy y normalizar los valores de píxeles entre 0 y 1
    imagen = imagen.reshape(1, 28, 28)  # Añadir una dimensión más para que coincida con el formato de entrada del modelo
    prediction = model.predict([imagen])
    numero_predicho = prediction.argmax()
    label_prediccion.config(text="El número es un: " + str(numero_predicho))


# Función para borrar el lienzo de tk

def borrar_lienzo():
    lienzo.delete("all")
    label_prediccion.config(text="El número es un: ")


# Configuración de la ventana
ventana = tk.Tk()
ventana.title("Detector de números escritos a mano")

# Crear lienzo para dibujar
lienzo = tk.Canvas(ventana, width=200, height=200, bg="white")
lienzo.grid(row=0, column=0, columnspan=2)

# Crear botón para predecir el número

boton_predecir = tk.Button(ventana, text="Predecir", command=predecir_numero)
boton_predecir.grid(row=1, column=0)

# Crear botón para borrar el lienzo

boton_borrar = tk.Button(ventana, text="Borrar", command=borrar_lienzo)
boton_borrar.grid(row=1, column=1)

# Crear etiqueta para mostrar la predicción

label_prediccion = tk.Label(ventana, text="El número es un: ")
label_prediccion.grid(row=2, column=0, columnspan=2)

# Función para dibujar en el lienzo

def dibujar(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    lienzo.create_oval(x1, y1, x2, y2, fill="black")

lienzo.bind("<B1-Motion>", dibujar)

# Cargar el modelo previamente entrenado

from tensorflow.keras.models import load_model
model = load_model('modelo_mnist_keras')

# Iniciar la ventana
ventana.mainloop()