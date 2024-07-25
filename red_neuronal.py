import tensorflow as tf
import numpy as np

#le pasamos grados celsius para que los compare con los fahrenheit
celsius = np.array([-40, -10, 0, 8, 15, 22, 38, 40, 50, 60, 65, 75, 100], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100, 104, 122, 140, 149, 167, 212], dtype=float)

#creamos las capas
capa = tf.keras.layers.Dense(units=3, input_shape=[1])
capa2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([capa, capa2, salida])

#va ajustar cada coneccion con un 0.001 de margen de error
modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.001),
    loss='mean_squared_error'
)

#lo entrenamos
print("Comenzando entrenamiento...")
#le decimos que compare celsius con fahrenheit y esto lo haga un total de 10000 vueltas
historial = modelo.fit(celsius, fahrenheit, epochs=10000, verbose=False)
print("Modelo entrenado")

#tabla que muestra un grafico de como aprende
import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitudde perdida")
plt.plot(historial.history["loss"])

print("Hagamos una prediccion")
resultado = modelo.predict([957.0])
print("El resultado es " + str(resultado) + " fahrenheit!")













