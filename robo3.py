from vpython import *
import numpy as np

# Configuración de la escena
scene = canvas(title='Trayectoria del Robot Uniciclo', width=800, height=600)

# Parámetros de simulación
dt = 0.1  # Intervalo de tiempo
t_total = 30  # Tiempo total de simulación
t = np.arange(0, t_total+dt, dt)  # Vector de tiempo
x = np.zeros_like(t)  # Inicialización del vector de posiciones en x
y = np.zeros_like(t)  # Inicialización del vector de posiciones en y
theta = np.zeros_like(t)  # Inicialización del vector de ángulos

# Parámetros del robot
L = 0.5  # Distancia entre las ruedas
v_l = 1.0  # Velocidad de la rueda izquierda
v_r = 10  # Velocidad de la rueda derecha

# Velocidades de traslación y rotación
v = (v_r + v_l) / 2  # Velocidad lineal
w = (v_r - v_l) / L  # Velocidad angular

# Creación del robot y su rastro
robot = cylinder(pos=vector(0, 0, 0), axis=vector(1, 0, 0), radius=0.1, color=color.red)
trail = curve(color=color.blue)

# Simulación
for i in range(1, len(t)):
    rate(30)  # Controla la velocidad de la simulación
    theta[i] = theta[i-1] + w*dt  # Actualización del ángulo
    x[i] = x[i-1] + v * np.cos(theta[i]) * dt  # Actualización de la posición en x
    y[i] = y[i-1] + v * np.sin(theta[i]) * dt  # Actualización de la posición en y

    # Actualización de la posición del robot y su rastro
    robot.pos = vector(x[i], y[i], 0)
    robot.axis = vector(np.cos(theta[i]), np.sin(theta[i]), 0)
    trail.append(pos=vector(x[i], y[i], 0))

# Agregar etiquetas y título a la gráfica
label(pos=vector(5, 5, 0), text='Trayectoria del Robot Uniciclo', xoffset=20, yoffset=20, space=30, height=16, border=4, font='sans')
