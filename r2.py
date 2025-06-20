import numpy as np
import pygame

# Definir el ambiente como una matriz cuadrada
# 0 representa un espacio vacío, 1 representa un obstáculo, 'M' representa la meta, etc.

mapa = [
    ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
    ['X', 'X', 'X', 'X', 'X', 0, 'X', 'X', 'X', 'X', 'X'],
    ['X', 'X', 'X', 'X', 'X', 0, 'X', 'X', 'X', 'X', 'X'],
    ['X', 'X', 'X', 'X', 'X', 0, 1, 0, 0, 0, 'X'],
    ['X',         0, 0, 0, 0, 0,'X',0, 'X', 'X', 'X'],
    ['X',         0,'X',1,'X',0,'X',0, 'X', 'X', 'X'],
    ['X',         0,'M',0,'X',0, 1, 0, 'X', 'X', 'X'],
    ['X',         0,'X',0,'X',0,'X',0, 'X', 'X', 'X'],
    ['X',         0, 1, 0, 0, 0,'X',0, 'X', 'X', 'X'],
    ['X', 'X', 'X', 'X', 'X', 0,'X',0, 'X', 'X', 'X'],
    ['X', 'X', 'X', 'X', 'X', 0, 0, 0, 'X', 'X', 'X'],
    ['X', 'X', 'X', 'X', 'X', 0,'X',0, 'X', 'X', 'X'],
    ['X', 'X', 'X', 'X', 'X', 0,'X',0, 'X', 'X', 'X'],
    ['X', 'X', 'X', 'X', 'X', 0, 1, 0, 'X', 'X', 'X'],
    ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X']
]

# Convertir la matriz a un array numpy para facilitar su manipulación
mapa_np = np.array(mapa)

# Obtener las dimensiones del ambiente
rows, cols = mapa_np.shape

# Crear un diccionario para mapear las coordenadas (x, y) a un índice único de estado
state_index = {}
index_state = {}
# Índice para representar cada estado
idx = 0

# Mapear las coordenadas a los índices de estado
for i in range(rows):
    for j in range(cols):
        if mapa_np[i, j] != 'X':  # Solo mapear si no es inaccesible
            state_index[(i, j)] = idx
            index_state[idx] = (i, j)
            idx += 1

# Número de estados y acciones
num_states = len(state_index)
num_actions = 4

def q_learning(alpha=0.1, gamma=0.9, epsilon=0.1, num_episodes=5000):
    # Inicialización de la matriz Q con valores arbitrarios
    Q = np.zeros((num_states, num_actions))

    # Función para verificar si una posición está dentro del mapa y no es un obstáculo
    def is_valid_move(x, y):
        return 0 <= x < rows and 0 <= y < cols and mapa_np[x, y] != 'X'

    for episodio in range(num_episodes):
        state = state_index[(1, 5)]  # Empezamos desde el estado inicial
        if episodio % 50 ==0:
            move_robot([1,5])
        done = False

        while not done:
            # Seleccionar la acción basada en la política epsilon-greedy
            if np.random.rand() < epsilon:
                action = np.random.choice(num_actions)
            else:
                action = np.argmax(Q[state])

            # Mapear la acción a un movimiento (arriba, abajo, izquierda, derecha)
            x, y = index_state[state]
            if action == 0 and is_valid_move(x - 1, y):  # Arriba
                next_state = state_index[(x - 1, y)]
            elif action == 1 and is_valid_move(x + 1, y):  # Abajo
                next_state = state_index[(x + 1, y)]
            elif action == 2 and is_valid_move(x, y - 1):  # Izquierda
                next_state = state_index[(x, y - 1)]
            elif action == 3 and is_valid_move(x, y + 1):  # Derecha
                next_state = state_index[(x, y + 1)]
            else:  # Si la acción nos sacaría fuera del mapa, nos quedamos en el mismo lugar
                next_state = state

            # Obtener la recompensa del nuevo estado
            if mapa_np[index_state[next_state]] == 'M':  # Si llegamos a la meta
                reward = 10
                done = True
                print("llego")
            elif mapa_np[index_state[next_state]] == 0:  # Si es una casilla vacía
                reward = -0.1
            else:  # Si es un obstáculo
                reward = -1

            # Actualizar la matriz Q
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            if episodio % 50 ==0:
                move_robot(index_state[next_state])
            # Actualizar el estado actual
            state = next_state

    return 

# Llamar a la función q_learning y obtener la política óptima
#optimal_path, optimal_directions = q_learning(mapa)

#print("Camino óptimo:", optimal_path)
#print("Direcciones óptimas:", optimal_directions)

def sarsa(alpha=0.1, gamma=0.9, epsilon=0.1, num_episodes=5000):
    # Inicialización de la matriz Q con valores arbitrarios
    Q = np.zeros((num_states, num_actions))

    # Función para verificar si una posición está dentro del mapa y no es un obstáculo
    def is_valid_move(x, y):
        return 0 <= x < rows and 0 <= y < cols and mapa_np[x, y] != 'X'

    for episodio in range(num_episodes):
        state = state_index[(1, 5)]  # Empezamos desde el estado inicial
        done = False

        # Seleccionar la primera acción usando la política epsilon-greedy
        if np.random.rand() < epsilon:
            action = np.random.choice(num_actions)
        else:
            action = np.argmax(Q[state])

        while not done:
            # Mapear la acción a un movimiento (arriba, abajo, izquierda, derecha)
            x, y = index_state[state]
            if action == 0 and is_valid_move(x - 1, y):  # Arriba
                next_state = state_index[(x - 1, y)]
            elif action == 1 and is_valid_move(x + 1, y):  # Abajo
                next_state = state_index[(x + 1, y)]
            elif action == 2 and is_valid_move(x, y - 1):  # Izquierda
                next_state = state_index[(x, y - 1)]
            elif action == 3 and is_valid_move(x, y + 1):  # Derecha
                next_state = state_index[(x, y + 1)]
            else:  # Si la acción nos sacaría fuera del mapa, nos quedamos en el mismo lugar
                next_state = state

            # Obtener la recompensa del nuevo estado
            if mapa_np[index_state[next_state]] == 'M':  # Si llegamos a la meta
                reward = 10
                done = True
                print("llego")
            elif mapa_np[index_state[next_state]] == 0:  # Si es una casilla vacía
                reward = -0.1
            else:  # Si es un obstáculo
                reward = -1

            # Seleccionar la próxima acción usando la política epsilon-greedy
            if np.random.rand() < epsilon:
                next_action = np.random.choice(num_actions)
            else:
                next_action = np.argmax(Q[next_state])

            # Actualizar la matriz Q usando la fórmula de SARSA
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

            if episodio % 50 ==0:
                move_robot(index_state[next_state])
            # Actualizar el estado actual
            state = next_state
            action = next_action

    return Q




def td_zero(mapa, alpha=0.1, gamma=0.9, epsilon=0.1, num_episodes=5000): #Demora mas porque le chante 5000 iteraciones
    # Inicialización de la matriz V con valores arbitrarios
    V = np.zeros(num_states)

    # Función para verificar si una posición está dentro del mapa y no es un obstáculo
    def is_valid_move(x, y):
        return 0 <= x < rows and 0 <= y < cols and mapa_np[x, y] != 'X'

    for episodio in range(num_episodes):
        state = state_index[(1, 5)]  # Empezamos desde el estado inicial <- Actualizar aleatoriamente esta wea cuando se implemente con pygame
        done = False

        while not done:
            # Mapear la acción a un movimiento (arriba, abajo, izquierda, derecha)
            x, y = index_state[state]
            possible_actions = []

            if is_valid_move(x - 1, y):  # Arriba
                possible_actions.append(state_index[(x - 1, y)])
            if is_valid_move(x + 1, y):  # Abajo
                possible_actions.append(state_index[(x + 1, y)])
            if is_valid_move(x, y - 1):  # Izquierda
                possible_actions.append(state_index[(x, y - 1)])
            if is_valid_move(x, y + 1):  # Derecha
                possible_actions.append(state_index[(x, y + 1)])
            # Obtener la recompensa del nuevo estado
            next_state = np.random.choice(possible_actions)  # Escoger un próximo estado al azar
            move_robot(index_state[next_state])
            # Obtener la recompensa del nuevo estado
            if mapa_np[index_state[next_state]] == 'M':  # Si llegamos a la meta
                reward = 10
                done = True
                print("llego")
            elif mapa_np[index_state[next_state]] == 0:  # Si es una casilla vacía
                reward = -0.1
            else:  # Si es un obstáculo
                reward = -1

            # Actualizar el valor del estado actual usando la fórmula TD(0)
            V[state] = V[state] + alpha * (reward + gamma * V[next_state] - V[state])
            # Actualizar el estado actual
            state = next_state
    return V

#Parte 2: Interfaz Grafica --------------------------------------------------------------
pygame.init()

# Dimensiones de la ventana
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Tamaño de las celdas de la matriz
CELL_SIZE = 50

# Crear la ventana
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

# Tamaño deseado para las texturas
TEXTURE_SIZE = (CELL_SIZE, CELL_SIZE)

# Cargar y escalar las texturas
texture_pasto = pygame.image.load('textures/grass.jpg').convert_alpha()
texture_pasto = pygame.transform.scale(texture_pasto, TEXTURE_SIZE)

texture_pared = pygame.image.load('textures/arbol.png').convert_alpha()
texture_pared = pygame.transform.scale(texture_pared, TEXTURE_SIZE)

texture_muralla = pygame.image.load('textures/rock.png').convert_alpha()
texture_muralla = pygame.transform.scale(texture_muralla, TEXTURE_SIZE)

texture_meta = pygame.image.load('textures/meta.png').convert_alpha()
texture_meta = pygame.transform.scale(texture_meta, TEXTURE_SIZE)

texture_robot = pygame.image.load('textures/robot.png').convert_alpha()
texture_robot = pygame.transform.scale(texture_robot, TEXTURE_SIZE)

# Función para dibujar la matriz con texturas
def draw_matrix(screen, matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            cell_rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            # Dibujar la textura de pasto primero
            screen.blit(texture_pasto, cell_rect)
            
            # Sobrepasar la textura de pared si corresponde
            if matrix[i][j] == 1:
                screen.blit(texture_pared, cell_rect)
            
            # Sobrepasar la textura de muralla si corresponde
            elif matrix[i][j] == 2:
                screen.blit(texture_muralla, cell_rect)
            
            elif matrix[i][j] == 3:
                screen.blit(texture_meta, cell_rect)

def move_robot(future_position):
    #x, y = future_position
    y, x = future_position
    draw_matrix(screen, matrix)
    robot_celda = pygame.Rect((x-1) * CELL_SIZE, (y-1) * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    screen.blit(texture_robot, robot_celda)
    pygame.display.flip()
    pygame.time.wait(100)
    
matrix = np.array([[1, 1, 1, 1, 0, 1, 1, 1, 1], #0: camino libre
                   [1, 1, 1, 1, 0, 1, 1, 1, 1], #1: pared
                   [1, 1, 1, 1, 0, 2, 0, 0, 0], #2: muralla
                   [0, 0, 0, 0, 0, 1, 0, 1, 1], #3: meta
                   [0, 1, 2, 1, 0, 1, 0, 1, 1],
                   [0, 3, 0, 1, 0, 2, 0, 1, 1],
                   [0, 1, 0, 1, 0, 1, 0, 1, 1],
                   [0, 2, 0, 0, 0, 1, 0, 1, 1],
                   [1, 1, 1, 1, 0, 0, 0, 1, 1],
                   [1, 1, 1, 1, 0, 1, 0, 1, 1],
                   [1, 1, 1, 1, 0, 1, 0, 1, 1],
                   [1, 1, 1, 1, 0, 2, 0, 1, 1]                 
])

screen.fill((255, 255, 255))  # Limpia la pantalla
draw_matrix(screen, matrix)

# Ejemplo de uso
#V, optimal_policy3, optimal_path3 = td_zero(mapa)
td_zero(mapa)
q_learning()

Q = sarsa()

print("Matriz Q actualizada:")
print(Q)
# Imprimir los valores de estado y la política óptima
#print("Valores de Estado (V):")
#print(V)
#print("\nPolítica Óptima:")
#print(optimal_policy3)
#print("\nRuta Óptima:")
#print(optimal_path3)

while True:
    #Eventos de pygame
    for event in pygame.event.get():

        #Evento de cierre
        if event.type == pygame.QUIT:
            pygame.quit()

        #Eventos de botones

    pygame.display.flip()
