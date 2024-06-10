import numpy as np

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

def q_learning(mapa, alpha=0.1, gamma=0.9, epsilon=0.1, num_episodes=5000):
    # Inicialización de la matriz Q con valores arbitrarios
    Q = np.zeros((num_states, num_actions))

    # Función para verificar si una posición está dentro del mapa y no es un obstáculo
    def is_valid_move(x, y):
        return 0 <= x < rows and 0 <= y < cols and mapa_np[x, y] != 'X'

    for _ in range(num_episodes):
        state = state_index[(1, 5)]  # Empezamos desde el estado inicial
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
            elif mapa_np[index_state[next_state]] == 0:  # Si es una casilla vacía
                reward = -0.1
            else:  # Si es un obstáculo
                reward = -1

            # Actualizar la matriz Q
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

            # Actualizar el estado actual
            state = next_state

    # Derivar la política óptima a partir de la matriz Q entrenada
    def get_optimal_policy():
        state = state_index[(1, 5)]
        path = [(1, 5)]
        directions = []

        while mapa_np[index_state[state]] != 'M':
            action = np.argmax(Q[state])
            x, y = index_state[state]
            if action == 0 and is_valid_move(x - 1, y):
                next_state = state_index[(x - 1, y)]
                directions.append('N')
            elif action == 1 and is_valid_move(x + 1, y):
                next_state = state_index[(x + 1, y)]
                directions.append('S')
            elif action == 2 and is_valid_move(x, y - 1):
                next_state = state_index[(x, y - 1)]
                directions.append('O')
            elif action == 3 and is_valid_move(x, y + 1):
                next_state = state_index[(x, y + 1)]
                directions.append('E')
            else:
                next_state = state
            path.append(index_state[next_state])
            state = next_state

        return path, directions

    return get_optimal_policy()

# Llamar a la función q_learning y obtener la política óptima
optimal_path, optimal_directions = q_learning(mapa)

print("Camino óptimo:", optimal_path)
print("Direcciones óptimas:", optimal_directions)

def sarsa(mapa, alpha=0.1, gamma=0.9, epsilon=0.1, num_episodes=5000):
    # Inicialización de la matriz Q con valores arbitrarios
    Q = np.zeros((num_states, num_actions))

    # Función para verificar si una posición está dentro del mapa y no es un obstáculo
    def is_valid_move(x, y):
        return 0 <= x < rows and 0 <= y < cols and mapa_np[x, y] != 'X'

    for _ in range(num_episodes):
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

            # Actualizar el estado y la acción actuales
            state = next_state
            action = next_action

    # Derivar la política óptima a partir de la matriz Q entrenada
    def get_optimal_policy():
        state = state_index[(1, 5)]
        path = [(1, 5)]
        directions = []

        while mapa_np[index_state[state]] != 'M':
            action = np.argmax(Q[state])
            x, y = index_state[state]
            if action == 0 and is_valid_move(x - 1, y):
                next_state = state_index[(x - 1, y)]
                directions.append('N')
            elif action == 1 and is_valid_move(x + 1, y):
                next_state = state_index[(x + 1, y)]
                directions.append('S')
            elif action == 2 and is_valid_move(x, y - 1):
                next_state = state_index[(x, y - 1)]
                directions.append('O')
            elif action == 3 and is_valid_move(x, y + 1):
                next_state = state_index[(x, y + 1)]
                directions.append('E')
            else:
                next_state = state
            path.append(index_state[next_state])
            state = next_state

        return path, directions

    return get_optimal_policy()

# Llamar a la función sarsa y obtener la política óptima
optimal_path2, optimal_directions2 = sarsa(mapa)

print("Camino óptimo:", optimal_path2)
print("Direcciones óptimas:", optimal_directions2)




def td_zero(mapa, alpha=0.1, gamma=0.9, epsilon=0.1, num_episodes=5000): #Demora mas porque le chante 5000 iteraciones
    # Inicialización de la matriz V con valores arbitrarios
    V = np.zeros(num_states)

    # Función para verificar si una posición está dentro del mapa y no es un obstáculo
    def is_valid_move(x, y):
        return 0 <= x < rows and 0 <= y < cols and mapa_np[x, y] != 'X'

    for _ in range(num_episodes):
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

            next_state = np.random.choice(possible_actions)  # Escoger un próximo estado al azar

            # Obtener la recompensa del nuevo estado
            if mapa_np[index_state[next_state]] == 'M':  # Si llegamos a la meta
                reward = 10
                done = True
            elif mapa_np[index_state[next_state]] == 0:  # Si es una casilla vacía
                reward = -0.1
            else:  # Si es un obstáculo
                reward = -1

            # Actualizar el valor del estado actual usando la fórmula TD(0)
            V[state] = V[state] + alpha * (reward + gamma * V[next_state] - V[state])

            # Actualizar el estado actual
            state = next_state
    # Derivar la política óptima a partir de los valores de estado
    def get_optimal_policy():
        state = state_index[(1, 5)]
        path = [(1, 5)]
        directions = []

        while mapa_np[index_state[state]] != 'M':
            x, y = index_state[state]
            best_value = float('-inf')
            best_action = None
            for dx, dy, action in [(-1, 0, 'N'), (1, 0, 'S'), (0, -1, 'O'), (0, 1, 'E')]:
                if is_valid_move(x + dx, y + dy):
                    next_state = state_index[(x + dx, y + dy)]
                    if V[next_state] > best_value:
                        best_value = V[next_state]
                        best_action = action
            directions.append(best_action)
            x, y = index_state[state]
            if best_action == 'N':
                state = state_index[(x - 1, y)]
            elif best_action == 'S':
                state = state_index[(x + 1, y)]
            elif best_action == 'O':
                state = state_index[(x, y - 1)]
            elif best_action == 'E':
                state = state_index[(x, y + 1)]
            path.append(index_state[state])
            path.append(index_state[state])

        return directions, path

    # Obtener la política óptima
    optimal_policy, optimal_path = get_optimal_policy()

    return V, optimal_policy, optimal_path

# Ejemplo de uso
V, optimal_policy3, optimal_path3 = td_zero(mapa)

# Imprimir los valores de estado y la política óptima
print("Valores de Estado (V):")
print(V)
print("\nPolítica Óptima:")
print(optimal_policy3)
print("\nRuta Óptima:")
print(optimal_path3)