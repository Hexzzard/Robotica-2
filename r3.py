import numpy as np
import pygame
import random

# Definición del mapa y recompensas
matrix = np.array([
    [1, 1, 1, 1, 0, 1, 1, 1, 1], 
    [1, 1, 1, 1, 0, 1, 1, 1, 1], 
    [1, 1, 1, 1, 0, 2, 0, 0, 0], 
    [0, 0, 0, 0, 0, 1, 0, 1, 1], 
    [0, 1, 2, 1, 0, 1, 0, 1, 1],
    [0, 3, 0, 1, 0, 2, 0, 1, 1],
    [0, 1, 0, 1, 0, 1, 0, 1, 1],
    [0, 2, 0, 0, 0, 1, 0, 1, 1],
    [1, 1, 1, 1, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 0, 1, 0, 1, 1],
    [1, 1, 1, 1, 0, 1, 0, 1, 1],
    [1, 1, 1, 1, 0, 2, 0, 1, 1]                 
])

rewards = {
    0: -0.1,  # camino libre
    1: -1,    # pared (inaccesible)
    2: -1,    # obstáculo
    3: 10     # meta
}

# Definición de las acciones
actions = ["N", "S", "E", "O"]

# Probabilidad de éxito
success_prob = 0.9

# Función auxiliar para tomar acción
def take_action(state, action, matrix, rewards):
    row, col = state
    if action == "N":
        next_state = (row - 1, col) if row > 0 and matrix[row - 1, col] != 1 else state
    elif action == "S":
        next_state = (row + 1, col) if row < matrix.shape[0] - 1 and matrix[row + 1, col] != 1 else state
    elif action == "E":
        next_state = (row, col + 1) if col < matrix.shape[1] - 1 and matrix[row, col + 1] != 1 else state
    elif action == "O":
        next_state = (row, col - 1) if col > 0 and matrix[row, col - 1] != 1 else state

    reward = rewards[matrix[next_state]]
    done = matrix[next_state] == 3

    if random.uniform(0, 1) > success_prob:
        next_state = state

    return next_state, reward, done

# Algoritmo Q-Learning con visualización
def q_learning_visual(matrix, rewards, actions, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000):
    q_table = np.zeros((*matrix.shape, len(actions)))
    for episode in range(episodes):
        # Seleccionar un estado inicial aleatorio que no sea una pared
        state = (random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1))
        while matrix[state] == 1:
            state = (random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1))
        
        draw_matrix(screen, matrix)
        move_robot(state)
        display_message("Training Q-Learning")

        while True:
            if random.uniform(0, 1) < epsilon:
                action = random.choice(actions)
            else:
                action = actions[np.argmax(q_table[state])]

            next_state, reward, done = take_action(state, action, matrix, rewards)
            best_next_action = np.argmax(q_table[next_state])
            q_table[state][actions.index(action)] += alpha * (
                reward + gamma * q_table[next_state][best_next_action] - q_table[state][actions.index(action)]
            )

            move_robot(next_state)
            if done:
                break
            state = next_state
    return q_table

# Algoritmo SARSA con visualización
def sarsa_visual(matrix, rewards, actions, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000):
    q_table = np.zeros((*matrix.shape, len(actions)))
    for episode in range(episodes):
        # Seleccionar un estado inicial aleatorio que no sea una pared
        state = (random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1))
        while matrix[state] == 1:
            state = (random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1))
        
        action = random.choice(actions) if random.uniform(0, 1) < epsilon else actions[np.argmax(q_table[state])]
        
        draw_matrix(screen, matrix)
        move_robot(state)
        display_message("Training SARSA")

        while True:
            next_state, reward, done = take_action(state, action, matrix, rewards)
            next_action = random.choice(actions) if random.uniform(0, 1) < epsilon else actions[np.argmax(q_table[next_state])]
            q_table[state][actions.index(action)] += alpha * (
                reward + gamma * q_table[next_state][actions.index(next_action)] - q_table[state][actions.index(action)]
            )

            move_robot(next_state)
            if done:
                break
            state, action = next_state, next_action
    return q_table

# Algoritmo TD(0) con visualización
def td_zero_visual(matrix, rewards, actions, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000):
    value_table = np.zeros(matrix.shape)
    for episode in range(episodes):
        # Seleccionar un estado inicial aleatorio que no sea una pared
        state = (random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1))
        while matrix[state] == 1:
            state = (random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1))
        
        draw_matrix(screen, matrix)
        move_robot(state)
        display_message("Training TD(0)")

        while True:
            action = random.choice(actions) if random.uniform(0, 1) < epsilon else actions[np.argmax([value_table[take_action(state, a, matrix, rewards)[0]] for a in actions])]
            next_state, reward, done = take_action(state, action, matrix, rewards)
            value_table[state] += alpha * (reward + gamma * value_table[next_state] - value_table[state])

            move_robot(next_state)
            if done:
                break
            state = next_state
    return value_table

# Parte 2: Interfaz Gráfica con Pygame
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
texture_pasto = pygame.image.load('/home/gallo/Documents/GitHub/Robotica-2/textures/grass.jpg').convert_alpha()
texture_pasto = pygame.transform.scale(texture_pasto, TEXTURE_SIZE)

texture_pared = pygame.image.load('/home/gallo/Documents/GitHub/Robotica-2/textures/arbol.png').convert_alpha()
texture_pared = pygame.transform.scale(texture_pared, TEXTURE_SIZE)

texture_muralla = pygame.image.load('/home/gallo/Documents/GitHub/Robotica-2/textures/rock.png').convert_alpha()
texture_muralla = pygame.transform.scale(texture_muralla, TEXTURE_SIZE)

texture_meta = pygame.image.load('/home/gallo/Documents/GitHub/Robotica-2/textures/meta.png').convert_alpha()
texture_meta = pygame.transform.scale(texture_meta, TEXTURE_SIZE)

texture_robot = pygame.image.load('/home/gallo/Documents/GitHub/Robotica-2/textures/robot.png').convert_alpha()
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
    y, x = future_position
    draw_matrix(screen, matrix)
    robot_celda = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    screen.blit(texture_robot, robot_celda)
    pygame.display.flip()
    pygame.time.wait(50)  # Ajusta el tiempo de espera para ver el movimiento más claramente

def display_message(message):
    font = pygame.font.Font(None, 36)
    text = font.render(message, True, (255, 255, 255))
    text_rect = text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT - 20))
    screen.blit(text, text_rect)
    pygame.display.flip()

# Entrenar los algoritmos con visualización
print("Training Q-Learning...")
q_table = q_learning_visual(matrix, rewards, actions)

print("Training SARSA...")
sarsa_table = sarsa_visual(matrix, rewards, actions)

print("Training TD(0)...")
td_value_table = td_zero_visual(matrix, rewards, actions)

# Imprimir las tablas finales
print("Q-Table (Q-Learning):")      # 1000 Episodios cada uno, usar uno a la vez mientras
print(q_table)                      # no se implementen los botones
                                    # eliminando funciones de los otros 2
print("\nQ-Table (SARSA):")
print(sarsa_table)

print("\nValue Table (TD(0)):")
print(td_value_table)

# Visualizar las políticas aprendidas
def execute_policy(policy, actions, value_based=False):
    state = (random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1))
    while matrix[state] == 1:
        state = (random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1))
    
    draw_matrix(screen, matrix)
    move_robot(state)

    while matrix[state] != 3:
        if value_based:
            action_values = [policy[take_action(state, a, matrix, rewards)[0]] for a in actions]
            action = actions[np.argmax(action_values)]
        else:
            action = actions[np.argmax(policy[state])]

        next_state, _, done = take_action(state, action, matrix, rewards)
        move_robot(next_state)
        if done:
            break
        state = next_state

# Visualizar las políticas finales
print("\nVisualizando Q-Learning...")
display_message("Executing Q-Learning")
execute_policy(q_table, actions)
pygame.time.wait(2000)

print("Visualizando Sarsa...")
display_message("Executing Sarsa")
execute_policy(sarsa_table, actions)
pygame.time.wait(2000)

print("Visualizando TD(0)...")
display_message("Executing TD(0)")
execute_policy(td_value_table, actions, value_based=True)
pygame.time.wait(2000)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    pygame.display.flip()
