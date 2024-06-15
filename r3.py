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
    0: -0.1,  # Camino libre
    1: -1,    # Pared (inaccesible)
    2: -1,    # Obstáculo
    3: 10     # Meta
}

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

    return next_state, reward, done

# Algoritmo Q-Learning
def q_learning(matrix, rewards, actions, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000):
    q_table = np.zeros((*matrix.shape, len(actions)))
    for episode in range(episodes):
        state = (random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1))
        while matrix[state] == 1:
            state = (random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1))

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

            if done:
                break
            state = next_state

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes} completed for Q-Learning.")
    return q_table

# Algoritmo SARSA
def sarsa(matrix, rewards, actions, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000):
    q_table = np.zeros((*matrix.shape, len(actions)))
    for episode in range(episodes):
        state = (random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1))
        while matrix[state] == 1:
            state = (random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1))

        action = random.choice(actions) if random.uniform(0, 1) < epsilon else actions[np.argmax(q_table[state])]

        while True:
            next_state, reward, done = take_action(state, action, matrix, rewards)
            next_action = random.choice(actions) if random.uniform(0, 1) < epsilon else actions[np.argmax(q_table[next_state])]
            q_table[state][actions.index(action)] += alpha * (
                reward + gamma * q_table[next_state][actions.index(next_action)] - q_table[state][actions.index(action)]
            )

            if done:
                break
            state, action = next_state, next_action

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes} completed for SARSA.")
    return q_table

# Algoritmo TD(0) con visualización solo en el último episodio
def td_zero_visual(matrix, rewards, actions, alpha=0.7, gamma=0.9, epsilon=0.1, episodes=1000):
    value_table = np.zeros(matrix.shape)
    for episode in range(episodes):
        state = (random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1))
        while matrix[state] == 1 or matrix[state] == 3:
            state = (random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1))
        previous_state = None  # Para verificar si el robot se queda quieto
        pdone = False
        for t in range(1000):  # Máximo de 1000 pasos por episodio
            if random.uniform(0, 1) < epsilon:
                action = random.choice(actions)
            else:
                action_values = []
                for a in actions:
                    paction = take_action(state, a, matrix, rewards)[0]
                    if paction[0] != state:
                        action_values.append([value_table[paction],a])
                action = sorted(action_values, key=lambda x: x[0], reverse=True)[0][1]

            next_state, reward, done = take_action(state, action, matrix, rewards)

            # Evitar quedarse en el mismo lugar
            if next_state == previous_state:
                continue

            value_table[state] += alpha * (reward + gamma * value_table[next_state] - value_table[state])

            if episode == episodes - 1:  # Solo mostrar la visualización en el último episodio
                move_robot(next_state)  # Mover el robot visualmente
            if pdone:
                break
            if done:
                pdone = True
            previous_state = state
            state = next_state

    print(np.around(value_table, decimals=2))
    return value_table

# Parte 2: Interfaz Gráfica con Pygame
pygame.init()

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
CELL_SIZE = 50

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

TEXTURE_SIZE = (CELL_SIZE, CELL_SIZE)

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

def draw_matrix(screen, matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            cell_rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            screen.blit(texture_pasto, cell_rect)
            if matrix[i][j] == 1:
                screen.blit(texture_pared, cell_rect)
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
    pygame.time.wait(50)

# Entrenar los algoritmos sin visualización durante el entrenamiento
print("Training Q-Learning...")
#q_table = q_learning(matrix, rewards, actions)

print("Training SARSA...")
#sarsa_table = sarsa(matrix, rewards, actions)

print("Training TD(0)...")
td_value_table = td_zero_visual(matrix, rewards, actions)  # Entrenamiento sin visualización

# Imprimir las tablas finales
print("Q-Table (Q-Learning):")
#print(q_table)

print("\nQ-Table (SARSA):")
#print(sarsa_table)

print("\nValue Table (TD(0)):")
#print(td_value_table)

def extract_policy_from_q_table(q_table, actions):
    policy = np.zeros(matrix.shape, dtype=str)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if matrix[row, col] == 1:
                policy[row, col] = 'X'
            elif matrix[row, col] == 3:
                policy[row, col] = 'M'
            else:
                best_action = actions[np.argmax(q_table[(row, col)])]
                policy[row, col] = best_action
    return policy

def extract_policy_from_value_table(value_table, actions):
    policy = np.zeros(matrix.shape, dtype=str)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if matrix[row, col] == 1:
                policy[row, col] = 'X'
            elif matrix[row, col] == 3:
                policy[row, col] = 'M'
            else:
                action_values = [value_table[take_action((row, col), a, matrix, rewards)[0]] for a in actions]
                best_action = actions[np.argmax(action_values)]
                policy[row, col] = best_action
    return policy

#q_policy = extract_policy_from_q_table(q_table, actions)
#sarsa_policy = extract_policy_from_q_table(sarsa_table, actions)
td_policy = extract_policy_from_value_table(td_value_table, actions)

# Imprimir políticas para verificar razonabilidad
print("\nQ-Learning Policy:")
#print(q_policy)

print("\nSARSA Policy:")
#print(sarsa_policy)

print("\nTD(0) Policy:")
print(td_policy)

def execute_policy(policy, actions, value_based=False):
    state = (random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1))
    while matrix[state] == 1:
        state = (random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1))
    
    draw_matrix(screen, matrix)
    move_robot(state)

    steps = 0
    max_steps = 100  # Evitar ciclos infinitos

    while matrix[state] != 3 and steps < max_steps:
        if value_based:
            action_values = [policy[take_action(state, a, matrix, rewards)[0]] for a in actions]
            action = actions[np.argmax(action_values)]
        else:
            action = policy[state]

        next_state, _, done = take_action(state, action, matrix, rewards)
        move_robot(next_state)
        if done:
            break
        state = next_state
        steps += 1

    if matrix[state] == 3:
        print("Robot reached the goal!")
    else:
        print("Robot did not reach the goal within the step limit.")

# Visualizar las políticas finales
print("\nVisualizando Q-Learning...")
execute_policy(q_policy, actions)
pygame.time.wait(2000)

print("Visualizando Sarsa...")
execute_policy(sarsa_policy, actions)
pygame.time.wait(2000)

# Visualizar la exploración de TD(0)
print("Visualizando TD(0)...")
td_zero_visual(matrix, rewards, actions)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    pygame.display.flip()
