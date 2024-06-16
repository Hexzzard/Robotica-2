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

#funcion que determina si la accion es valida 
def mov_is_valid(state, action, matrix):
    row, col = state
    #por cada accion se revisa que no se traspase los limites y no choque con algun obstaculo
    if action == "N":
        next_state = (row - 1, col) if row > 0 and matrix[row - 1, col] != 1 and matrix[row - 1, col] != 2 else False
    elif action == "S":
        next_state = (row + 1, col) if row < matrix.shape[0] - 1 and matrix[row + 1, col] != 1 and matrix[row + 1, col] != 2 else False
    elif action == "E":
        next_state = (row, col + 1) if col < matrix.shape[1] - 1 and matrix[row, col + 1] != 1 and matrix[row, col + 1] != 2 else False
    elif action == "O":
        next_state = (row, col - 1) if col > 0 and matrix[row, col - 1] != 1 and matrix[row, col - 1] != 2 else False

    return next_state

#funcion que calcula la siguiente posicion y su respectiva recompensa
def take_action(state, action, matrix, rewards):
    #revisamos que el movimiento sea valido
    next_state = mov_is_valid(state, action, matrix)

    if not next_state: #si no lo es, se queda en la misma posicion
        next_state = state
    
    #calculamos la recompensa y si es que llego a la meta
    reward = rewards[matrix[next_state]]
    done = matrix[next_state] == 3

    return next_state, reward, done

#funcion que posiciona al robot
def posicionar_robot():
    #ubicamos al robot en una posicion aleatoria
    state = (random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1))
    #si se posiciono sobre la meta o algun obstaculo, vuelve a posicionarlo en otra ubicacion aleatoria
    while matrix[state] == 1 or matrix[state] == 3 or matrix[state] == 2:
        state = (random.randint(0, matrix.shape[0] - 1), random.randint(0, matrix.shape[1] - 1))

    return state
    
#funcion del algoritmo Q-Learning
def q_learning(matrix, rewards, actions, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000):
    q_table = np.zeros((*matrix.shape, len(actions))) #creamos la q_table
    for episode in range(episodes): #iteramos por el rango de episodios

        #posicionamos al robot
        state = posicionar_robot()

        for paso in range(100):
            if random.uniform(0, 1) < epsilon: #robot explora
                action = random.choice(actions) #realiza una accion aleatoria
            else: #robot explota
                #toma el maximo valor de las acciones que puede realizar
                action = actions[np.argmax(q_table[state])] 
            
            #calculamos la siguiente posicion si se realiza la accion calculada
            next_state, reward, done = take_action(state, action, matrix, rewards)

            #actualizamos la q_table
            best_next_action = np.argmax(q_table[next_state])
            value = alpha * (reward + gamma * q_table[next_state][best_next_action] - q_table[state][actions.index(action)])
            q_table[state][actions.index(action)] += value

            if episode == episodes - 1: #solo se vera el movimiento de la ultima iteracion
                move_robot(next_state) #mover robot
            
            if done: #el episodio termina si llega a la meta
                break

            state = next_state #actualizamos el estado

    return q_table

#funcion del algoritmo SARSA
def sarsa(matrix, rewards, actions, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000):
    q_table = np.zeros((*matrix.shape, len(actions))) #creamos la q_table

    for episode in range(episodes):#iteramos por el rango de episodios
        #posicionar al robot (S)
        state = posicionar_robot()

        #seleccionar la siguiente accion (A)
        if random.uniform(0, 1) < epsilon: #robot explora
            action = random.choice(actions) #realiza una accion aleatoria
        else: #robot explota
            #toma el maximo valor de las acciones que puede realizar
            action = actions[np.argmax(q_table[state])]

        for paso in range(100):
            #realizar la accion y obtener la recompensa (R), y el siguiente estado (S')
            next_state, reward, done = take_action(state, action, matrix, rewards)

            #seleccionar la siguiente accion usando el ultimo estado (A')
            if random.uniform(0, 1) < epsilon: #robot explora
                next_action = random.choice(actions) #realiza una accion aleatoria
            else: #robot explota
                #toma el maximo valor de las acciones que puede realizar
                next_action = actions[np.argmax(q_table[next_state])]

            #actualizar la q_table
            value = alpha * (reward + gamma * q_table[next_state][actions.index(next_action)] - q_table[state][actions.index(action)])
            q_table[state][actions.index(action)] += value

            if episode == episodes - 1: #solo se vera el movimiento de la ultima iteracion
                move_robot(next_state) #mover robot

            if done: #si llego a la meta el ciclo termina
                break
            
            #actualizamos el estado (S) y la accion (A) actual. (son el S y A de la siguiente iteracion)
            state, action = next_state, next_action

    return q_table

#funcion del algoritmo TD(0)
def td_zero_visual(matrix, rewards, actions, alpha=0.7, gamma=0.9, epsilon=0.1, episodes=1000):
    value_table = np.zeros(matrix.shape) #creamos la value table

    for episode in range(episodes): #iteramos por el rango de episodios
        
        #posicionar al robot
        state = state = posicionar_robot()
        pdone = False

        for paso in range(100):
            if random.uniform(0, 1) < epsilon: #robot explora
                action = random.choice(actions) 

            else: #robot explota
                action_values = [] 
                for a in actions: #evaluara todos los posibles movimientos
                    mov = mov_is_valid(state, a, matrix)
                    if mov:
                        action_values.append([value_table[mov],a])
                #y se queda con el movimiento de mayor recompensa segun la value table
                action = sorted(action_values, key=lambda x: x[0], reverse=True)[0][1]

            next_state, reward, done = take_action(state, action, matrix, rewards)

            #actualizamos el valor de la value table
            value_table[state] += alpha * (reward + gamma * value_table[next_state] - value_table[state])
            
            if pdone:
                break

            if done: #necesitamos que cuando llege a la meta, actualice el valor de la value table
                pdone = True

            if episode == episodes - 1: #solo se vera el movimiento de la ultima iteracion
                move_robot(next_state) #mover robot
            
            state = next_state #actualizamos estado

    return value_table

#funcion que calcula la politica optima a partir de la q_table
def extract_policy_from_q_table(q_table, actions):
    #creamos un mapa en donde escribiremos la politica optima de cada estado
    policy = np.zeros(matrix.shape, dtype=str)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            #escribira una X o P si se trata de un obstaculo y un M para la meta
            if matrix[row, col] == 1:
                policy[row, col] = 'X'
            elif matrix[row, col] == 2:
                policy[row, col] = 'P'
            elif matrix[row, col] == 3:
                policy[row, col] = 'M'
            else:
                #si es un camino libre se escribira la accion que indica la q_table
                best_action = actions[np.argmax(q_table[(row, col)])]
                policy[row, col] = best_action
    return policy

#funcion que calcula la politica optima a partir de la value table
#PD: recordar que la value table solo contiene estado y valor
def extract_policy_from_value_table(value_table, actions):
    #creamos un mapa en donde escribiremos la politica optima de cada estado
    policy = np.zeros(matrix.shape, dtype=str)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            #escribira una X o P si se trata de un obstaculo y un M para la meta
            if matrix[row, col] == 1:
                policy[row, col] = 'X'
            elif matrix[row, col] == 2:
                policy[row, col] = 'P'
            elif matrix[row, col] == 3:
                policy[row, col] = 'M'
            else:
                #si es un camino libre se escribira la accion que otorge la mayor recompensa segun la value table
                action_values = []
                for a in actions:
                    mov = mov_is_valid((row, col), a, matrix)
                    if mov:
                        action_values.append([value_table[mov],a])
                policy[row, col] = sorted(action_values, key=lambda x: x[0], reverse=True)[0][1]

    return policy

# Parte 2: Interfaz Gráfica con Pygame
pygame.init()

#elementos del pygame
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
CELL_SIZE = 50

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
screen.fill((255, 255, 255))
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

#funcion que dibuja el mapa
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

#funcion que mueve al robot
def move_robot(future_position):
    y, x = future_position
    draw_matrix(screen, matrix)
    robot_celda = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    screen.blit(texture_robot, robot_celda)
    pygame.display.flip()
    pygame.time.wait(100)

#botones
color_boton = (255, 255, 255)  #color del boton
button_font = pygame.font.Font(None, 30) #fuente del boton

tamaño_head = pygame.Rect(CELL_SIZE*11, CELL_SIZE*1, 240, 40) #posicion y tamaño del boton
tamaño_boton1 = pygame.Rect(CELL_SIZE*10, CELL_SIZE*2.5, 240, 40)
tamaño_boton2 = pygame.Rect(CELL_SIZE*10, CELL_SIZE*4, 240, 40)
tamaño_boton3 = pygame.Rect(CELL_SIZE*10, CELL_SIZE*5.5, 240, 40)

head = button_font.render("Algoritmos", True, (0, 0, 0)) #texto
boton1 = button_font.render("Algoritmo Q-Learning", True, (0, 0, 0)) 
boton2 = button_font.render("Algoritmo SARSA", True, (0, 0, 0)) 
boton3 = button_font.render("Algoritmo TD(0)", True, (0, 0, 0)) 

pygame.draw.rect(screen, (0, 0, 0), tamaño_boton1, 2) #dibujar borde
pygame.draw.rect(screen, (0, 0, 0), tamaño_boton2, 2)
pygame.draw.rect(screen, (0, 0, 0), tamaño_boton3, 2)

screen.blit(head, (tamaño_head.x+10, tamaño_head.y+5))
screen.blit(boton1, (tamaño_boton1.x+10, tamaño_boton1.y+5))
screen.blit(boton2, (tamaño_boton2.x+10, tamaño_boton2.y+5))
screen.blit(boton3, (tamaño_boton3.x+10, tamaño_boton3.y+5))

draw_matrix(screen, matrix)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        
        #Eventos de botones
        elif event.type == pygame.MOUSEBUTTONDOWN:
            #cuando se presione el boton de su algoritmo llama a su funcion y se visualiza el movimiento
            if tamaño_boton1.collidepoint(pygame.mouse.get_pos()): #Q-learning
                print("Ejecutando Q-Learning...")
                q_table = q_learning(matrix, rewards, actions)
                print("Q-Table (Q-Learning):")
                print(np.around(q_table, decimals=2))
                q_policy = extract_policy_from_q_table(q_table, actions)
                print("\nQ-Learning Policy:")
                print(q_policy)

            if tamaño_boton2.collidepoint(pygame.mouse.get_pos()): #SARSA
                print("Ejecutando SARSA...")
                sarsa_table = sarsa(matrix, rewards, actions)
                print("\nQ-Table (SARSA):")
                print(np.around(sarsa_table, decimals=2))
                sarsa_policy = extract_policy_from_q_table(sarsa_table, actions)
                print("\nSARSA Policy:")
                print(sarsa_policy)

            if tamaño_boton3.collidepoint(pygame.mouse.get_pos()): #TD(0)
                print("Ejecutando TD(0)...")
                td_value_table = td_zero_visual(matrix, rewards, actions)
                print("\nValue Table (TD(0)):")
                print(np.around(td_value_table, decimals=2))
                td_policy = extract_policy_from_value_table(td_value_table, actions)
                print("\nTD(0) Policy:")
                print(td_policy)

    pygame.display.flip()