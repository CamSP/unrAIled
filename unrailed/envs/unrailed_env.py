import time

import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pygame, sys, time, random
from pygame.surfarray import array3d
from pygame import display

class UnrailedEnv(gym.Env):
    def __init__(self):
        super(UnrailedEnv, self).__init__()

        self.width = 46
        self.heigth = 20
        self.cell_size = 20
        
        self.agent_pos1 = [9, 16]
        self.agent_pos2 = [9, 17]
        self.agent_pos3 = [9, 18]
        self.agent_pos4 = [9, 19]


        self.population = {
            0: {
                "name": "EMPTY",
                "color":pygame.Color(0, 0, 0)
                },
            1: {
                "name": "GRASS",
                "color":pygame.transform.scale(pygame.image.load("./assets/plain.png"), (self.cell_size, self.cell_size))
                },
            2: {
                "name": "TREE",
                "color":pygame.transform.scale(pygame.image.load("./assets/tree-a.png"), (self.cell_size, self.cell_size))
                },
            3: {
                "name": "ROCK",
                "color":pygame.transform.scale(pygame.image.load("./assets/iron.png"), (self.cell_size, self.cell_size))
                },
            4: {
                "name": "SOLID",
                "color":pygame.transform.scale(pygame.image.load("./assets/rock.png"), (self.cell_size, self.cell_size))
                },
            5: {
                "name": "WATHER",
                "color":pygame.transform.scale(pygame.image.load("./assets/water.png"), (self.cell_size, self.cell_size))
                },
            6: {
                "name": "INITIAL HOUSE",
                "color":pygame.transform.scale(pygame.image.load("./assets/station-l.png"), (self.cell_size, self.cell_size))
                },
            7: {
                "name": "FINAL HOUSE",
                "color":pygame.transform.scale(pygame.image.load("./assets/station-l.png"), (self.cell_size, self.cell_size))
                },
            7: {
                "name": "BRIDGE",
                "color":pygame.transform.scale(pygame.image.load("./assets/bridge.png"), (self.cell_size, self.cell_size))
                },
            8: {
                "name": "RAIL",
                "color":pygame.transform.scale(pygame.image.load("./assets/rail-rl.png"), (self.cell_size, self.cell_size))
                },
            "PLAYER": {
                "name": "PLAYER1",
                "color":pygame.transform.scale(pygame.image.load("./assets/face-1.png"), (self.cell_size, self.cell_size))
                },
        }
        self.grid = np.random.randint(low=0,high=len(self.population.keys())-1, size=(self.heigth ,self.width), dtype=np.uint8)
        self.display = pygame.display.set_mode((self.cell_size*self.width, self.cell_size*self.heigth))
        pygame.display.set_caption("Unrailed")
        
    def reset(self):
        self.agent_pos1 = [16, 9]
        return self.agent_pos1
        
    def step(self, action):
        if action == 0:  # Mover arriba
            self.agent_pos1[0] = max(0, self.agent_pos1[0] - 1)
        elif action == 1:  # Mover abajo
            self.agent_pos1[0] = min(self.heigth - 1, self.agent_pos1[0] + 1)
        elif action == 2:  # Mover izquierda
            self.agent_pos1[1] = max(0, self.agent_pos1[1] - 1)
        elif action == 3:  # Mover derecha
            self.agent_pos1[1] = min(self.width - 1, self.agent_pos1[1] + 1)
        
        return self.agent_pos1, 0, False, {}
    
    def render(self):
        self.display.fill((156, 215, 86))
        for i in range(self.heigth):
            for j in range(self.width):
                if self.grid[i, j] == 0:
                    pygame.draw.rect(self.display, (0, 0, 0), (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))
                else:
                    self.display.blit(self.population[self.grid[i, j]]["color"], [self.cell_size*j, self.cell_size*i])
                    
                # Draw players
                if [i, j] == self.agent_pos1:
                    self.display.blit(self.population["PLAYER"]["color"], [self.cell_size*j, self.cell_size*i])
        pygame.display.flip()


pygame.init()

# Crear el entorno Gym
env = UnrailedEnv()

# Bucle principal
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                _, _, done, _ = env.step(0)
            elif event.key == pygame.K_DOWN:
                _, _, done, _ = env.step(1)
            elif event.key == pygame.K_LEFT:
                _, _, done, _ = env.step(2)
            elif event.key == pygame.K_RIGHT:
                _, _, done, _ = env.step(3)
            if done:
                env.reset()
    
    env.render()
    
pygame.quit()