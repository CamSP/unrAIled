import time

import os

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
        self.PATH = os.path.dirname(os.path.abspath(__file__))
        self.width = 46
        self.heigth = 20
        self.cell_size = 20
        
        self.agents_pos = [
            #[y, x, deg, item, #items]
            [9, 16, 0, 0, 0],
            [10, 16, 0, 0, 0],
            [11, 16, 0, 0, 0],
            [12, 16, 0, 0, 0],
        ]


        self.population = {
            0: {
                "name": "EMPTY",
                "color":pygame.Color(0, 0, 0)
                },
            1: {
                "name": "GRASS",
                "color":pygame.transform.scale(pygame.image.load(self.PATH + "/assets/plain.png"), (self.cell_size, self.cell_size))
                },
            2: {
                "name": "TREE",
                "color":pygame.transform.scale(pygame.image.load(self.PATH + "/assets/tree-a.png"), (self.cell_size, self.cell_size))
                },
            3: {
                "name": "ROCK",
                "color":pygame.transform.scale(pygame.image.load(self.PATH + "/assets/iron.png"), (self.cell_size, self.cell_size))
                },
            4: {
                "name": "SOLID",
                "color":pygame.transform.scale(pygame.image.load(self.PATH + "/assets/rock.png"), (self.cell_size, self.cell_size))
                },
            5: {
                "name": "WATHER",
                "color":pygame.transform.scale(pygame.image.load(self.PATH + "/assets/water.png"), (self.cell_size, self.cell_size))
                },
            6: {
                "name": "HOUSE",
                "color":pygame.transform.scale(pygame.image.load(self.PATH + "/assets/station-l.png"), (self.cell_size, self.cell_size))
                },
            7: {
                "name": "BRIDGE",
                "color":pygame.transform.scale(pygame.image.load(self.PATH + "/assets/bridge.png"), (self.cell_size, self.cell_size))
                },
            8: {
                "name": "RAIL",
                "color":pygame.transform.scale(pygame.image.load(self.PATH + "/assets/rail-rl.png"), (self.cell_size, self.cell_size))
                },
            "PLAYER0": {
                "name": "PLAYER0",
                "color":pygame.transform.scale(pygame.image.load(self.PATH + "/assets/face-1.png"), (self.cell_size, self.cell_size))
                },
            "PLAYER1": {
                "name": "PLAYER1",
                "color":pygame.transform.scale(pygame.image.load(self.PATH + "/assets/face-2.png"), (self.cell_size, self.cell_size))
                },
            "PLAYER2": {
                "name": "PLAYER2",
                "color":pygame.transform.scale(pygame.image.load(self.PATH + "/assets/face-3.png"), (self.cell_size, self.cell_size))
                },
            "PLAYER3": {
                "name": "PLAYER3",
                "color":pygame.transform.scale(pygame.image.load(self.PATH + "/assets/face-4.png"), (self.cell_size, self.cell_size))
                },
        }

        self.items = {
            1:{
                "name": "WOOD",
                "color": pygame.transform.scale(pygame.image.load(self.PATH + "/assets/resource-wood.png"), (self.cell_size, self.cell_size))
            },
            2:{
                "name": "IRON",
                "color": pygame.transform.scale(pygame.image.load(self.PATH + "/assets/resource-iron.png"), (self.cell_size, self.cell_size))
            },
            3:{
                "name": "AXE",
                "color": pygame.transform.scale(pygame.image.load(self.PATH + "/assets/resource-axe.png"), (self.cell_size, self.cell_size))
            },
            4:{
                "name": "PICKAXE",
                "color": pygame.transform.scale(pygame.image.load(self.PATH + "/assets/resource-pickaxe.png"), (self.cell_size, self.cell_size))
            },
            5:{
                "name": "BACKET",
                "color": pygame.transform.scale(pygame.image.load(self.PATH + "/assets/resource-backet.png"), (self.cell_size, self.cell_size))
            },
            6:{
                "name": "EMPTY_BACKET",
                "color": pygame.transform.scale(pygame.image.load(self.PATH + "/assets/resource-backet-empty.png"), (self.cell_size, self.cell_size))
            },
        }

        self.valid_blocks = [1, 7, 8]
        self.grid = np.array([
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [3,3,3,3,3,3,3,3,3,3,3,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [4,4,4,4,4,4,4,4,4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [5,5,5,5,5,5,5,5,5,5,5,1,1,1,1,1,1,1,1,1,1,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [6,6,6,6,6,6,6,6,6,6,6,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [7,7,7,7,7,7,7,7,7,7,7,1,1,1,1,1,1,1,1,1,1,1,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [8,8,8,8,8,8,8,8,8,8,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,6,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,7,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        ])

        #self.grid = np.random.randint(low=0,high=len(self.population.keys())-1, size=(self.heigth ,self.width), dtype=np.uint8)
        #print(self.grid.shape)
        self.display = pygame.display.set_mode((self.cell_size*self.width, self.cell_size*self.heigth))
        pygame.display.set_caption("Unrailed")
        
    def reset(self):
        self.agents_pos = np.asarray([
            #[y, x, deg, item, #items]
            [9, 16, 0, 0, 0],
            [10, 16, 0, 0, 0],
            [11, 16, 0, 0, 0],
            [12, 16, 0, 0, 0],
        ])
        return self.agents_pos

        
    def step(self, action, player):
        if action == 0:
            self.agents_pos[player][2] = 90
            if self.agents_pos[player][0] == 0:
                self.agents_pos[player][0] = max(0, self.agents_pos[player][0] - 1)
            elif self.grid[self.agents_pos[player][0]-1, self.agents_pos[player][1]] in self.valid_blocks:
                self.agents_pos[player][0] = self.agents_pos[player][0] - 1
        if action == 1:
            self.agents_pos[player][2] = 270
            if self.agents_pos[player][0] == self.heigth-1:
                self.agents_pos[player][0] = min(self.heigth - 1, self.agents_pos[player][0] + 1)  
            elif self.grid[self.agents_pos[player][0]+1, self.agents_pos[player][1]] in self.valid_blocks:
                self.agents_pos[player][0] = self.agents_pos[player][0] + 1
        if action == 2:
            self.agents_pos[player][2] = 180
            if self.agents_pos[player][1] == 0:
                self.agents_pos[player][1] = max(0, self.agents_pos[player][1] - 1)
            elif self.grid[self.agents_pos[player][0], self.agents_pos[player][1]-1] in self.valid_blocks:
                self.agents_pos[player][1] = self.agents_pos[player][1] - 1
        if action == 3:
            self.agents_pos[player][2] = 0
            if self.agents_pos[player][1] == self.width-1:
                self.agents_pos[player][1] = min(self.width - 1, self.agents_pos[player][1] + 1)
            elif self.grid[self.agents_pos[player][0], self.agents_pos[player][1]+1] in self.valid_blocks:
                self.agents_pos[player][1] = self.agents_pos[player][1] + 1
            
        
        return self.agents_pos, 0, False, {}
    
    def render(self):
        self.display.fill((156, 215, 86))
        for i in range(self.heigth):
            for j in range(self.width):
                if self.grid[i, j] == 0:
                    pygame.draw.rect(self.display, (0, 0, 0), (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size))
                else:
                    self.display.blit(self.population[self.grid[i, j]]["color"], [self.cell_size*j, self.cell_size*i])
                    
                # Draw players
                if [i, j] == self.agents_pos[0][:2]:
                    self.display.blit(pygame.transform.rotate(self.population["PLAYER0"]["color"], self.agents_pos[0][2]), [self.cell_size*j, self.cell_size*i])
                elif [i, j] == self.agents_pos[1][:2]:
                    self.display.blit(pygame.transform.rotate(self.population["PLAYER1"]["color"], self.agents_pos[1][2]), [self.cell_size*j, self.cell_size*i])
                elif [i, j] == self.agents_pos[2][:2]:
                    self.display.blit(pygame.transform.rotate(self.population["PLAYER2"]["color"], self.agents_pos[2][2]), [self.cell_size*j, self.cell_size*i])
                elif [i, j] == self.agents_pos[3][:2]:
                    self.display.blit(pygame.transform.rotate(self.population["PLAYER3"]["color"], self.agents_pos[3][2]), [self.cell_size*j, self.cell_size*i])
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
                _, _, done, _ = env.step(0, 0)
            elif event.key == pygame.K_DOWN:
                _, _, done, _ = env.step(1, 0)
            elif event.key == pygame.K_LEFT:
                _, _, done, _ = env.step(2, 0)
            elif event.key == pygame.K_RIGHT:
                _, _, done, _ = env.step(3, 0)
            if done:
                env.reset()
    
    env.render()
    
pygame.quit()