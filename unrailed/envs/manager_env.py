import time

import os

import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pygame, sys, time, random
from pygame.surfarray import array3d
from pygame import display

class ResourcesEnv(gym.Env):
    def __init__(self):
        super(ResourcesEnv, self).__init__()
        self.cell_size = 20
        self.wood = 0
        self.rock = 0
        self.rail = 0
        self.r_front = 1
        self.temperature = 0
        self.data = [self.wood, self.rock, self.rail, self.r_front, self.temperature]
        self.reward = 0
        
    def reset(self):
        
        pass

        
    def step(self, action):
        reward = 0
        wood_action = 0
        rock_action = 1
        rail_action = 2
        cold_action = 3

        # recolectar madera
        if wood_action in action and self.wood < 3:
            self.wood = self.wood + 1
            reward = reward + 1

        # recolectar roca
        if rock_action in action and self.rock < 3:
            self.rock = self.rock + 1
            reward = reward + 1
        
        # fabricacion de rail
        if self.wood > 0 and self.rock > 0 and self.rail < 3:
            self.rail = self.rail + 1
            self.wood = self.wood - 1
            self.rock = self.rock - 1
            reward = reward + 3

        # Poner railes
        if rail_action in action and self.rail > 0:
            self.r_front = self.r_front + self.rail
            reward = reward + self.rail*2
            self.rail = 0 
            

        # Enfriar tren
        if self.temperature != 0 and cold_action in action:
            self.temperature = 0
            reward = reward + 5

        # Movimiento del tren, Cambios de temperatura
        

        
        self.reward = self.reward + reward
        self.data = [self.wood, self.rock, self.rail, self.r_front, self.temperature]
        return self.data, 0, False, {}
    
    def render(self):
        pass


pygame.init()

# Crear el entorno Gym
env = ResourcesEnv()

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