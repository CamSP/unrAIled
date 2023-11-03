import numpy as np
import gymnasium as gym
import pygame
from pygame.surfarray import array3d
from pygame import display

class ResourcesEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(ResourcesEnv, self).__init__()
        self.action_space = gym.spaces.MultiDiscrete(np.array([4, 4, 4, 4]))
        self.wood = 0
        self.rock = 0
        self.rail = 0
        self.r_front = 3
        self.temperature = 0
        self.current_obs = [self.wood, self.rock, self.rail, self.r_front, self.temperature]
        self.reward = 0
        self.time = 100
        self.display = None
        self.clock = None
        self.max_wood = 3
        self.max_rock = 3
        self.max_rail = 3
        self.max_r_front = 100
        self.max_temperature = 3
        self.current_action = [0, 0, 0, 0]
        self.observation_space = gym.spaces.MultiDiscrete(np.array([self.max_wood, self.max_rock, self.max_rail, self.max_r_front, self.max_temperature])+1)
        self.render_mode = render_mode

        
    def reset(self, seed=0):
        self.wood = 0
        self.rock = 0
        self.rail = 0
        self.r_front = 3
        self.temperature = 0
        self.reward = 0
        self.time = 100
        self.current_obs = [self.wood, self.rock, self.rail, self.r_front, self.temperature]
        return self.current_obs, {}

    def step(self, action):
        reward = 0
        done = False
        self.time = self.time - 1
        wood_action = 0
        rock_action = 1
        rail_action = 2
        cold_action = 3
        
        self.current_action = action

        #print("Actions [wood, rock, rail, cold]: ", str(action))
        # fabricacion de rail
        if self.wood > 0 and self.rock > 0 and self.rail < self.max_rail:
            self.rail = self.rail + 1
            self.wood = self.wood - 1
            self.rock = self.rock - 1
            reward = reward + 3

        # recolectar madera
        if wood_action in action and self.wood < self.max_wood:
            self.wood = self.wood + 1
            reward = reward + 1

        # recolectar roca
        if rock_action in action and self.rock < self.max_rock:
            self.rock = self.rock + 1
            reward = reward + 1
        
        # Poner railes
        if np.count_nonzero(np.array(action) == rail_action) > 1:
            reward = reward - 100
        if rail_action in action and self.rail == 3:
            self.r_front = self.r_front + self.rail
            reward = reward + 30
            self.rail = 0 
        
        # Descarrilmiento
        if self.r_front == 0:
            reward = reward - 10000
            done = True

        # Enfriar tren
        if np.count_nonzero(np.array(action) == cold_action) > 1:
            reward = reward - 100
            
        if self.temperature == 2 and cold_action in action:
            self.temperature = 0
            reward = reward + 20

        # Cambios de temperatura
        if np.random.randint(0, 5) == 1:
            self.temperature = self.temperature + 1
        
        # Explosion del tren
        if self.temperature == 3:
            reward = reward -10000
            done = True

        # Enfriar tren
        if self.temperature > 1 and cold_action in action:
            self.temperature = 0
            reward = reward + 5
        
        # Movimiento del tren 
        if np.random.randint(0, 5) == 1:
            self.r_front = self.r_front - 1

        if self.time == 0:
            done = True
        
        self.reward = self.reward + reward
        self.current_obs = [self.wood, self.rock, self.rail, self.r_front, self.temperature]
        return self.current_obs, self.reward, done, False,{}
    
    def render(self, mode=None):
        self.width = 700
        self.heigth = 400
        pygame.display.set_caption("Unrailed Centralizado")
        if mode != None:
            self.render_mode = mode
        
        if self.display is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.display = pygame.display.set_mode((self.width, self.heigth))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        
        
        self.font = pygame.font.Font('freesansbold.ttf', 30)
        text_wood = self.font.render('Wood', True, (255, 255, 255))
        text_n_wood = self.font.render(str(self.current_obs[0]), True, (255, 255, 255))
        text_rock = self.font.render('Rock', True, (255, 255, 255))
        text_n_rock = self.font.render(str(self.current_obs[1]), True, (255, 255, 255))
        text_rails = self.font.render('Rails', True, (255, 255, 255))
        text_n_rails = self.font.render(str(self.current_obs[2]), True, (255, 255, 255))
        text_front = self.font.render('Front', True, (255, 255, 255))
        text_n_front = self.font.render(str(self.current_obs[3]), True, (255, 255, 255))
        text_temp = self.font.render('Temp', True, (255, 255, 255))
        text_n_temp = self.font.render(str(self.current_obs[4]), True, (255, 255, 255))               
        text_reward = self.font.render('Reward', True, (255, 255, 255))
        text_n_reward = self.font.render(str(self.reward), True, (255, 255, 255))               
        text_action = self.font.render(str(self.current_action), True, (255, 255, 255))               
        text_time = self.font.render(str(self.time), True, (255, 255, 255))               
        
        
        self.display.fill((0, 0, 0))
        self.display.blit(text_wood, (50, 50))
        self.display.blit(text_n_wood, (50, 150))
        self.display.blit(text_rock, (150, 50))
        self.display.blit(text_n_rock, (150, 150))
        self.display.blit(text_rails, (250, 50))
        self.display.blit(text_n_rails, (250, 150))
        self.display.blit(text_front, (350, 50))
        self.display.blit(text_n_front, (350, 150))
        self.display.blit(text_temp, (450, 50))
        self.display.blit(text_n_temp, (450, 150))
        self.display.blit(text_reward, (550, 50))
        self.display.blit(text_n_reward, (550, 150))
        self.display.blit(text_action, (200, 300))
        self.display.blit(text_time, (500, 300))
        self.clock.tick(2)
        pygame.display.flip()
        # print("------------------------------")
        # print("Wood: " + str(self.current_obs[0]))
        # print("Rock: " + str(self.current_obs[1]))
        # print("Rails: " + str(self.current_obs[2]))
        # print("Front: " + str(self.current_obs[3]))
        # print("Temp: " + str(self.current_obs[4]))
        # print("reward: " + str(self.reward))
