import time

import os

import numpy as np

import functools
import gymnasium
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Dict


import pygame, sys, time, random
from pygame.surfarray import array3d
from pygame import display
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3
GRAP = 4
BREAK = 5

MOVES = ["UP", "DOWN", "RIGHT", "LEFT", "GRAP", "BREAK"]
NUM_ITERS = 100

class UnrailedEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "unrailed"}

    def __init__(self, render_mode=None, curriculum="move"):
        if render_mode == "human":
            pygame.init()
        self.PATH = os.path.dirname(os.path.abspath(__file__))
        self.width = 46
        self.heigth = 20
        self.cell_size = 20
        self.population = {
            0: {
                "name": "PLAYER",
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
                "name": "WATER",
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
            9:{
                "name": "WOOD",
                "color": pygame.transform.scale(pygame.image.load(self.PATH + "/assets/resource-wood.png"), (self.cell_size, self.cell_size))
            },
            10:{
                "name": "IRON",
                "color": pygame.transform.scale(pygame.image.load(self.PATH + "/assets/resource-iron.png"), (self.cell_size, self.cell_size))
            },
            11:{
                "name": "AXE",
                "color": pygame.transform.scale(pygame.image.load(self.PATH + "/assets/resource-axe.png"), (self.cell_size, self.cell_size))
            },
            12:{
                "name": "PICKAXE",
                "color": pygame.transform.scale(pygame.image.load(self.PATH + "/assets/resource-pickaxe.png"), (self.cell_size, self.cell_size))
            },
            13:{
                "name": "BUCKET",
                "color": pygame.transform.scale(pygame.image.load(self.PATH + "/assets/resource-backet.png"), (self.cell_size, self.cell_size))
            },
            14:{
                "name": "EMPTY_BUCKET",
                "color": pygame.transform.scale(pygame.image.load(self.PATH + "/assets/resource-backet-empty.png"), (self.cell_size, self.cell_size))
            },
            15:{
                "name": "ENGINE",
                "color": pygame.transform.scale(pygame.image.load(self.PATH + "/assets/engine.png"), (self.cell_size, self.cell_size))
            },
            16:{
                "name": "CRAFTING",
                "color": pygame.transform.scale(pygame.image.load(self.PATH + "/assets/CraftingWagon.png"), (self.cell_size, self.cell_size))
            },
            17:{
                "name": "STORAGE",
                "color": pygame.transform.scale(pygame.image.load(self.PATH + "/assets/StorageWagon.png"), (self.cell_size, self.cell_size))
            },
            18:{
                "name": "TANK",
                "color": pygame.transform.scale(pygame.image.load(self.PATH + "/assets/TankWagon.png"), (self.cell_size, self.cell_size))
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
        
        
        
        self.grid = np.array([
            [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,2,2,2,2],
            [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,2,2,2,2,2],
            [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,2,2,2,2,2,2,2,2],
            [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,4,4,4,4,4,4,4,4,4,4,4,4,4,2,2,2,2,2,2,2,2,2,2,2,2],
            [2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,4,4,4,4,4,4,4,4,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2],
            [2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2],
            [1,1,1,1,1,1,1,1,1,1,1,12,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,6,6,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,8,8,1,1],
            [1,1,1,1,6,6,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,11,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [2,2,2,3,3,3,1,1,1,1,1,1,1,14,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [2,2,3,3,3,3,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,5,5,1,1,1,1,1,1,1,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1],
            [2,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,1,1,1,5,5,5,5,5,1,1,1,2,2,2,2,2,2,2,3,3,3,3,1,1,1,1,1,1,1],
            [3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,5,5,5,5,5,5,5,5,5,5,5,2,2,2,3,3,3,3,3,3,3,1,1,1,1,1,1],
            [3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,5,5,5,5,5,5,5,5,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3],
            [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,5,5,5,5,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3],
            [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3],
            [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3]
        ])


        
        self.agents = ["player_" + str(r) for r in range(4)]
        
                            #arriba, derecha, abajo, izquierda   
        self.orientations = [0, 1, 2, 3]
        self.actions = [0, 1, 2, 3, 4, 5]
        self.breakeble = [2, 3]
        self.pickeable = [8, 9, 10, 11, 12, 13, 14]
        self.valid_blocks = [1, 7, 8, 9, 10, 11, 12, 13, 14]
        self.path = [(9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8)]
        self.final = (7, 42)
            
        self.train = {
            "ENGINE": (9, 5),
            "TANK": (9, 4),
            "CRAFTING": (9, 3),
            "STORAGE": (9, 2),
        }
        
        for rail in self.path:
            self.grid[rail[0]][rail[1]] = 8
            
        self.grid[self.train["ENGINE"][0]][self.train["ENGINE"][1]] = 15
        self.grid[self.train["TANK"][0]][self.train["TANK"][1]] = 18
        self.grid[self.train["CRAFTING"][0]][self.train["CRAFTING"][1]] = 17
        self.grid[self.train["STORAGE"][0]][self.train["STORAGE"][1]] = 16
        self.grid[self.final[0]][self.final[1]] = 8
        
        self.state = {
            "temp": 0,
            "wood": 0,
            "iron": 0,
            "storage": 3,
            "front": 2
        }
                    
        self.valid_items = {
            0: [9, 11],
            1: [10, 12],
            2: [8],
            3: [13, 14]
        }
        self.valid_actions_blocks = {
            0: [2],
            1: [3],
            2: [8],
            3: [5]
        }
        self.valid_action_targets = {
            0: [2],
            1: [3],
            2: [18],
            3: [5],
        }
        self.drops = {
            2: 9,
            3: 10,
        }
        self.tools = [11, 12, 13, 14]
        self.orientations = [0, 90, 180, 270]
        self.valid_tools_targets = {
            11: [2],
            12: [3],
            13: [18],
            14: [5],
        }
        
        
        self.agents_pos = [
            #[y, x, deg, item]
            [9, 16, 0],
            [10, 16, 0],
            [11, 16, 0],
            [12, 16, 0],
        ]
        
        self.rewards_values = {
            "pick_item": 10,
            "break_block": 10,
            "fill_bucket": 10,
            "pick_same_item": -100,
            "pick_wrong_item": -100,
            "complete_task": 1000,
        }

        self.render_mode = render_mode
        
        self.rewards = {agent: 0 for agent in self.agents}
        
        self.observations = {
            self.agents[i]: {
            "vision": self.getAgentView(self.agents_pos[i][0], self.agents_pos[i][1]),
            "position": [self.agents_pos[i][0], self.agents_pos[i][1]],
            "orientation": self.agents_pos[i][2],
            "item": 0,
            "wood_item": [0, 0],
            "rock_item": [0, 0],
            "pickaxe": [0, 0],
            "axe": [0, 0],
            "bucket": [0, 0],
            "rock_block": [0, 0],
            "tree_block": [0, 0],
            "water": [0, 0],
            "refri": [0, 0],
            "deposit": [0, 0],
            "factory": [0, 0],
            "riel": [0, 0],
            "house": [0, 0],
            "TASK": 0,
        } for i in range(len(self.agents)) 
        }
        
        self.display = pygame.display.set_mode((self.cell_size*self.width, self.cell_size*self.heigth))
        pygame.display.set_caption("Unrailed")

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        obs = Dict({
            "vision": Box(low=0, high=15, shape=(5,5)),
            "position": MultiDiscrete([45, 20]),
            "orientation": Discrete(4),
            "item": Discrete(7),
            "wood_item": MultiDiscrete([45, 20]),
            "rock_item": MultiDiscrete([45, 20]),
            "pickaxe": MultiDiscrete([45, 20]),
            "axe": MultiDiscrete([45, 20]),
            "bucket": MultiDiscrete([45, 20]),
            "rock_block": MultiDiscrete([45, 20]),
            "tree_block": MultiDiscrete([45, 20]),
            "water": MultiDiscrete([45, 20]),
            "refri": MultiDiscrete([45, 20]),
            "deposit": MultiDiscrete([45, 20]),
            "factory": MultiDiscrete([45, 20]),
            "riel": MultiDiscrete([45, 20]),
            "house": MultiDiscrete([45, 20]),
            "TASK": Discrete(4)
        })
        return obs

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(len(self.actions))

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        
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


        # if len(self.agents) == 4:
        #     string = "Current state: Agent1: {} , Agent2: {}, Agent3: {}, Agent4: {}".format(
        #         MOVES[self.state[self.agents[0]]], 
        #         MOVES[self.state[self.agents[1]]],
        #         MOVES[self.state[self.agents[2]]],
        #         MOVES[self.state[self.agents[3]]],
        #     )
        # else:
        #     string = "Game over"
        # print(string)
        
    def player_move(self, action, player):
        if action == 0:
            self.agents_pos[player][2] = 90
            if self.agents_pos[player][0] == 0:
                self.agents_pos[player][0] = max(0, self.agents_pos[player][0] - 1)
            elif self.grid[self.agents_pos[player][0]-1, self.agents_pos[player][1]] in self.valid_blocks and [self.agents_pos[player][0]-1, self.agents_pos[player][1]] not in [[pos[0], pos[1]] for pos in self.agents_pos]:
                self.agents_pos[player][0] = self.agents_pos[player][0] - 1
        if action == 1:
            self.agents_pos[player][2] = 270
            if self.agents_pos[player][0] == self.heigth-1:
                self.agents_pos[player][0] = min(self.heigth - 1, self.agents_pos[player][0] + 1)  
            elif self.grid[self.agents_pos[player][0]+1, self.agents_pos[player][1]] in self.valid_blocks and [self.agents_pos[player][0]+1, self.agents_pos[player][1]] not in [[pos[0], pos[1]] for pos in self.agents_pos]:
                self.agents_pos[player][0] = self.agents_pos[player][0] + 1
        if action == 2:
            self.agents_pos[player][2] = 180
            if self.agents_pos[player][1] == 0:
                self.agents_pos[player][1] = max(0, self.agents_pos[player][1] - 1)
            elif self.grid[self.agents_pos[player][0], self.agents_pos[player][1]-1] in self.valid_blocks and [self.agents_pos[player][0], self.agents_pos[player][1]-1] not in [[pos[0], pos[1]] for pos in self.agents_pos]:
                self.agents_pos[player][1] = self.agents_pos[player][1] - 1
        if action == 3:
            self.agents_pos[player][2] = 0
            if self.agents_pos[player][1] == self.width-1:
                self.agents_pos[player][1] = min(self.width - 1, self.agents_pos[player][1] + 1)
            elif self.grid[self.agents_pos[player][0], self.agents_pos[player][1]+1] in self.valid_blocks and [self.agents_pos[player][0], self.agents_pos[player][1]+1] not in [[pos[0], pos[1]] for pos in self.agents_pos]:
                self.agents_pos[player][1] = self.agents_pos[player][1] + 1
                

    def close(self):
        #pygame.quit()
        pass

    def reset(self, seed=None, options=None):
        
        self.rewards = {agent: 0 for agent in self.agents}

        self.observations = {
            self.agents[i]: {
            "vision": self.getAgentView(self.agents_pos[i][0], self.agents_pos[i][1]),
            "position": (self.agents_pos[i][0], self.agents_pos[i][1]),
            "orientation": self.agents_pos[i][2],
            "item": 0,
            "wood_item": (0, 0),
            "rock_item": (0, 0),
            "pickaxe": (0, 0),
            "axe": (0, 0),
            "bucket": (0, 0),
            "rock_block": (0, 0),
            "tree_block": (0, 0),
            "water": (0, 0),
            "refri": (0, 0),
            "deposit": (0, 0),
            "factory": (0, 0),
            "riel": (0, 0),
            "house": (0, 0),
            "TASK": 0,
        } for i in range(len(self.agents)) 
        }
        
        infos = {agent: {} for agent in self.agents}
        
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
            [9,9,9,9,9,9,9,9,9,9,9,1,1,1,1,1,1,1,1,1,1,1,5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [10,10,10,10,10,10,10,10,10,10,10,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [11,11,11,11,11,11,11,11,11,11,11,1,1,1,1,1,1,1,1,1,1,1,6,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [12,12,12,12,12,12,12,12,12,12,12,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [13,13,13,13,13,13,13,13,13,13,13,1,1,1,1,1,1,1,1,1,1,1,7,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [14,14,14,14,14,14,14,14,14,14,14,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [15,15,15,15,15,15,15,15,15,15,15,1,1,1,1,1,1,1,1,1,1,1,8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [16,16,16,16,16,16,16,16,16,16,16,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [17,17,17,17,17,17,17,17,17,17,17,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [18,18,18,18,18,18,18,18,18,18,18,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [14,14,14,14,14,14,14,14,14,14,14,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        ])
        self.agents_pos = [
            #[y, x, deg, item]
            [9, 16, 0],
            [10, 16, 0],
            [11, 16, 0],
            [12, 16, 0],
        ]

        return self.observations, infos
    

    def find_resource(self, grid, initial_position, target):
        # Obtener las posiciones donde el valor sea igual a n
        positions = np.argwhere(grid == target) 
        # Calcular la distancia a la posición objetivo
        distances = np.linalg.norm(positions - initial_position, axis=1)  
        if len(distances) > 0:
            # Encontrar el índice de la posición con la distancia mínima
            closer_distance = np.argmin(distances)  
            # Obtener la posición con la distancia mínima
            posicion_mas_cercana = positions[closer_distance]  
            return tuple(posicion_mas_cercana)
        else:
            return (-1, -1)
        
    def getAgentView(self, x, y):
        row_start = max(0, x - 1)
        row_end = min(self.grid.shape[0], x + 2)
        col_start = max(0, y - 1)
        col_end = min(self.grid.shape[1], y + 2)

        # Crear una matriz de -1 con las dimensiones de la ventana
        view = np.full((3, 3), -1)

        # Obtener las coordenadas válidas en la ventana
        view_row_start = max(0, 1 - x)
        view_row_end = view_row_start + (row_end - row_start)
        view_col_start = max(0, 1 - y)
        view_col_end = view_col_start + (col_end - col_start)

        # Actualizar la parte válida de la ventana con los valores de la matriz
        view[view_row_start:view_row_end, view_col_start:view_col_end] = self.grid[row_start:row_end, col_start:col_end]
        return view

    def step(self, actions):        
        # rewards for all agents are placed in the rewards dictionary to be returned
        
        for i, agent in enumerate(self.agents):
            reward = 0
            element_in_position = self.grid[self.agents_pos[i][0]][self.agents_pos[i][1]]
            up = self.grid[self.agents_pos[i][0]-1][self.agents_pos[i][1]]
            right = self.grid[self.agents_pos[i][0]][min(self.width - 1, self.agents_pos[i][1] + 1)]
            down = self.grid[min(self.heigth - 1, self.agents_pos[i][0] + 1)][self.agents_pos[i][1]]
            left = self.grid[self.agents_pos[i][0]][self.agents_pos[i][1]-1]
            near = [[right, 0, 1], [up, -1, 0], [left, 0, -1], [down, 1, 0]]
            initial_agent_observation = self.observations[agent].copy()
            perform_action = False    
            # Movimiento
            if actions[i] in self.actions[:4]:
                self.player_move(actions[i], i)
                
            # Agarrar/soltar
            if actions[i] == 4:
                # Evita que el agente coja el mismo item
                for j, orientation in enumerate(self.orientations):
                    
                    if self.agents_pos[i][2] == orientation:  
                        if near[j][0] == 17 and self.state["wood"] < 3 and initial_agent_observation["item"] == 9:
                            self.state["wood"] = self.state["wood"]+1
                            print(self.state)
                            self.observations[agent]["item"] = 0
                            reward = reward + self.rewards_values["complete_task"]
                            perform_action = True
                            continue
                        elif near[j][0] == 17 and self.state["iron"] < 3 and initial_agent_observation["item"] == 10:
                            self.state["iron"] = self.state["iron"]+1
                            print(self.state)
                            self.observations[agent]["item"] = 0
                            reward = reward + self.rewards_values["complete_task"]
                            perform_action = True
                            continue

                        elif near[j][0] == 16 and self.state["storage"] > 0 and initial_agent_observation["item"] != 8:
                            self.state["storage"] = self.state["storage"]-1
                            self.observations[agent]["item"] = 8
                            if initial_agent_observation["TASK"] == 2:
                                reward = reward + self.rewards_values["complete_task"]
                            else:
                                reward = reward + self.rewards_values["pick_wrong_item"]   
                            perform_action = True                         
                
                if perform_action:
                    continue
                
                
                if initial_agent_observation["item"] != 0 and element_in_position == 1:
                    self.grid[self.agents_pos[i][0]][self.agents_pos[i][1]] = initial_agent_observation["item"]
                    self.observations[agent]["item"] = 0
                
                if initial_agent_observation["item"] == element_in_position:
                    reward = reward + self.rewards_values["pick_same_item"]
                # Si esta en un lugar con un item
                if element_in_position in self.pickeable:
                    # Si el item es de la tarea, lo recompensa
                    
                    if element_in_position in self.valid_items[initial_agent_observation["TASK"]]:
                        reward = reward + self.rewards_values["pick_item"]
                        
                    # Si el item no es de la tarea, lo castiga
                    else:
                        reward = reward + self.rewards_values["pick_wrong_item"]
                        
                    # Se actualiza el mapa
                    self.grid[self.agents_pos[i][0]][self.agents_pos[i][1]] = 1 
                    self.observations[agent]["item"] = element_in_position

                # Si el rail se coloca en al lado del ultmo rail, se añade al path
                if initial_agent_observation["item"] != 0:
                    if initial_agent_observation["item"] == 8:
                        if (abs(self.agents_pos[i][0] - self.path[-1][0]) == 1 and self.agents_pos[i][1] == self.path[-1][1]) or (self.agents_pos[i][0] == self.path[-1][0] and abs(self.agents_pos[i][1] - self.path[-1][1]) == 1):
                            self.path.append((self.agents_pos[i][0],self.agents_pos[i][1]))
                            if self.path[-1][0] == self.end[0]-1 and self.path[-1][1] == self.end[1]:
                                done = True
                    else:
                        self.grid[self.agents_pos[i][0]][self.agents_pos[i][1]] = initial_agent_observation["item"]
                    
                    self.observations[agent]["item"] = element_in_position
                      
            # Romper/Recoger                        
            if actions[i] == 5 and initial_agent_observation["item"] in self.tools:
                for j, orientation in enumerate(self.orientations):
                    if self.agents_pos[i][2] == orientation:                        
                        # Evalua que el item que tenga sea valido para romper el bloque al que apunta
                        if near[j][0] in self.valid_tools_targets[initial_agent_observation["item"]] :#and near[j][0] in self.valid_action_targets[initial_agent_observation["TASK"]]:
                            
                            # Si es el cubo y esta frente a agua, se llena el cubo
                            if initial_agent_observation["item"] == 13:
                                self.observations[agent]["item"] = 14
                                reward = reward + self.rewards_values["fill_bucket"]
                                
                            # Si es el cubo lleno y esta frente a la locomotora, se vacia el cubo y se completa la tarea
                            elif initial_agent_observation["item"] == 14:
                                self.observations[agent]["item"] = 13
                                reward = reward + self.rewards_values["complete_task"]
                                self.state["temp"] = 0
                                
                            # Si es otro item, se remplaza el bloque por su drop
                            else:
                                self.grid[self.agents_pos[i][0]+near[j][1]][self.agents_pos[i][1]+near[j][2]] = self.drops[near[j][0]]
                                reward = reward + self.rewards_values["pick_item"]
                        
                        # Si el item es erroneo, se le castiga
                        else:
                            reward = reward + self.rewards_values["pick_wrong_item"]



            # update temp            
            if np.random.randint(0, 10000) == 1:
                self.state["temp"] = self.state["temp"] + 1
                
            if self.state["temp"] == 3:
                done = True
            
            # update crafting
            if self.state["wood"] > 0 and self.state["iron"] > 0 and self.state["storage"] < 3:
                self.state["wood"] = self.state["wood"] - 1
                self.state["iron"] = self.state["iron"] - 1
                self.state["storage"] = self.state["storage"] + 1
                
            
            # Set new window 
            view = self.getAgentView(self.agents_pos[i][0], self.agents_pos[i][1])
            self.observations[agent]["view"] = view
            
            self.rewards[agent] = self.rewards[agent] + reward

        train_position = (5, 5)
        
        new_states = {
            "wood_item": self.find_resource(self.grid, train_position, 9),
            "rock_item": self.find_resource(self.grid, train_position, 10),
            "pickaxe": self.find_resource(self.grid, train_position, 12),
            "axe": self.find_resource(self.grid, train_position, 11),
            "bucket": self.find_resource(self.grid, train_position, 13), # TODO: cambiar a cubo lleno y vacio
            "rock_block": self.find_resource(self.grid, train_position, 3),
            "tree_block": self.find_resource(self.grid, train_position, 2),
            "water": self.find_resource(self.grid, train_position, 5),
            "refri": self.find_resource(self.grid, train_position, 18),
            "deposit": self.find_resource(self.grid, train_position, 17),
            "factory": self.find_resource(self.grid, train_position, 16),
            "riel": self.find_resource(self.grid, train_position, 8), # TODO: cambiar a ultimo riel
            "house": self.find_resource(self.grid, train_position, 6), # TODO: cambiar a cord fija
        }

        


        # current observation is just the other player's most recent action
        for i, agent in enumerate(self.agents):
            self.observations[agent]["position"] = [self.agents_pos[i][0], self.agents_pos[i][1]],
            self.observations[agent]["wood_item"] = new_states["wood_item"]
            self.observations[agent]["rock_item"] = new_states["rock_item"]
            self.observations[agent]["pickaxe"] = new_states["pickaxe"]
            self.observations[agent]["axe"] = new_states["axe"]
            self.observations[agent]["bucket"] = new_states["bucket"]
            self.observations[agent]["rock_block"] = new_states["rock_block"]
            self.observations[agent]["tree_block"] = new_states["tree_block"]
            self.observations[agent]["water"] = new_states["water"]
            self.observations[agent]["refri"] = new_states["refri"]
            self.observations[agent]["deposit"] = new_states["deposit"]
            self.observations[agent]["factory"] = new_states["factory"]
            self.observations[agent]["riel"] = new_states["riel"]
            self.observations[agent]["house"] = new_states["house"]
            self.observations[agent]["TASK"] = 0 # COLOCAR TASK
        
        terminations = {agent: False for agent in self.agents}

        #self.num_moves += 1
        #env_truncation = self.num_moves >= NUM_ITERS
        truncations = {agent: False for agent in self.agents}

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        # if env_truncation:
        #     self.agents = []
        if self.render_mode == "human":
            self.render()
        return self.observations, self.rewards, terminations, truncations, infos


pygame.init()

# Crear el entorno Gym
env = UnrailedEnv(render_mode="human")

# Bucle principal
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                _, _, done, _, _ = env.step([0,0,0,0])
            elif event.key == pygame.K_DOWN:
                _, _, done, _, _ = env.step([1,0,0,0])
            elif event.key == pygame.K_LEFT:
                _, _, done, _, _ = env.step([2,0,0,0])
            elif event.key == pygame.K_RIGHT:
                _, _, done, _, _ = env.step([3,0,0,0])
            elif event.key == pygame.K_q:
                _, _, done, _, _ = env.step([4,0,0,0])
            elif event.key == pygame.K_e:
                _, _, done, _, _ = env.step([5,0,0,0])
            # if done:
            #     env.reset()
    
    env.render()
    
#pygame.quit()