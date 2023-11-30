import time

import os

import numpy as np

import functools
import gymnasium
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Dict
from gymnasium.wrappers import FlattenObservation


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
        self.possible_agents = ["player_" + str(r) for r in range(4)]
        
                            #arriba, derecha, abajo, izquierda   
        self.orientations = [0, 1, 2, 3]
        self.actions = [0, 1, 2, 3, 4, 5]
        self.breakeble = [2, 3]
        self.pickeable = [9, 10, 11, 12, 13, 14]
        self.valid_blocks = [1, 7, 8, 9, 10, 11, 12, 13, 14]
        self.path = [(9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8)]
        self.end = (7, 42)
            
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
        self.grid[self.end[0]][self.end[1]] = 8
        
        self.state = {
            "temp": 0,
            "wood": 0,
            "iron": 0,
            "storage": 3,
            "front": len(self.path) - self.path.index(self.train["ENGINE"])
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
        self.obs_args = {
            "vision": [0, 25],
            "position": [25, 26],
            "orientation": 27,
            "item": 28,
            "wood_item": [29, 30],
            "rock_item": [31, 32],
            "pickaxe": [33, 34],
            "axe": [35, 36],
            "bucket": [37, 38],
            "full_bucket": [39, 40],
            "rock_block": [41, 42],
            "tree_block": [43, 44],
            "water": [45, 46],
            "refri": [47, 48],
            "deposit": [49, 50],
            "factory": [51, 52],
            "riel": [53, 54],
            "house": [55, 56],
            "task": 57,
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
        
        self.train_move = 0
                
        self.rewards = {agent: 0 for agent in self.agents}
        

        self.observations = {
            self.agents[i]:  list(np.zeros([1, 58])[0].astype(np.int8)) for i in range(len(self.agents)) 
        }
        for i, agent in enumerate(self.agents):
            self.observations[agent][self.obs_args["vision"][0]:self.obs_args["vision"][1]+1] = list(self.getAgentView(self.agents_pos[0][0], self.agents_pos[0][1]).reshape((1, 25))[0])
            self.observations[agent][self.obs_args["position"][0]:self.obs_args["position"][1]+1] = [self.agents_pos[i][0], self.agents_pos[i][1]]
            self.observations[agent][self.obs_args["orientation"]] = self.agents_pos[i][2]
            self.observations[agent][self.obs_args["item"]] = 0
            self.observations[agent][self.obs_args["wood_item"][0]:self.obs_args["wood_item"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"], 9)
            self.observations[agent][self.obs_args["rock_item"][0]:self.obs_args["rock_item"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"], 10)
            self.observations[agent][self.obs_args["pickaxe"][0]:self.obs_args["pickaxe"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"], 12)
            self.observations[agent][self.obs_args["axe"][0]:self.obs_args["axe"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"], 11)
            self.observations[agent][self.obs_args["bucket"][0]:self.obs_args["bucket"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"], 13)
            self.observations[agent][self.obs_args["full_bucket"][0]:self.obs_args["full_bucket"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"],14)
            self.observations[agent][self.obs_args["rock_block"][0]:self.obs_args["rock_block"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"], 3)
            self.observations[agent][self.obs_args["tree_block"][0]:self.obs_args["tree_block"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"], 2)
            self.observations[agent][self.obs_args["water"][0]:self.obs_args["water"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"], 5)
            self.observations[agent][self.obs_args["refri"][0]:self.obs_args["refri"][1]+1] = self.train["TANK"]
            self.observations[agent][self.obs_args["deposit"][0]:self.obs_args["deposit"][1]+1] = self.train["STORAGE"]
            self.observations[agent][self.obs_args["factory"][0]:self.obs_args["factory"][1]+1] = self.train["CRAFTING"]
            self.observations[agent][self.obs_args["riel"][0]:self.obs_args["riel"][1]+1] = self.path[-1]
            self.observations[agent][self.obs_args["house"][0]:self.obs_args["house"][1]+1] = self.end
            self.observations[agent][56] = 0
        
                
        if render_mode == "human":
            self.display = pygame.display.set_mode((self.cell_size*self.width, self.cell_size*self.heigth))
            pygame.display.set_caption("Unrailed")
        

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):

        obs = MultiDiscrete([18, 18, 18, 18, 18, 
                             18, 18, 18, 18, 18,
                             18, 18, 18, 18, 18,
                             18, 18, 18, 18, 18,
                             18, 18, 18, 18, 18, # vision
                             45, 20, # position
                             3,  # orientation
                             6,  # item
                             45, 20, # wood_item
                             45, 20, # rock_item
                             45, 20, # pickaxe
                             45, 20, # axe
                             45, 20, # bucket
                             45, 20, # full_bucket
                             45, 20, # rock_block
                             45, 20, # tree_block
                             45, 20, # water
                             45, 20, # refri
                             45, 20, # deposit
                             45, 20, # factory
                             45, 20, # riel
                             45, 20, # house
                             3 # task
                             ])
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
        self.possible_agents = ["player_" + str(r) for r in range(4)]
        self.agents = ["player_" + str(r) for r in range(4)]
        self.rewards = {agent: 0 for agent in self.agents}
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
        

        self.train = {
            "ENGINE": (9, 5),
            "TANK": (9, 4),
            "CRAFTING": (9, 3),
            "STORAGE": (9, 2),
        }
        self.path = [(9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8)]
        
        for rail in self.path:
            self.grid[rail[0]][rail[1]] = 8
            
        self.grid[self.train["ENGINE"][0]][self.train["ENGINE"][1]] = 15
        self.grid[self.train["TANK"][0]][self.train["TANK"][1]] = 18
        self.grid[self.train["CRAFTING"][0]][self.train["CRAFTING"][1]] = 17
        self.grid[self.train["STORAGE"][0]][self.train["STORAGE"][1]] = 16
        self.grid[self.end[0]][self.end[1]] = 8
        
        self.state = {
            "temp": 0,
            "wood": 0,
            "iron": 0,
            "storage": 3,
            "front": len(self.path) - self.path.index(self.train["ENGINE"])
        }

        for i, agent in enumerate(self.agents):
            self.observations[agent][self.obs_args["vision"][0]:self.obs_args["vision"][1]+1] = list(self.getAgentView(self.agents_pos[0][0], self.agents_pos[0][1]).reshape((1, 25))[0])
            self.observations[agent][self.obs_args["position"][0]:self.obs_args["position"][1]+1] = [self.agents_pos[i][0], self.agents_pos[i][1]]
            self.observations[agent][self.obs_args["orientation"]] = self.agents_pos[i][2]
            self.observations[agent][self.obs_args["item"]] = 0
            self.observations[agent][self.obs_args["wood_item"][0]:self.obs_args["wood_item"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"], 9)
            self.observations[agent][self.obs_args["rock_item"][0]:self.obs_args["rock_item"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"], 10)
            self.observations[agent][self.obs_args["pickaxe"][0]:self.obs_args["pickaxe"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"], 12)
            self.observations[agent][self.obs_args["axe"][0]:self.obs_args["axe"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"], 11)
            self.observations[agent][self.obs_args["bucket"][0]:self.obs_args["bucket"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"], 13)
            self.observations[agent][self.obs_args["full_bucket"][0]:self.obs_args["full_bucket"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"],14)
            self.observations[agent][self.obs_args["rock_block"][0]:self.obs_args["rock_block"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"], 3)
            self.observations[agent][self.obs_args["tree_block"][0]:self.obs_args["tree_block"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"], 2)
            self.observations[agent][self.obs_args["water"][0]:self.obs_args["water"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"], 5)
            self.observations[agent][self.obs_args["refri"][0]:self.obs_args["refri"][1]+1] = self.train["TANK"]
            self.observations[agent][self.obs_args["deposit"][0]:self.obs_args["deposit"][1]+1] = self.train["STORAGE"]
            self.observations[agent][self.obs_args["factory"][0]:self.obs_args["factory"][1]+1] = self.train["CRAFTING"]
            self.observations[agent][self.obs_args["riel"][0]:self.obs_args["riel"][1]+1] = self.path[-1]
            self.observations[agent][self.obs_args["house"][0]:self.obs_args["house"][1]+1] = self.end
            self.observations[agent][56] = 0
        
        infos = {agent: {} for agent in self.agents}
        
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
        row_start = max(0, x - 2)
        row_end = min(self.grid.shape[0], x + 3)
        col_start = max(0, y - 2)
        col_end = min(self.grid.shape[1], y + 3)

        # Crear una matriz de -1 con las dimensiones de la ventana
        view = np.full((5, 5), -1)

        # Obtener las coordenadas válidas en la ventana
        view_row_start = max(0, 2 - x)
        view_row_end = view_row_start + (row_end - row_start)
        view_col_start = max(0, 2 - y)
        view_col_end = view_col_start + (col_end - col_start)

        # Actualizar la parte válida de la ventana con los valores de la matriz
        view[view_row_start:view_row_end, view_col_start:view_col_end] = self.grid[row_start:row_end, col_start:col_end]
        return view.flatten()

    def step(self, actions):        
        # rewards for all agents are placed in the rewards dictionary to be returned
        done = False
        change_state = False
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
            if actions[agent] in self.actions[:4]:
                self.player_move(actions[agent], i)
                
            # Agarrar/soltar
            if actions[agent] == 4:
                # Evita que el agente coja el mismo item
                for j, orientation in enumerate(self.orientations):
                    
                    if self.agents_pos[i][2] == orientation:  
                        if near[j][0] == 17 and self.state["wood"] < 3 and initial_agent_observation[self.obs_args["item"]] == 9:
                            self.state["wood"] = self.state["wood"]+1
                            self.observations[agent][self.obs_args["item"]] = 0
                            reward = reward + self.rewards_values["complete_task"]
                            perform_action = True
                            change_state = True
                            continue
                        elif near[j][0] == 17 and self.state["iron"] < 3 and initial_agent_observation[self.obs_args["item"]] == 10:
                            self.state["iron"] = self.state["iron"]+1
                            self.observations[agent][self.obs_args["item"]] = 0
                            reward = reward + self.rewards_values["complete_task"]
                            perform_action = True
                            change_state = True
                            continue

                        elif near[j][0] == 16 and self.state["storage"] > 0 and initial_agent_observation[self.obs_args["item"]] != 8:
                            self.state["storage"] = self.state["storage"]-1
                            self.observations[agent][self.obs_args["item"]] = 8
                            if initial_agent_observation[56] == 2:
                                reward = reward + self.rewards_values["complete_task"]
                            else:
                                reward = reward + self.rewards_values["pick_wrong_item"]   
                            perform_action = True  
                            change_state = True                       
                
                if perform_action:
                    continue
                
                
                if initial_agent_observation[self.obs_args["item"]] != 0 and element_in_position == 1:
                    self.grid[self.agents_pos[i][0]][self.agents_pos[i][1]] = initial_agent_observation[self.obs_args["item"]]
                    self.observations[agent][self.obs_args["item"]] = 0
                
                if initial_agent_observation[self.obs_args["item"]] == element_in_position:
                    reward = reward + self.rewards_values["pick_same_item"]
                # Si esta en un lugar con un item
                if element_in_position in self.pickeable:
                    # Si el item es de la tarea, lo recompensa
                    
                    if element_in_position in self.valid_items[initial_agent_observation[56]]:
                        reward = reward + self.rewards_values["pick_item"]
                        
                    # Si el item no es de la tarea, lo castiga
                    else:
                        reward = reward + self.rewards_values["pick_wrong_item"]
                        
                    # Se actualiza el mapa
                    self.grid[self.agents_pos[i][0]][self.agents_pos[i][1]] = 1 
                    self.observations[agent][self.obs_args["item"]] = element_in_position

                # Si el rail se coloca en al lado del ultmo rail, se añade al path
                if initial_agent_observation[self.obs_args["item"]] == 8:
                    if (abs(self.agents_pos[i][0] - self.path[-1][0]) == 1 and self.agents_pos[i][1] == self.path[-1][1]) or (self.agents_pos[i][0] == self.path[-1][0] and abs(self.agents_pos[i][1] - self.path[-1][1]) == 1):
                        self.path.append((self.agents_pos[i][0],self.agents_pos[i][1]))
                        # Llega al final
                        if self.path[-1][0] == self.end[0]-1 and self.path[-1][1] == self.end[1]:
                            done = True
                      
            # Romper/Recoger                        
            if actions[agent] == 5 and initial_agent_observation[self.obs_args["item"]] in self.tools:
                for j, orientation in enumerate(self.orientations):
                    if self.agents_pos[i][2] == orientation:                        
                        # Evalua que el item que tenga sea valido para romper el bloque al que apunta
                        if near[j][0] in self.valid_tools_targets[initial_agent_observation[self.obs_args["item"]]] :#and near[j][0] in self.valid_action_targets[initial_agent_observation["task"]]:
                            
                            # Si es el cubo y esta frente a agua, se llena el cubo
                            if initial_agent_observation[self.obs_args["item"]] == 13:
                                self.observations[agent][self.obs_args["item"]] = 14
                                reward = reward + self.rewards_values["fill_bucket"]
                                
                            # Si es el cubo lleno y esta frente a la locomotora, se vacia el cubo y se completa la tarea
                            elif initial_agent_observation[self.obs_args["item"]] == 14:
                                self.observations[agent][self.obs_args["item"]] = 13
                                reward = reward + self.rewards_values["complete_task"]
                                self.state["temp"] = 0
                                change_state = True
                                
                            # Si es otro item, se remplaza el bloque por su drop
                            else:
                                self.grid[self.agents_pos[i][0]+near[j][1]][self.agents_pos[i][1]+near[j][2]] = self.drops[near[j][0]]
                                reward = reward + self.rewards_values["pick_item"]
                        
                        # Si el item es erroneo, se le castiga
                        else:
                            reward = reward + self.rewards_values["pick_wrong_item"]            
            
            # Set new window 
            view = self.getAgentView(self.agents_pos[i][0], self.agents_pos[i][1])
            self.observations[agent][self.obs_args["vision"][0]:self.obs_args["vision"][1]] = view
            print(self.rewards)
            self.rewards[agent] = self.rewards[agent] + reward
        
        # update temp            
        if np.random.randint(0, 10000) == 1:
            self.state["temp"] = self.state["temp"] + 1
            change_state = True
            
        if self.state["temp"] == 3:
            # Explota el tren
            done = True
        
        # update crafting
        if self.state["wood"] > 0 and self.state["iron"] > 0 and self.state["storage"] < 3:
            self.state["wood"] = self.state["wood"] - 1
            self.state["iron"] = self.state["iron"] - 1
            self.state["storage"] = self.state["storage"] + 1
            change_state = True
        
        for rail in self.path:
            self.grid[rail] = 8
            
            
        # train move
        self.train_move = self.train_move + 1/50
        if round(self.train_move, ndigits=1) == 1:
            self.train_move = 0
            try: 
                position = self.path.index(self.train["ENGINE"])
                self.train["STORAGE"] = self.train["CRAFTING"]
                self.train["CRAFTING"] = self.train["TANK"]
                self.train["TANK"] = self.train["ENGINE"]
                self.train["ENGINE"] = self.path[position+1]
            except: 
                # Descarrilamiento
                done = True
            
        self.grid[self.train["ENGINE"][0]][self.train["ENGINE"][1]] = 15
        self.grid[self.train["TANK"][0]][self.train["TANK"][1]] = 18
        self.grid[self.train["CRAFTING"][0]][self.train["CRAFTING"][1]] = 17
        self.grid[self.train["STORAGE"][0]][self.train["STORAGE"][1]] = 16
        
        
        new_states = {
            "wood_item": self.find_resource(self.grid, self.train["ENGINE"], 9),
            "rock_item": self.find_resource(self.grid, self.train["ENGINE"], 10),
            "pickaxe": self.find_resource(self.grid, self.train["ENGINE"], 12),
            "axe": self.find_resource(self.grid, self.train["ENGINE"], 11),
            "bucket": self.find_resource(self.grid, self.train["ENGINE"], 13),
            "full_bucket": self.find_resource(self.grid, self.train["ENGINE"],14),
            "rock_block": self.find_resource(self.grid, self.train["ENGINE"], 3),
            "tree_block": self.find_resource(self.grid, self.train["ENGINE"], 2),
            "water": self.find_resource(self.grid, self.train["ENGINE"], 5),
            "refri": self.train["TANK"],
            "deposit": self.train["STORAGE"],
            "factory": self.train["CRAFTING"],
            "riel": self.path[-1],
        }
        

        # current observation is just the other player's most recent action            
        for i, agent in enumerate(self.agents):
            self.observations[agent][self.obs_args["position"][0]:self.obs_args["position"][1]+1] = [self.agents_pos[i][0], self.agents_pos[i][1]]
            self.observations[agent][self.obs_args["orientation"]] = self.agents_pos[i][2]
            self.observations[agent][self.obs_args["wood_item"][0]:self.obs_args["wood_item"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"], 9)
            self.observations[agent][self.obs_args["rock_item"][0]:self.obs_args["rock_item"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"], 10)
            self.observations[agent][self.obs_args["pickaxe"][0]:self.obs_args["pickaxe"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"], 12)
            self.observations[agent][self.obs_args["axe"][0]:self.obs_args["axe"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"], 11)
            self.observations[agent][self.obs_args["bucket"][0]:self.obs_args["bucket"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"], 13)
            self.observations[agent][self.obs_args["full_bucket"][0]:self.obs_args["full_bucket"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"],14)
            self.observations[agent][self.obs_args["rock_block"][0]:self.obs_args["rock_block"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"], 3)
            self.observations[agent][self.obs_args["tree_block"][0]:self.obs_args["tree_block"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"], 2)
            self.observations[agent][self.obs_args["water"][0]:self.obs_args["water"][1]+1] = self.find_resource(self.grid, self.train["ENGINE"], 5)
            self.observations[agent][self.obs_args["refri"][0]:self.obs_args["refri"][1]+1] = self.train["TANK"]
            self.observations[agent][self.obs_args["deposit"][0]:self.obs_args["deposit"][1]+1] = self.train["STORAGE"]
            self.observations[agent][self.obs_args["factory"][0]:self.obs_args["factory"][1]+1] = self.train["CRAFTING"]
            self.observations[agent][self.obs_args["riel"][0]:self.obs_args["riel"][1]+1] = self.path[-1]
            self.observations[agent][self.obs_args["house"][0]:self.obs_args["house"][1]+1] = self.end
            self.observations[agent][56] = 0
        
            
                
        
        terminations = {agent: done for agent in self.agents}
        if done:
            self.agents = []
        truncations = {agent: False for agent in self.agents}

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if self.render_mode == "human":
            self.render()
        return self.observations, self.rewards, terminations, truncations, infos


# pygame.init()

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
                _, _, done, _, _ = env.step({'player_0': 0, 'player_1': 0, 'player_2': 0, 'player_3': 0})
            elif event.key == pygame.K_DOWN:
                _, _, done, _, _ = env.step({'player_0': 1, 'player_1': 1, 'player_2': 1, 'player_3': 1})
            elif event.key == pygame.K_LEFT:
                _, _, done, _, _ = env.step({'player_0': 2, 'player_1': 2, 'player_2': 2, 'player_3': 2})
            elif event.key == pygame.K_RIGHT:
                _, _, done, _, _ = env.step({'player_0': 3, 'player_1': 3, 'player_2': 3, 'player_3': 3})
            elif event.key == pygame.K_q:
                _, _, done, _, _ = env.step({'player_0': 4, 'player_1': 4, 'player_2': 4, 'player_3': 4})
            elif event.key == pygame.K_e:
                _, _, done, _, _ = env.step({'player_0': 5, 'player_1': 5, 'player_2': 5, 'player_3': 5})
            # if done:
            #     env.reset()
    
    env.render()
    
#pygame.quit()