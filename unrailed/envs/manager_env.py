import numpy as np
import gym

class ResourcesEnv(gym.Env):
    def __init__(self):
        super(ResourcesEnv, self).__init__()
        self.wood = 0
        self.rock = 0
        self.rail = 0
        self.r_front = 3
        self.temperature = 0
        self.data = [self.wood, self.rock, self.rail, self.r_front, self.temperature]
        self.reward = 0
        self.time = 100
        
    def reset(self):
        self.wood = 0
        self.rock = 0
        self.rail = 0
        self.r_front = 3
        self.temperature = 0
        self.reward = 0
        self.time = 100

    def step(self, action):
        reward = 0
        done = False
        self.time = self.time - 1
        wood_action = 0
        rock_action = 1
        rail_action = 2
        cold_action = 3

        # fabricacion de rail
        if self.wood > 0 and self.rock > 0 and self.rail < 3:
            self.rail = self.rail + 1
            self.wood = self.wood - 1
            self.rock = self.rock - 1
            reward = reward + 3

        # recolectar madera
        if wood_action in action and self.wood < 3:
            self.wood = self.wood + 1
            reward = reward + 1

        # recolectar roca
        if rock_action in action and self.rock < 3:
            self.rock = self.rock + 1
            reward = reward + 1
        
        # Poner railes
        if rail_action in action and self.rail > 0:
            self.r_front = self.r_front + self.rail
            reward = reward + self.rail*2
            self.rail = 0 
        
        # Descarrilmiento
        if self.r_front == 0:
            reward = reward - 100
            done = True

        # Enfriar tren
        if self.temperature != 0 and cold_action in action:
            self.temperature = 0
            reward = reward + 5

        # Cambios de temperatura
        if np.random.randint(0, 20) == 1:
            self.temperature = self.temperature + 1
        
        # Explosion del tren
        if self.temperature == 3:
            reward = reward - 100
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
        self.data = [self.wood, self.rock, self.rail, self.r_front, self.temperature]
        return self.data, 0, done, {}
    
    def render(self, mode="human"):
        print("------------------------------")
        print("Wood: " + str(self.data[0]))
        print("Rock: " + str(self.data[1]))
        print("Rails: " + str(self.data[2]))
        print("Front: " + str(self.data[3]))
        print("Temp: " + str(self.data[4]))


# pygame.init()

# # Crear el entorno Gym
# env = ResourcesEnv()

# # Bucle principal
# running = True
# while running:
#     time.sleep(1)
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False
#         elif event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_UP:
#                 _, _, done, _ = env.step([0])
#             elif event.key == pygame.K_DOWN:
#                 _, _, done, _ = env.step([1])
#             elif event.key == pygame.K_LEFT:
#                 _, _, done, _ = env.step([2])
#             elif event.key == pygame.K_RIGHT:
#                 _, _, done, _ = env.step([3])
#             if done:
#                 env.reset()
    
#     env.render()
    
# pygame.quit()