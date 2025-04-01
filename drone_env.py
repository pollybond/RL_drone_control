import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pymunk

class Drone2DEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Физическое пространство
        self.space = pymunk.Space()
        self.space.gravity = (0, -900)
        
        # Параметры дрона
        self.drone_size = (40, 20)
        self.target_pos = (400, 50)
        
        # Пространства действий и наблюдений
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -1000, -1000, -np.pi, -10, 0]),
            high=np.array([800, 600, 1000, 1000, np.pi, 10, 1]),
            dtype=np.float32
        )
        
        # Инициализация дрона
        self._init_drone()

    def _init_drone(self):
        if hasattr(self, 'drone_body'):
            self.space.remove(self.drone_body, self.drone_shape)
            
        mass = 1
        moment = pymunk.moment_for_box(mass, self.drone_size)
        self.drone_body = pymunk.Body(mass, moment)
        self.drone_body.position = 400, 500
        self.drone_shape = pymunk.Poly.create_box(self.drone_body, self.drone_size)
        self.space.add(self.drone_body, self.drone_shape)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_drone()
        return self._get_state(), {}

    def step(self, action):
        # Применение действия
        if action == 1:  # Влево
            self.drone_body.apply_impulse_at_local_point((-100, 0), (0, 0))
        elif action == 2:  # Вправо
            self.drone_body.apply_impulse_at_local_point((100, 0), (0, 0))
        elif action == 3:  # Вверх
            self.drone_body.apply_impulse_at_local_point((0, 200), (0, 0))
        
        self.space.step(1/60)
        
        # Получение состояния
        obs = self._get_state()
        
        # Расчет награды
        reward, terminated = self._calculate_reward(obs)
        truncated = False
        info = {}
        
        if self.render_mode == "human":
            self._render_frame()
            
        return obs, reward, terminated, truncated, info

    def _get_state(self):
        return np.array([
            self.drone_body.position.x,
            self.drone_body.position.y,
            self.drone_body.velocity.x,
            self.drone_body.velocity.y,
            self.drone_body.angle,
            self.drone_body.angular_velocity,
            1 if self.drone_body.position.y <= 55 else 0
        ], dtype=np.float32)

    def _calculate_reward(self, state):
        x, y, vx, vy, angle, angular_vel, contact = state
        target_x, target_y = self.target_pos
        distance = np.sqrt((x-target_x)**2 + (y-target_y)**2) / 800
    
        reward = -distance - 0.1 * abs(angle)
    
        terminated = False
        # Условия завершения:
        if contact:  # Дрон коснулся земли
            terminated = True
            if distance < 0.1:  # Посадка в цели
                reward += 10
            else:  # Крушение
                reward -= 100
        elif x < 0 or x > 800 or y < 0 or y > 600:  # Выход за границы
            terminated = True
            reward -= 100
    
        return float(reward), bool(terminated)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((800, 600))
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((800, 600))
        canvas.fill((255, 255, 255))
        
        # Отрисовка дрона
        vertices = self.drone_shape.get_vertices()
        pygame.draw.polygon(
            canvas, (0, 0, 255),
            [(v.x + self.drone_body.position.x, v.y + self.drone_body.position.y) 
             for v in vertices]
        )
        
        # Отрисовка цели
        pygame.draw.circle(canvas, (0, 255, 0), self.target_pos, 15)
        
        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), 
                axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()