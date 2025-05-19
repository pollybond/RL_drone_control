"""
test_drone.py

Тестирование среды с помощью случайных действий.
"""

import os
import pygame
import numpy as np
from drone_env import Drone2DEnv

class DroneTester:
    def __init__(self):
        self.screenshot_count = 0
        self.episode_count = 0
        self.ensure_directories()

        self.env = Drone2DEnv(render_mode='rgb_array')
        self.clock = pygame.time.Clock()

        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Drone Simulation")

    def ensure_directories(self):
        os.makedirs("screenshots", exist_ok=True)

    def save_screenshot(self, frame):
        filename = f"screenshots/ep_{self.episode_count:02d}_step_{self.screenshot_count:04d}.png"
        pygame.image.save(frame, filename)
        self.screenshot_count += 1

    def run_test(self, num_episodes=3, max_steps=200, screenshot_interval=10):
        try:
            for episode in range(num_episodes):
                self.episode_count = episode + 1
                self.screenshot_count = 0
                obs, _ = self.env.reset()
                total_reward = 0
                done = False
                step = 0

                print(f"\n=== Эпизод {self.episode_count} ===")
                while not done and step < max_steps:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt

                    action = self.env.action_space.sample()
                    obs, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    step += 1

                    frame = self.env.render()
                    if frame is not None:
                        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                        self.screen.blit(frame_surface, (0, 0))
                        pygame.display.flip()

                    if step % screenshot_interval == 0:
                        self.save_screenshot(frame_surface)
                        print(f"Скриншот на шаге {step}")

                    print(f"Шаг {step}: Позиция ({obs[0]:.1f}, {obs[1]:.1f}), Награда: {reward:.2f}")
                    self.clock.tick(60)

                print(f"Эпизод завершён. Всего шагов: {step}, Общая награда: {total_reward:.2f}\n")

        except KeyboardInterrupt:
            print("\nТестирование прервано пользователем.")
        finally:
            self.env.close()
            pygame.quit()


if __name__ == "__main__":
    tester = DroneTester()
    tester.run_test(num_episodes=3, max_steps=200, screenshot_interval=10)