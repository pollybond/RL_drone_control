import os
import pygame
import numpy as np
from drone_env import Drone2DEnv

class DroneTester:
    def __init__(self):
        self.screenshot_count = 0
        self.episode_count = 0
        self.ensure_directories()
        
        # Инициализация среды
        self.env = Drone2DEnv(render_mode='rgb_array')  # Изменено на rgb_array
        self.clock = pygame.time.Clock()
        
        # Инициализация pygame для отображения
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Drone Simulation")
        
    def ensure_directories(self):
        """Создаёт необходимые директории для сохранения результатов"""
        os.makedirs("screenshots", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
    
    def save_screenshot(self, frame):
        """Сохраняет текущий кадр в файл"""
        if frame is not None:
            filename = f"screenshots/ep_{self.episode_count:02d}_step_{self.screenshot_count:04d}.png"
            pygame.image.save(frame, filename)
            self.screenshot_count += 1
            return True
        return False
    
    def run_test(self, num_episodes=3, max_steps=200, screenshot_interval=10):
        """Запускает тестирование среды"""
        try:
            for episode in range(num_episodes):
                self.episode_count = episode + 1
                self.screenshot_count = 0
                obs, _ = self.env.reset()
                total_reward = 0
                done = False
                step = 0
                
                print(f"\n=== Начало эпизода {self.episode_count} ===")
                
                while not done and step < max_steps:
                    # Обработка событий pygame
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt
                    
                    # Выполнение шага
                    action = self.env.action_space.sample()
                    obs, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    step += 1
                    
                    # Получение кадра для отображения
                    frame = self.env.render()
                    
                    # Отображение на экране
                    if frame is not None:
                        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                        self.screen.blit(frame_surface, (0, 0))
                        pygame.display.flip()
                    
                    # Сохранение скриншотов
                    if step % screenshot_interval == 0 and frame is not None:
                        if self.save_screenshot(frame_surface):
                            print(f"Сохранён скриншот шага {step}")
                    
                    # Вывод информации
                    print(f"Шаг {step:03d} | Позиция: ({obs[0]:.1f}, {obs[1]:.1f}) | Награда: {reward:.2f}")
                    
                    # Контроль скорости
                    self.clock.tick(60)
                
                print(f"=== Эпизод {self.episode_count} завершён ===")
                print(f"Всего шагов: {step}")
                print(f"Общая награда: {total_reward:.2f}\n")
            
        except KeyboardInterrupt:
            print("\nТестирование прервано пользователем")
        finally:
            self.env.close()
            pygame.quit()

if __name__ == "__main__":
    tester = DroneTester()
    tester.run_test(
        num_episodes=3,
        max_steps=200,
        screenshot_interval=10
    )