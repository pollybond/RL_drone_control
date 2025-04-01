import pygame
from drone_env import Drone2DEnv

def manual_control():
    # Инициализация pygame
    pygame.init()
    
    # Создание экрана для отображения
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Drone Manual Control")
    
    # Инициализация среды
    env = Drone2DEnv(render_mode='human')
    
    clock = pygame.time.Clock()
    env.reset()
    
    running = True
    while running:
        action = 0  # По умолчанию - нет действия
        
        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = 1
                elif event.key == pygame.K_RIGHT:
                    action = 2
                elif event.key == pygame.K_UP:
                    action = 3
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        # Выполнение шага в среде
        _, _, terminated, truncated, _ = env.step(action)
        
        # Рендеринг
        env.render()
        
        # Если эпизод завершен, сброс среды
        if terminated or truncated:
            env.reset()
        
        # Ограничение FPS
        clock.tick(60)
    
    # Корректное завершение
    env.close()
    pygame.quit()

if __name__ == "__main__":
    manual_control()