"""
manual_control.py

Ручное управление дроном с помощью клавиатуры.
Позволяет протестировать среду в режиме реального времени.

Управление:
- ← (влево): действие 1
- → (вправо): действие 2
- ↑ (вверх): действие 3
- ESC: выход из приложения
"""

import pygame
from drone_env import Drone2DEnv


def manual_control():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Drone Manual Control")

    env = Drone2DEnv(render_mode='human')
    env.reset()

    clock = pygame.time.Clock()
    running = True

    print("Управление: ← — влево, → — вправо, ↑ — вверх, ESC — выход")

    try:
        while running:
            action = 0

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

            _, reward, terminated, truncated, _ = env.step(action)
            env.render()

            if terminated or truncated:
                print(f"Эпизод завершён. Последняя награда: {reward}")
                env.reset()

            clock.tick(60)

    except KeyboardInterrupt:
        print("\nРучное управление прервано.")
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    manual_control()