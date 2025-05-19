import matplotlib.pyplot as plt
import pandas as pd

# Используется мониторинг от Stable Baselines3
train_data = pd.read_csv("logs/train_monitor.csv", skiprows=1)
plt.plot(train_data["r"])
plt.title("Кумулятивная награда за эпизод")
plt.xlabel("Эпизод")
plt.ylabel("Награда")
plt.grid(True)
plt.savefig("logs/reward_curve.png")
plt.show()