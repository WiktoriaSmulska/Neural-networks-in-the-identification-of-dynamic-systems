import numpy as np

OUTPUT_FILE = "rossler.txt"

N = 10000
DT = 0.01

# Parametry Rösslera
A = 0.2
B = 0.2
C = 5.7

# Warunki początkowe
state = np.array([1.0, 1.0, 1.0], dtype=float)


def rossler(state):
    x, y, z = state

    dx = -y - z
    dy = x + A * y
    dz = B + z * (x - C)

    return np.array([dx, dy, dz], dtype=float)


def rk4_step(state, dt):
    k1 = rossler(state)
    k2 = rossler(state + dt * k1 / 2)
    k3 = rossler(state + dt * k2 / 2)
    k4 = rossler(state + dt * k3)

    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    for _ in range(N):
        file.write(f"{state[0]:.15g},{state[1]:.15g},{state[2]:.15g}\n")
        state = rk4_step(state, DT)

print(f"Zapisano plik: {OUTPUT_FILE}")