import numpy as np

OUTPUT_FILE = "halvorsen.txt"

N = 10000
DT = 0.01

# Parametr Halvorsena
A = 1.4

# Warunki początkowe
state = np.array([1.0, 0.0, 0.0], dtype=float)


def halvorsen(state):
    x, y, z = state

    dx = -A * x - 4 * y - 4 * z - y**2
    dy = -A * y - 4 * z - 4 * x - z**2
    dz = -A * z - 4 * x - 4 * y - x**2

    return np.array([dx, dy, dz], dtype=float)


def rk4_step(state, dt):
    k1 = halvorsen(state)
    k2 = halvorsen(state + dt * k1 / 2)
    k3 = halvorsen(state + dt * k2 / 2)
    k4 = halvorsen(state + dt * k3)

    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    for _ in range(N):
        file.write(f"{state[0]:.15g},{state[1]:.15g},{state[2]:.15g}\n")
        state = rk4_step(state, DT)

print(f"Zapisano plik: {OUTPUT_FILE}")