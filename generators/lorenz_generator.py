import numpy as np

# Parametry układu Lorenza
SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0

# Ustawienia generowania
DT = 0.01
N = 10000

# Warunek początkowy
x = 1.9656868951164
y = 0.0
z = 0.0

OUTPUT_FILE = "lorenz.txt"


def lorenz(state):
    x, y, z = state

    dx = SIGMA * (y - x)
    dy = x * (RHO - z) - y
    dz = x * y - BETA * z

    return np.array([dx, dy, dz])


def rk4_step(state, dt):
    k1 = lorenz(state)
    k2 = lorenz(state + dt * k1 / 2)
    k3 = lorenz(state + dt * k2 / 2)
    k4 = lorenz(state + dt * k3)

    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


state = np.array([x, y, z], dtype=float)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    for _ in range(N):
        file.write(f"{state[0]},{state[1]},{state[2]}\n")
        state = rk4_step(state, DT)

print(f"Zapisano plik: {OUTPUT_FILE}")