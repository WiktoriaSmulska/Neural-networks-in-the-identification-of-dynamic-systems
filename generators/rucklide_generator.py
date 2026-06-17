import numpy as np

OUTPUT_FILE = "rucklide.txt"

N = 10000
DT = 0.01

# Parametry układu Rucklidge'a
K = 2.0
LAMBDA = 6.7

# Warunki początkowe
state = np.array([1.0, 0.0, 4.5], dtype=float)


def rucklide(state):
    x, y, z = state

    dx = -K * x + LAMBDA * y - y * z
    dy = x
    dz = -z + y**2

    return np.array([dx, dy, dz], dtype=float)


def rk4_step(state, dt):
    k1 = rucklide(state)
    k2 = rucklide(state + dt * k1 / 2)
    k3 = rucklide(state + dt * k2 / 2)
    k4 = rucklide(state + dt * k3)

    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    for _ in range(N):
        file.write(f"{state[0]:.15g},{state[1]:.15g},{state[2]:.15g}\n")
        state = rk4_step(state, DT)

print(f"Zapisano plik: {OUTPUT_FILE}")