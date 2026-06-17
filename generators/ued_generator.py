import numpy as np

# Generator układu Uedy:
# x' = y
# y' = -k*y - x^3 + A*cos(omega*t)

K = 0.05
A = 7.5
OMEGA = 1.0

DT = 0.02
N = 10000

# Warunki początkowe podobne do Twojego przykładu
x = 1.12123516760042
y = 0.0
t = 0.0

OUTPUT_FILE = "ueda.txt"


def ueda(state, t):
    x, y = state

    dx = y
    dy = -K * y - x**3 + A * np.cos(OMEGA * t)

    return np.array([dx, dy], dtype=float)


def rk4_step(state, t, dt):
    k1 = ueda(state, t)
    k2 = ueda(state + dt * k1 / 2, t + dt / 2)
    k3 = ueda(state + dt * k2 / 2, t + dt / 2)
    k4 = ueda(state + dt * k3, t + dt)

    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


state = np.array([x, y], dtype=float)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    for _ in range(N):
        file.write(f"{state[0]:.15g},{state[1]:.15g}\n")

        state = rk4_step(state, t, DT)
        t += DT

print(f"Zapisano plik: {OUTPUT_FILE}")