import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider, Button

# ==============================
# DOUBLE PENDULUM SIMULATION
# ==============================

plt.style.use('dark_background')

# Parameters
g = 9.81
L1 = 1.0
L2 = 1.0
m1 = 1.0
m2 = 1.0

# Initial Conditions
theta1_0 = np.pi / 2
theta2_0 = np.pi / 2
theta1_dot_0 = 0.0
theta2_dot_0 = 0.0

# Time Parameters
dt = 0.01
t_max = 20
t = np.arange(0, t_max, dt)
n_steps = len(t)

# Arrays
theta1 = np.zeros(n_steps)
theta2 = np.zeros(n_steps)
theta1_dot = np.zeros(n_steps)
theta2_dot = np.zeros(n_steps)

# Initial Values
theta1[0] = theta1_0
theta2[0] = theta2_0
theta1_dot[0] = theta1_dot_0
theta2_dot[0] = theta2_dot_0

# ==============================
# DERIVATIVES FUNCTION
# ==============================

def derivatives(theta1, theta2, theta1_dot, theta2_dot):

    delta = theta2 - theta1

    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta)**2
    den2 = (L2 / L1) * den1

    dtheta1_dot = (
        m2 * L1 * theta1_dot**2 * np.sin(delta) * np.cos(delta)
        + m2 * g * np.sin(theta2) * np.cos(delta)
        + m2 * L2 * theta2_dot**2 * np.sin(delta)
        - (m1 + m2) * g * np.sin(theta1)
    ) / den1

    dtheta2_dot = (
        -m2 * L2 * theta2_dot**2 * np.sin(delta) * np.cos(delta)
        + (m1 + m2) * g * np.sin(theta1) * np.cos(delta)
        - (m1 + m2) * L1 * theta1_dot**2 * np.sin(delta)
        - (m1 + m2) * g * np.sin(theta2)
    ) / den2

    return dtheta1_dot, dtheta2_dot

# ==============================
# RK4 METHOD
# ==============================

for i in range(n_steps - 1):

    # k1
    k1_v1, k1_v2 = derivatives(
        theta1[i],
        theta2[i],
        theta1_dot[i],
        theta2_dot[i]
    )

    k1_t1 = theta1_dot[i]
    k1_t2 = theta2_dot[i]

    # k2
    k2_v1, k2_v2 = derivatives(
        theta1[i] + 0.5 * dt * k1_t1,
        theta2[i] + 0.5 * dt * k1_t2,
        theta1_dot[i] + 0.5 * dt * k1_v1,
        theta2_dot[i] + 0.5 * dt * k1_v2
    )

    k2_t1 = theta1_dot[i] + 0.5 * dt * k1_v1
    k2_t2 = theta2_dot[i] + 0.5 * dt * k1_v2

    # k3
    k3_v1, k3_v2 = derivatives(
        theta1[i] + 0.5 * dt * k2_t1,
        theta2[i] + 0.5 * dt * k2_t2,
        theta1_dot[i] + 0.5 * dt * k2_v1,
        theta2_dot[i] + 0.5 * dt * k2_v2
    )

    k3_t1 = theta1_dot[i] + 0.5 * dt * k2_v1
    k3_t2 = theta2_dot[i] + 0.5 * dt * k2_v2

    # k4
    k4_v1, k4_v2 = derivatives(
        theta1[i] + dt * k3_t1,
        theta2[i] + dt * k3_t2,
        theta1_dot[i] + dt * k3_v1,
        theta2_dot[i] + dt * k3_v2
    )

    k4_t1 = theta1_dot[i] + dt * k3_v1
    k4_t2 = theta2_dot[i] + dt * k3_v2

    # Update
    theta1[i + 1] = theta1[i] + (dt / 6) * (
        k1_t1 + 2*k2_t1 + 2*k3_t1 + k4_t1
    )

    theta2[i + 1] = theta2[i] + (dt / 6) * (
        k1_t2 + 2*k2_t2 + 2*k3_t2 + k4_t2
    )

    theta1_dot[i + 1] = theta1_dot[i] + (dt / 6) * (
        k1_v1 + 2*k2_v1 + 2*k3_v1 + k4_v1
    )

    theta2_dot[i + 1] = theta2_dot[i] + (dt / 6) * (
        k1_v2 + 2*k2_v2 + 2*k3_v2 + k4_v2
    )

# ==============================
# POSITIONS
# ==============================

x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)

x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

# ==============================
# ENERGY
# ==============================

KE1 = 0.5 * m1 * (L1 * theta1_dot)**2
KE2 = 0.5 * m2 * (
    (L1 * theta1_dot)**2 +
    (L2 * theta2_dot)**2
)

PE1 = -(m1 + m2) * g * L1 * np.cos(theta1)
PE2 = -m2 * g * L2 * np.cos(theta2)

Energy = KE1 + KE2 + PE1 + PE2

# ==============================
# FIGURE SETUP
# ==============================

fig = plt.figure(figsize=(12, 6))

# Animation Axis
ax = fig.add_subplot(121)

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)

ax.set_aspect('equal')

ax.set_title("Double Pendulum Simulation")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

# Energy Graph Axis
ax2 = fig.add_subplot(122)

ax2.plot(t, Energy, color='orange')

ax2.set_title("Energy vs Time")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Energy")

# Pendulum Line
line, = ax.plot(
    [],
    [],
    'o-',
    lw=3,
    color='cyan',
    markersize=10,
    markerfacecolor='yellow'
)

# Motion Trail
trail, = ax.plot([], [], '-', lw=1.5, color='magenta', alpha=0.7)

trail_x = []
trail_y = []

# Time Text
time_text = ax.text(
    -2.3,
    2.1,
    '',
    fontsize=12,
    color='white'
)

# ==============================
# ANIMATION FUNCTION
# ==============================

def animate(i):

    line.set_data(
        [0, x1[i], x2[i]],
        [0, y1[i], y2[i]]
    )

    trail_x.append(x2[i])
    trail_y.append(y2[i])

    trail.set_data(trail_x, trail_y)

    time_text.set_text(f"Time = {t[i]:.2f} s")

    return line, trail, time_text

# ==============================
# CREATE ANIMATION
# ==============================

ani = animation.FuncAnimation(
    fig,
    animate,
    frames=n_steps,
    interval=15,
    blit=True
)

# ==============================
# SAVE GIF (OPTIONAL)
# ==============================

# Uncomment to save animation
# ani.save("double_pendulum.gif", writer='pillow', fps=60)

# ==============================
# SHOW
# ==============================

plt.tight_layout()
plt.show()