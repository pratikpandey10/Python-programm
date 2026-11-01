import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Gravitational constant
G = 6.674e-11  

# Masses (kg)
M_sun = 1.989e30
M_earth = 5.972e24
M_moon = 7.348e22
masses = [M_sun, M_earth, M_moon]

AU = 1.496e11   
earth_moon = 3.84e8  

def positions_func(t):
    
    omega_earth = 2 * np.pi / (365 * 24 * 3600)  
    omega_moon = 2 * np.pi / (27.3 * 24 * 3600)  
    
    sun = np.array([0.0, 0.0])
    earth = np.array([AU * np.cos(omega_earth*t), AU * np.sin(omega_earth*t)])
    moon = earth + np.array([earth_moon * np.cos(omega_moon*t), earth_moon * np.sin(omega_moon*t)])
    
    return [sun, earth, moon]

def derivatives(t, y):
    x, y_pos, vx, vy = y
    pos_sat = np.array([x, y_pos])
    ax, ay = 0, 0
    
    for m, r in zip(masses, positions_func(t)):
        r_vec = r - pos_sat
        dist = np.linalg.norm(r_vec)
        ax += G * m * r_vec[0] / dist**3
        ay += G * m * r_vec[1] / dist**3

    return [vx, vy, ax, ay]
x0 = AU + earth_moon + 1e6  
y0 = 0.0
vx0 = 0.0
vy0 = 1000.0  
y_init = [x0, y0, vx0, vy0]
t_span = (0, 10*24*3600)
t_eval = np.linspace(t_span[0], t_span[1], 5000)
solution = solve_ivp(derivatives, t_span, y_init, t_eval=t_eval)
plt.figure(figsize=(7,7))
plt.plot(solution.y[0], solution.y[1], label="Satellite", lw=0.8)
sun, earth, moon = positions_func(0)
plt.scatter(sun[0], sun[1], color="yellow", s=200, label="Sun")
plt.scatter(earth[0], earth[1], color="blue", s=100, label="Earth")
plt.scatter(moon[0], moon[1], color="gray", s=50, label="Moon")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.legend()
plt.axis("equal")
plt.show()
