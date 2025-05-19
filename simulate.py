# simulate.py - Runs the braking simulation using pre-trained Q-table

import os
import numpy as np
import pygame
import matplotlib.pyplot as plt
import ctypes
import imageio.v2 as imageio

# === Load Trained Q-Table ===
Q = np.load("Q_table.npy")

# === Config ===
ACTIONS = [0, -500, -1000, -1750, -2500]
ACTION_COLORS = {0: (0,200,0), -500: (255,255,0), -1000: (255,165,0), -1750: (200,0,0), -2500: (139,0,0)}
SCALE = 8
WIDTH, HEIGHT = 1000, 300
STOP_SIGN_X = 800
PS = STOP_SIGN_X / SCALE
DT = 0.1
CAR_LENGTH = 60
CAR_HEIGHT = 30
frames = []
log = []

# === Helpers ===
def quantize_state(p, v, ps, m):
    d = p - ps
    if d < -1: d_idx = 0
    elif d < 2: d_idx = 1
    elif d < 5: d_idx = 2
    elif d < 10: d_idx = 3
    elif d < 30: d_idx = 4
    elif d < 70: d_idx = 5
    elif d < 130: d_idx = 6
    else: d_idx = 7
    if v < 2: v_idx = 0
    elif v < 10: v_idx = 1
    elif v < 40: v_idx = 2
    else: v_idx = 3
    ps_idx = 0 if ps < 100 else 1
    m_idx = 0 if m < 100 else 1
    return d_idx, v_idx, ps_idx, m_idx

def get_state_index(p, v, ps, m):
    d_idx, v_idx, ps_idx, m_idx = quantize_state(p, v, ps, m)
    return d_idx * 16 + v_idx * 4 + ps_idx * 2 + m_idx

def rk4_step(p, v, f, m, dt):
    def deriv(p_, v_):
        return v_, f/m
    k1_p, k1_v = deriv(p, v)
    k2_p, k2_v = deriv(p + 0.5*dt*k1_p, v + 0.5*dt*k1_v)
    k3_p, k3_v = deriv(p + 0.5*dt*k2_p, v + 0.5*dt*k2_v)
    k4_p, k4_v = deriv(p + dt*k3_p, v + dt*k3_v)
    p_new = p + (dt/6.0)*(k1_p + 2*k2_p + 2*k3_p + k4_p)
    v_new = v + (dt/6.0)*(k1_v + 2*k2_v + 2*k3_v + k4_v)
    return p_new, max(0.0, v_new)

# === Drawing ===
def draw_stop_sign(surface):
    x = STOP_SIGN_X
    pygame.draw.line(surface, (200,0,0), (x, HEIGHT//2), (x, HEIGHT//2-30), 4)
    sign_center = (x, HEIGHT//2-30)
    pygame.draw.circle(surface, (200,0,0), sign_center, 15)
    txt = FONT.render("STOP", True, (255,255,255))
    text_rect = txt.get_rect(center=sign_center)
    surface.blit(txt, text_rect)

def draw_car(surface, front_x, y, color):
    car_x = int(front_x - CAR_LENGTH)
    car_x = max(0, car_x)
    pygame.draw.rect(surface, color, (car_x, y, CAR_LENGTH, CAR_HEIGHT), border_radius=6)
    pygame.draw.rect(surface, (200,200,255), (car_x+10, y+5, 20, 15), border_radius=3)
    pygame.draw.rect(surface, (200,200,255), (car_x+35, y+5, 15, 15), border_radius=3)
    pygame.draw.circle(surface, (20,20,20), (car_x+15, y+CAR_HEIGHT), 7)
    pygame.draw.circle(surface, (20,20,20), (car_x+CAR_LENGTH-15, y+CAR_HEIGHT), 7)

# === Pygame Init ===
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
hwnd = pygame.display.get_wm_info().get('window')
if hwnd:
    ctypes.windll.user32.SetForegroundWindow(hwnd)
pygame.display.set_caption("Braking Simulation")
FONT = pygame.font.SysFont("arial", 16)
clock = pygame.time.Clock()

# === Sim State ===
p, v, m = 0.0, 70.0, 100.0
running, stopped, manual_forward_mode = True, False, False
f = 0.0
stopped_counter = 0

# === Main Loop ===
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill((240,240,240))
    pygame.draw.line(screen, (40,40,40), (0, HEIGHT//2), (WIDTH, HEIGHT//2), 4)
    draw_stop_sign(screen)

    if not stopped:
        d = p - PS
        if d > 200:
            f = 0
        else:
            s = get_state_index(p, v, PS, m)
            f = ACTIONS[np.argmax(Q[s])]
        p, v = rk4_step(p, v, f, m, DT)
        if p >= PS:
            p, v, f, stopped = PS, 0.0, 0.0, True
    elif not manual_forward_mode:
        stopped_counter += 1
        f = 0.0
        if stopped_counter >= int(2.0/DT):
            manual_forward_mode = True
    else:
        f = 500
        p, v = rk4_step(p, v, f, m, DT)

    log.append((len(log)*DT, p, v, f))
    front_x = p * SCALE
    draw_car(screen, front_x, HEIGHT//2-40, ACTION_COLORS.get(f, (150,150,150)))

    bar_color, intensity = ((0,150,200) if manual_forward_mode else (200,0,0)), (min(v/100.0, 1.0) if manual_forward_mode else abs(f)/2500.0)
    bar_width = int(intensity*100)
    pygame.draw.rect(screen, bar_color, (50, HEIGHT-60, bar_width, 10))
    pygame.draw.rect(screen, (0,0,0), (50, HEIGHT-60, 100, 10), 2)

    info = [f"Time: {len(log)*DT:.1f}s", f"Pos: {p:.1f} ft", f"Vel: {v:.1f} ft/s", f"Force: {f:.0f} lbf"]
    for i, l in enumerate(info):
        screen.blit(FONT.render(l, True, (20,20,20)), (70, 20 + 20*i))
    if manual_forward_mode:
        action_text = "Forward accel"
    elif stopped:
        action_text = "Waiting at sign"
    else:
        try:
            idx = ACTIONS.index(f)
            labels = ["No braking", "Light braking", "Medium braking", "Hard braking", "Very hard braking"]
            action_text = labels[idx]
        except ValueError:
            action_text = "Unknown"
    screen.blit(FONT.render(f"Action: {action_text}", True, (20,20,20)), (70, 20 + 20*len(info) + 5))

    pygame.display.flip()
    frame_surface = pygame.display.get_surface()
    frame_array = pygame.surfarray.array3d(frame_surface)
    frame_array = np.transpose(frame_array, (1, 0, 2))
    frames.append(frame_array)
    clock.tick(30)

# === Save & Quit ===
pygame.quit()
print("Saving animation to braking_simulation.mp4...")
imageio.mimsave("braking_simulation.mp4", frames, fps=30)
print("Saved.")

# === Plot results ===
times, positions, velocities, forces = zip(*log)
plt.figure(figsize=(10,8))
plt.subplot(3,1,1); plt.plot(times, positions); plt.axhline(PS, color='r', ls='--'); plt.ylabel("Position (ft)")
plt.subplot(3,1,2); plt.plot(times, velocities); plt.ylabel("Velocity (ft/s)")
plt.subplot(3,1,3); plt.plot(times, forces); plt.ylabel("Brake Force (lbf)"); plt.xlabel("Time (s)")
plt.tight_layout(); plt.show()
