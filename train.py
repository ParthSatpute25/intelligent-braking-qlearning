# train.py - Q-table Training Only

import numpy as np
import random

# Constants
ACTIONS = [0, -500, -1000, -1750, -2500]
PS = 100.0  # Stop sign position
DT = 0.1
num_states = 128
num_actions = len(ACTIONS)

# Q-learning setup
Q = np.zeros((num_states, num_actions))
alpha, gamma, epsilon = 0.1, 0.95, 0.1
episodes = 20000

# === Helper functions (same as in simulation) ===
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

def compute_reward(p, v, ps):
    d = p - ps
    if d < -1:
        return -100
    elif abs(d) <= 1 and v < 1:
        return 100
    elif abs(d) <= 1 and v >= 1:
        return -10
    elif d > 1:
        return -v / 10
    else:
        return -5

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

# === Training Loop ===
episode_returns = []
for episode in range(episodes):
    epsilon = max(0.01, epsilon * 0.9995)
    p = 0.0
    v = random.uniform(40, 100)
    ps = PS
    m = 100.0
    ep_reward = 0.0
    for t in range(300):
        d = p - ps
        if d > 200:
            f = 0
        else:
            s = get_state_index(p, v, ps, m)
            a_idx = random.randrange(num_actions) if random.random() < epsilon else np.argmax(Q[s])
            f = ACTIONS[a_idx]
        p, v = rk4_step(p, v, f, m, DT)
        r = compute_reward(p, v, ps)
        ep_reward += r
        s2 = get_state_index(p, v, ps, m)
        Q[s, a_idx] += alpha * (r + gamma * np.max(Q[s2]) - Q[s, a_idx])
        if abs(p - ps) < 1 and v < 1:
            break
    episode_returns.append(ep_reward)
    if (episode + 1) % 500 == 0:
        print(f"Episode {episode+1}: avg return = {np.mean(episode_returns[-500:]):.2f}")

# Save Q-table
np.save("Q_table.npy", Q)
print("✅ Q-table saved as Q_table.npy")
