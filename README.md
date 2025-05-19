# 🚦 Intelligent Braking Control Using Q-Learning

This project implements a tabular Q-learning controller for an intelligent braking system in a one-dimensional (1D) vehicle simulation. The objective is to stop the vehicle precisely at a stop sign, using optimal discrete brake force actions while minimizing overshoot and stopping time. The simulation integrates vehicle dynamics using a 4th-order Runge-Kutta (RK4) solver and visualizes performance in real time using Pygame.

---

## 🚗 Features

* ✔️ RK4-based motion integration for smooth dynamics
* 🧠 Q-learning agent with 128 discrete states and 5 force actions
* 🔍 Reward shaping for precision braking behavior
* 📉 Penalization for overshoot and insufficient deceleration
* 🎥 Real-time Pygame animation and MP4 export
* 🎯 Post-stop forward acceleration (500 lbf after 2s delay)
* 📊 Velocity, position, and brake force plots

---

## 📁 Project Structure

```
.
├── train.py                  # Q-learning training script
├── simulate.py               # Braking simulation and animation
├── Q_table.npy               # Saved Q-table (auto-generated)
├── braking_simulation.mp4    # Full animation output (auto-generated)
└── README.md                 # Project overview and instructions
```

---

## ⚙️ How to Run

1. Install dependencies:

```bash
pip install pygame numpy matplotlib imageio imageio-ffmpeg
```

2. Train the agent:

```bash
python train.py
```

3. Run the simulation:

```bash
python simulate.py
```

After simulation ends, you’ll get:

* 📼 `braking_simulation.mp4`
* 🖼️ Time-series plots of position, velocity, and force

---

## 🎯 Reward Design

| Event                         | Reward  |
| ----------------------------- | ------- |
| Stopped near sign & low speed | +100    |
| Overshoot                     | -100    |
| High-speed stop near sign     | -10     |
| Passing beyond sign           | -v / 10 |
| Step cost                     | -5      |

---

## 🧠 Q-Learning Details

* **States**: 8 (distance) × 4 (velocity) × 2 (goal) × 2 (mass) = 128
* **Actions**: 0, -500, -1000, -1750, -2500 lbf
* **Learning Rate**: $\alpha = 0.1$
* **Discount Factor**: $\gamma = 0.95$
* **Exploration**: Epsilon-greedy decay to 0.01
* **Episodes**: 20,000

---

## 📊 Performance Highlights

* Learns to stop accurately within 1 ft of stop sign
* Braking actions intensify as position approaches target
* Smooth transition to forward motion after 2s stop
* Full dynamics plotted and recorded

---

## 📄 License

This project is released under the MIT License.

---

## 🙌 Acknowledgments

Developed as part of **EE 5322 – Intelligent Control** at the University of Texas at Arlington.
Special thanks to faculty and peers for discussions on reinforcement learning and control theory.
