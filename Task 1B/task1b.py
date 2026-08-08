"""
===================================================
    eLSI Sprint 1 - Task 1B : Q-Learning
===================================================

Participant template.

HOW TO RUN
  1. Open the Task 1B scene in CoppeliaSim.
  2. Start the bridge:   python3 bridge_task1b.py --eval
  3. Train:              python3 task1b_template.py --mode train
     Test (no learning): python3 task1b_template.py --mode test

MODES
  train : choose actions with exploration AND update the Q-table.
          The Q-table is saved to disk on exit.
  test  : load the saved Q-table, act greedily, and DO NOT update it.

WHAT YOU IMPLEMENT
  get_state()     - how to turn the 5 sensor values into a discrete state.
  get_reward()    - how good the latest reading is.
  choose_action() - which action to take in a given state (the policy).

Team ID: [ 957 ]
"""
import time
import os
import pickle
import random
import argparse
from connector_task1b import CoppeliaClient

SENSOR_ORDER = ['left_corner', 'left', 'middle', 'right', 'right_corner']

ACTIONS = [

    # Straight
    (9.2, 9.2),      # 0

    # Very Slight Left
    (8.9, 9.3),      # 1

    # Slight Left
    (8.4, 9.5),      # 2

    # Medium Left
    (7.3,10.2),      # 3

    # Sharp Left
    (5.8,10.9),      # 4

    # Extreme Left
    (4.2,11.4),      # 5

    # Very Slight Right
    (9.3,8.9),       # 6

    # Slight Right
    (9.5,8.4),       # 7

    # Medium Right
    (10.2,7.3),      # 8

    # Sharp Right
    (10.9,5.8),      # 9

    # Extreme Right
    (11.4,4.2)       # 10

]

ALPHA = 0.25
GAMMA = 0.94
EPSILON = 0.09
EPSILON_DECAY = 0.9994
MIN_EPSILON = 0.08

Q_TABLE_PATH = "q_table.pkl"

# Add this line here
LAST_SIDE = 0  # -1 for left, 1 for right, 0 for center
TRACK_MODE = 0
# 0 = White line
# 1 = Black line

def get_state(sensors):
    global LAST_SIDE, TRACK_MODE

    if isinstance(sensors, (list, tuple)):
        s = dict(zip(SENSOR_ORDER, sensors))
    else:
        s = sensors

    vals = [s[k] for k in SENSOR_ORDER]

    # ---------------------------------
    # Detect track colour using average
    # ---------------------------------

    avg = sum(vals) / 5.0

    # White section
    if TRACK_MODE == 0:

        if avg > 0.68:
            TRACK_MODE = 1

    # Black section
    else:

        if avg < 0.38:
            TRACK_MODE = 0

    invert = (TRACK_MODE == 1)

    # ---------------------------------
    # Invert if on black section
    # ---------------------------------

    if invert:
        processed = [1.0 - v for v in vals]
        thresh = 0.18
    else:
        processed = vals[:]
        thresh = 0.22

    binary = tuple(1 if v > thresh else 0 for v in processed)

    # ---------------------------------
    # Finish block
    # ---------------------------------

    if invert and binary == (1,1,1,1,1):
        print("🏁 Finish Block")
        return binary + (0,)

    # ---------------------------------
    # Position
    # ---------------------------------

    if binary[2]:
        pos = 0

    elif binary[0] or binary[1]:
        pos = -1
        LAST_SIDE = -1

    elif binary[3] or binary[4]:
        pos = 1
        LAST_SIDE = 1

    else:
        pos = 0

    return binary + (pos,)


def get_reward(sensors, state):
    lc, l, m, r, rc, pos = state
    total = lc + l + m + r + rc

    if m == 1:
        return 200.0 if total == 1 else 85.0
    if l == 1 or r == 1:
        return 55.0
    if lc == 1 or rc == 1:
        return -45.0
    if total == 0:
        return -65.0
    return -18.0


def choose_action(agent, state, training):
    global LAST_SIDE

    agent._ensure(state)
    q = agent.q_table[state]

    lc, l, m, r, rc, pos = state

    # =====================================================
    # STRAIGHT
    # =====================================================
    if (lc, l, m, r, rc) == (0,0,1,0,0):
        return 0

    # =====================================================
    # LEFT SIDE
    # =====================================================

    # Very Slight Left
    if (lc, l, m, r, rc) == (0,1,1,0,0):
        LAST_SIDE = -1
        return 1

    # Slight Left
    if (lc, l, m, r, rc) == (1,1,1,0,0):
        LAST_SIDE = -1
        return 2

    # Medium Left
    if (lc, l, m, r, rc) == (0,1,0,0,0):
        LAST_SIDE = -1
        return 3

    # Sharp Left
    if (lc, l, m, r, rc) == (1,1,0,0,0):
        LAST_SIDE = -1
        return 4

    # Extreme Left
    if (lc, l, m, r, rc) == (1,0,0,0,0):
        LAST_SIDE = -1
        return 5

    # =====================================================
    # RIGHT SIDE
    # =====================================================

    # Very Slight Right
    if (lc, l, m, r, rc) == (0,0,1,1,0):
        LAST_SIDE = 1
        return 6

    # Slight Right
    if (lc, l, m, r, rc) == (0,0,1,1,1):
        LAST_SIDE = 1
        return 7

    # Medium Right
    if (lc, l, m, r, rc) == (0,0,0,1,0):
        LAST_SIDE = 1
        return 8

    # Sharp Right
    if (lc, l, m, r, rc) == (0,0,0,1,1):
        LAST_SIDE = 1
        return 9

    # Extreme Right
    if (lc, l, m, r, rc) == (0,0,0,0,1):
        LAST_SIDE = 1
        return 10

    # =====================================================
    # LINE LOST
    # =====================================================

    if (lc + l + m + r + rc) == 0:

        if LAST_SIDE == -1:
            return 5

        if LAST_SIDE == 1:
            return 10

        return random.choice([5,10])

    # =====================================================
    # Q LEARNING FOR AMBIGUOUS STATES
    # =====================================================

    if training and random.random() < agent.epsilon:
        return random.randrange(len(ACTIONS))

    max_q = max(q)
    best = [i for i, v in enumerate(q) if v == max_q]

    return random.choice(best)


class QLearningAgent:
    def __init__(self, n_actions, alpha, gamma, epsilon, path):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.path = path
        self.q_table = {}

    def _ensure(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.n_actions

    def update(self, state, action, reward, next_state):
        self._ensure(state)
        self._ensure(next_state)
        best_next = max(self.q_table[next_state])
        td = reward + self.gamma * best_next
        self.q_table[state][action] += self.alpha * (td - self.q_table[state][action])

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                self.q_table = pickle.load(f)
            print(f"✅ Loaded {len(self.q_table)} states")
            return True
        return False

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"💾 Saved {len(self.q_table)} states")


def run(mode):
    training = (mode == "train")
    agent = QLearningAgent(len(ACTIONS), ALPHA, GAMMA, EPSILON, Q_TABLE_PATH)

    if not training:
        if not agent.load():
            print("No Q-table found!")
            return
    else:
        agent.load()

    client = CoppeliaClient(host="127.0.0.1", port=50002)
    client.connect()
    print(f"Connected | Mode: {mode}")

    prev_state = None
    prev_action = None
    step = 0

    try:
        while True:
            sensors = client.receive_sensor_data()
            if sensors is None:
                time.sleep(0.02)
                continue

            state = get_state(sensors)
            reward = get_reward(sensors, state)

            if training and prev_state is not None:
                agent.update(prev_state, prev_action, reward, state)
                agent.epsilon = max(MIN_EPSILON, agent.epsilon * EPSILON_DECAY)

            action = choose_action(agent, state, training)
            left, right = ACTIONS[action]

            print(f"Step {step:4d} | State: {state} | R: {reward:6.1f} | "
                  f"A: {action} ({left:.2f},{right:.2f}) | ε: {agent.epsilon:.3f}")

            client.send_motor_command(left, right, state=list(state), reward=reward, action=action)

            prev_state = state
            prev_action = action
            step += 1
            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        client.send_motor_command(0.0, 0.0)
        client.close()
        if training:
            agent.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
