"""
===================================================
    eLSI Sprint 1 - Task 1A : PID Line Following
===================================================

Participant template.

HOW TO RUN
  1. Open the Task 1A scene in CoppeliaSim.
  2. Start the bridge:   python3 bridge_task1a.py --eval
  3. Run this file:      python3 task1a_template.py

WHAT YOU IMPLEMENT
  Only control_loop(). Everything else (connecting, receiving sensors,
  sending motor commands) is handled for you by CoppeliaClient.
  Don't Edit this file except control_loop().
  You can add helper functions if you like.

Team ID: [ 957 ]
"""


import time
from connector_task1a import CoppeliaClient

SENSOR_ORDER = ['left_corner', 'left', 'middle', 'right', 'right_corner']

def control_loop(sensors):
    # Initialize persistent variables once
    if not hasattr(control_loop, "prev_error"):
        control_loop.prev_error = 0
        control_loop.invert = False
        control_loop.derivative_avg = 0 

    # 1. ROBUST INVERSION LOGIC
    outer_sum = sensors['left_corner'] + sensors['right_corner']
    if outer_sum > 1.5:
        control_loop.invert = True
    elif outer_sum < 0.5:
        control_loop.invert = False

    # 2. Calculate line position error & clean sensor data
    weights = {
        'left_corner': -2, 'left': -1, 'middle': 0, 'right': 1, 'right_corner': 2
    }

    numerator = 0
    denominator = 0
    line_values = []

    for sensor in SENSOR_ORDER:
        raw = sensors[sensor]
        # 'value' becomes 1.0 when it's directly over the line, regardless of the background color
        value = (1 - raw) if control_loop.invert else raw
        line_values.append(value)
        numerator += weights[sensor] * value
        denominator += value

    # Unpack processed sensor values (1.0 means line detected, 0.0 means background)
    lc, l, m, r, rc = line_values

    # 3. Line Recovery (If completely lost due to high momentum overshoot)
    if denominator < 0.15:
        # Boosted turn force to 4.0 to quickly snap back at 5.5 max speed
        turn_force = 4.0 
        if control_loop.prev_error < 0:
            return turn_force, -turn_force
        else:
            return -turn_force, turn_force

    position = numerator / denominator
    error = -position

    # 4. Smoothed PID Calculation
    Kp = 1.6  # Slightly increased to handle the faster 5.5 speed snap-back
    Ki = 0.0  
    Kd = 1.1  # Increased D-term to counteract the heavy momentum of 5.5 speed

    derivative = error - control_loop.prev_error
    control_loop.derivative_avg = (control_loop.derivative_avg * 0.6) + (derivative * 0.4)
    derivative = control_loop.derivative_avg

    pid = (Kp * error) + (Kd * derivative)
    control_loop.prev_error = error

    # 5. FIXED CORNER DETECTION SPEED REDUCTION LOGIC (TUNED FOR 5.5 SPEED)
    max_speed = 5.5
    min_speed = 1.0  # Lowered min_speed slightly to give a stronger braking effect from 5.5
    
    # Lowered threshold to 0.2 to catch the black line earlier before overshooting
    if lc > 0.2 or rc > 0.2:
        speed = min_speed
    else:
        # Full speed on straights, slight decay based on sudden derivative spikes
        speed = max_speed - 1.2 * abs(derivative)

    # Clamp the speed to safety bounds
    if speed > max_speed:
        speed = max_speed
    if speed < min_speed:
        speed = min_speed

    # 6. Apply PID and clamp final wheel velocities
    left_speed = speed - pid
    right_speed = speed + pid

    left_speed = max(-6, min(6, left_speed))
    right_speed = max(-6, min(6, right_speed))

    return left_speed, right_speed


def main():
    client = CoppeliaClient(host="127.0.0.1", port=50002)
    client.connect()
    print("Connected to bridge_task1a. Running... (Ctrl+C to stop)")

    last_sensors = None
    try:
        while True:
            sensors = client.receive_sensor_data()
            if sensors is not None:
                last_sensors = sensors
            if last_sensors is None:
                time.sleep(0.02)
                continue
            left, right = control_loop(last_sensors)
            client.send_motor_command(left, right)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        try:
            client.send_motor_command(0.0, 0.0)
        except Exception:
            pass
        client.close()

if __name__ == "__main__":
    main()