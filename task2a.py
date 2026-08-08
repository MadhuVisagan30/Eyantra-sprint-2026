import time
from connector import CoppeliaClient

SENSOR_ORDER = ['left_corner', 'left', 'middle', 'right', 'right_corner']

# PID tuning parameters
KP = 1.4
KI = 0.005
KD = 0.7
BASE_SPEED = 2.5

# Global tracking states
integral = 0.0
prev_error = 0.0
last_position = 0.0

def reset_pid():
    """Resets the PID controller states to prevent violent overcorrections."""
    global integral, prev_error, last_position
    integral = 0.0
    prev_error = 0.0
    last_position = 0.0

def calculate_line_error(sensors):
    global last_position
    weights = [-2.0, -1.0, 0.0, 1.0, 2.0]
    weighted_sum = 0.0
    total = 0.0
    for i, name in enumerate(SENSOR_ORDER):
        val = sensors.get(name, 0.0)
        weighted_sum += val * weights[i]
        total += val
    if total < 0.1:  
        return last_position
    error = weighted_sum / total
    last_position = error
    return error

def line_follow(sensors, speed):
    global integral, prev_error
    error = calculate_line_error(sensors)
    integral += error
    derivative = error - prev_error
    correction = KP * error + KI * integral + KD * derivative
    prev_error = error
    
    integral = max(min(integral, 5.0), -5.0)
    left = speed - correction
    right = speed + correction
    return max(min(left, 5.0), -1.5), max(min(right, 5.0), -1.5)

def detect_color(sensors):
    r = sensors.get('color_r', 0.0)
    g = sensors.get('color_g', 0.0)
    b = sensors.get('color_b', 0.0)
    if max(r, g, b) < 0.3:
        return None
    if r >= g and r >= b: return "red"
    if g >= r and g >= b: return "green"
    return "blue"

def main():
    client = CoppeliaClient(host="127.0.0.1", port=50002)
    client.connect()
    print("Connected. Running Sensor-Based Navigation...")
    
    state = 'FIND_COLOR'
    detected_color = None
    left_pickup_zone = False 
    turn_frames = 0 
    delivery_frames = 0 
    
    try:
        while True:
            sensors = client.receive_sensor_data()
            if sensors is None:
                time.sleep(0.01)
                continue
                
            prox = sensors.get('proximity', 0.0)
            
            # ----------------------------------------------------
            # STATE 1: Find color tile while driving normally
            # ----------------------------------------------------
            if state == 'FIND_COLOR':
                detected_color = detect_color(sensors)
                if detected_color is not None:
                    print(f"[STATE] Color Locked: {detected_color}. Changing to APPROACH_BOX.")
                    state = 'APPROACH_BOX'
                
                left, right = line_follow(sensors, BASE_SPEED)
                client.send_motor_command(left, right)
            
            # ----------------------------------------------------
            # STATE 2: Slow down to safely register and pick the box
            # ----------------------------------------------------
            elif state == 'APPROACH_BOX':
                left, right = line_follow(sensors, speed=1.2)
                client.send_motor_command(left, right)
                
                if 0.01 < prox < 0.13:
                    print(f"[STATE] Box detected at {prox:.3f}m. Executing PICK...")
                    client.send_motor_command(0.0, 0.0)
                    
                    success = client.send_pick()
                    print(f"-> PICK attempted. Success: {success}")
                    
                    if success:
                        state = 'DRIVE_TO_JUNCTION'
                        left_pickup_zone = False 
                    else:
                        print("-> PICK failed. Creeping forward...")
                        client.send_motor_command(1.0, 1.0)
            
            # ----------------------------------------------------
            # STATE 3: Drive via PID towards the main junction crossbar
            # ----------------------------------------------------
            elif state == 'DRIVE_TO_JUNCTION':
                lc = sensors.get('left_corner', 0.0)
                rc = sensors.get('right_corner', 0.0)
                
                if not left_pickup_zone:
                    if lc < 0.3 and rc < 0.3:
                        left_pickup_zone = True
                        print("[SHIELD] Safely cleared pickup marker. Junction detection ACTIVE.")
                
                if left_pickup_zone and lc > 0.5 and rc > 0.5:
                    print(f"[JUNCTION] Crossbar detected! Initiating turn state for: {detected_color}")
                    state = 'JUNCTION_TURN'
                    turn_frames = 0
                    continue
                
                left, right = line_follow(sensors, BASE_SPEED)
                client.send_motor_command(left, right)
            
            # ----------------------------------------------------
            # STATE 3.5: Execute the turn dynamically
            # ----------------------------------------------------
            elif state == 'JUNCTION_TURN':
                turn_frames += 1
                mid_val = sensors.get('middle', 0.0)
                left_val = sensors.get('left', 0.0)
                right_val = sensors.get('right', 0.0)
                
                if detected_color == "red":
                    client.send_motor_command(-1.2, 1.2)
                elif detected_color == "green":
                    client.send_motor_command(1.2, -1.2)
                elif detected_color == "blue":
                    left, right = line_follow(sensors, BASE_SPEED)
                    client.send_motor_command(left, right)
                
                if turn_frames > 12:
                    if detected_color == "red" and (mid_val > 0.4 or right_val > 0.4):
                        print("[JUNCTION] Left branch line caught. Engaging PID.")
                        reset_pid()
                        state = 'DELIVERED'
                        delivery_frames = 0
                            
                    elif detected_color == "green" and (mid_val > 0.4 or left_val > 0.4):
                        print("[JUNCTION] Right branch line caught. Engaging PID.")
                        reset_pid()
                        state = 'DELIVERED'
                        delivery_frames = 0
                            
                    elif detected_color == "blue":
                        if turn_frames > 20: 
                            print("[JUNCTION] Blue straight path stabilized. Engaging PID.")
                            reset_pid()
                            state = 'DELIVERED'
                            delivery_frames = 0

            # ----------------------------------------------------
            # STATE 4: Follow branch line directly to destination drop cup
            # ----------------------------------------------------
            elif state == 'DELIVERED':
                delivery_frames += 1
                
                # Safe sensor read fallback to prevent errors if keys are occasionally absent
                left_val = sensors.get('left', 0.0)
                mid_val = sensors.get('middle', 0.0)
                right_val = sensors.get('right', 0.0)
                
                left_speed, right_speed = line_follow(sensors, BASE_SPEED)
                client.send_motor_command(left_speed, right_speed)
                
                # Scaled shield duration based on target direction to avoid false tripping
                required_shield = 25 if detected_color == "blue" else 15
                
                if delivery_frames > required_shield:
                    # Target strict line checking configuration
                    if left_val > 0.45 and mid_val > 0.45 and right_val > 0.45:
                        print(f"[STATE] Precise terminal block detected!")
                        
                        # Command dead-stop instantly
                        client.send_motor_command(0.0, 0.0)
                        
                        # Immediate drop to circumvent sliding physics delays
                        success = client.send_drop()
                        print(f"-> DROP executed. Success: {success}")
                        
                        time.sleep(1.5)  
                        break
                
            time.sleep(0.02)
            
    except KeyboardInterrupt:
        print("\nAborted by user.")
    except Exception as e:
        # Catch and print the exact traceback string instead of silently closing
        print(f"\n[ERROR ENCOUNTERED]: {e}")
    finally:
        # Safe structural cleanup
        try:
            client.send_motor_command(0.0, 0.0)
            client.close()
            print("Connection severed cleanly.")
        except Exception:
            print("Force terminated connection.")

if __name__ == "__main__":
    main()