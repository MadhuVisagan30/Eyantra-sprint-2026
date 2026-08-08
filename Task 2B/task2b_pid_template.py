"""
===================================================
  eLSI Sprint 1 - Task 2B : PID Line Following + Pick & Place (dual line)
===================================================

Participant template (PID variant).

TASK 2B
  Follow the track (white line on black AND black line on white) through the
  checkpoints, pick the red and blue boxes near the circle, drop each in its
  matching colour drop zone, then finish at the white box.
  Boxes are handled ONE AT A TIME: pick one, deliver it, come back for the other.

HOW TO RUN
  1. Open the Task 2B scene in CoppeliaSim.
  2. Start the bridge:   python3 bridge_v1_2b.py --eval
  3. Run this file:      python3 task2b_pid_template.py

WHAT YOU IMPLEMENT
  control_loop()  - PID controller that returns (left_speed, right_speed).
  detect_color()  - identify the box colour from the RGB sensor.
  should_pick()   - decide when to pick a box (only when one is right next to you).
  should_drop()   - decide when to drop the carried box (at its matching zone).

Everything else (connecting, receiving sensors, sending motor/pick/drop
commands) is handled by CoppeliaClient. Don't edit outside the marked TODO
sections. You may add helper functions.

SENSOR PROTOCOL (from bridge_v1_2b.py):
  Line sensors:  'left_corner','left','middle','right','right_corner' — [0,1].
                 NOTE: this track has BOTH white-line-on-black and
                 black-line-on-white sections, so "on the line" is not always
                 "high" — design your error term to handle both.
  Proximity:     'proximity' — metres to nearest object; 1.0 = nothing in range.
  Color sensor:  'color_r','color_g','color_b' — [0,1].

Team ID: [ 957 ]
"""
import time
from connector_2b import CoppeliaClient

# The five line sensors, ordered left -> right across the robot ([0.0, 1.0]).
SENSOR_ORDER = ['left_corner', 'left', 'middle', 'right', 'right_corner']

# EASY TUNING CONFIGURATION - ADAPTIVE SPEEDS FOR BASE 12.0 RUNS
TUNING_CONFIG = {
    'startup_ticks': 8,           # Instant startup turnaround
    
    'node_align_ticks': 8,        # Normal micro-inching duration to clear node core 
    'node_align_extra_ticks': 18, # Extended forward inching past node at 3rd junction of Red run
    'node_align_speed': 4.0,      # High-speed blast over junction nodes 
    
    # --- PIVOT SEARCH LIMITS ---
    'node_turn_ticks_first': 8,   # Balanced blind window to clear junction core before arming search
    'search_limit_ticks_first': 18,# Extended sweep window to reliably catch left branch
    'recovery_ticks_first': 14,   # Snap-back window toward center track
    
    'node_turn_ticks_second': 8,  # Balanced second priority blind window down to match
    'search_limit_ticks_second': 18,# Extended sweep window for second priority choice
    'recovery_ticks_second': 14,  # Snap-back window toward straight tracking
    
    # --- PIVOT SPEEDS ---
    'pivot_speed_fast': 4.8,       # Controlled torque delivery to swing through the turn arc
    'pivot_speed_stop': 0.0,       # Stopped wheel speed for absolute pivot
    
    # --- JUNCTION DETECTION COOLDOWNS ---
    'junction_threshold': 0.45,    # High-sensitivity rapid junction trigger
    'junction_cooldown': 18,       # Stable sensor turnaround recovery window
    
    # --- CONTINUOUS BIAS SETTINGS ---
    'continuous_bias_strength': 0.85, # Heavy continuous rail nudge to lock loops at high velocity
    
    # --- POST-TURN DRIVE DURATION ---
    'post_turn_straight_ticks': 28,# Hits drop container zones at high velocity 
    
    # --- TRANSITION SMOOTHING CONFIG ---
    'transition_blind_ticks': 8,   # Rapid boundary line crossing speed 
    
    # --- VELOCITY THROTTLE BUFFER ---
    'post_junction_speed_recovery_ticks': 35  # Higher stabilization safety cushion before triggering top gear
}

# Global state tracking dictionary
ROBOT_STATE = {
    'mode': 'startup_drive',   
    'startup_ticks': TUNING_CONFIG['startup_ticks'],
    'junction_cooldown': 0,    
    'align_ticks': 0,          
    'turn_ticks': 0,          
    'search_ticks': 0,        
    'recovery_ticks': 0,      
    'carrying_color': None,
    'active_routing_color': None, 
    'grid_switched': False,    
    'grid_switch_confirm_ticks': 0, 
    'drop_prevent_ticks': 0,   
    'pick_prevent_ticks': 0,    
    'second_half_junctions': 0, 
    'carrying_junctions_passed': 0, # Track junctions passed after picking a box
    'post_turn_drop_ticks': -1,  
    'bypass_junction_count': 0,  # Counter tracking remaining junctions to skip after drop
    'grid_lockout_ticks': 0,
    
    # --- POST-DROP ACCOUNTABILITY ENGINE ---
    'heading_home': False,
    'post_drop_junctions_passed': 0,
    
    # --- BLUE RETURN / FIRST HALF RED RUN LOGIC ---
    'bypass_first_junc_after_blue_return': False,
    'strict_left_only_phase': False,
    'red_run_junction_count': 0, # Counts 1st half junctions during strict left phase
    
    # --- DYNAMIC GLOBAL ROUTING SEQUENCE STATE ---
    'box_sequence_count': 0,      
    
    # --- DUAL-DIRECTION SMOOTH TRANSITION CONTROLS ---
    'transition_straight_ticks': 0,
    'northbound_transition_blind': 0,
    
    # --- DYNAMIC VELOCITY TRACKER TICKS ---
    'speed_recovery_counter': 0
}

PID_DATA = {
    'prev_error': 0.0,
    'integral': 0.0,
    'kp': 15.5,   
    'ki': 0.001,
    'kd': 9.2     
}

def get_normalized_readings(sensors):
    """Normalizes sensors using absolute grid state instead of guessing background."""
    raw_vals = [sensors.get(s, 0.0) for s in SENSOR_ORDER]

    if ROBOT_STATE['grid_switched']:
        return raw_vals
    else:
        return [(1.0 - v) for v in raw_vals]

def _reset_pid():
    PID_DATA['prev_error'] = 0.0
    PID_DATA['integral'] = 0.0

def control_loop(sensors):
    vals = get_normalized_readings(sensors)
    
    # =========================================================================
    # STRICT JUNCTION BYPASS CONDITIONS
    # =========================================================================
    is_blue_carry_3rd_junc = (ROBOT_STATE['carrying_color'] == 'blue' and 
                              ROBOT_STATE['carrying_junctions_passed'] == 3)
    
    is_red_carry_3rd_junc = (ROBOT_STATE['carrying_color'] == 'red' and 
                             ROBOT_STATE['carrying_junctions_passed'] == 3)
    
    is_3rd_junc_post_pickup = (is_blue_carry_3rd_junc or is_red_carry_3rd_junc)
    
    is_red_2nd_junc_2nd_half = (ROBOT_STATE['carrying_color'] == 'red' and 
                               ROBOT_STATE['grid_switched'] and 
                               ROBOT_STATE['second_half_junctions'] == 2)
    
    is_bypassing_junction = (ROBOT_STATE['bypass_junction_count'] > 0 or 
                             is_3rd_junc_post_pickup or 
                             is_red_2nd_junc_2nd_half or
                             ROBOT_STATE['bypass_first_junc_after_blue_return'])
    # =========================================================================

    if ROBOT_STATE['mode'] in ['node_align', 'search_priority_1', 'recover_to_junction', 'search_priority_2', 'recover_straight'] and not is_bypassing_junction:
        base_speed = 7.0  
        ROBOT_STATE['speed_recovery_counter'] = TUNING_CONFIG['post_junction_speed_recovery_ticks']
    else:
        if ROBOT_STATE['speed_recovery_counter'] > 0 and not is_bypassing_junction:
            ROBOT_STATE['speed_recovery_counter'] -= 1
            base_speed = 7.0  
        else:
            base_speed = 15.0  # Full power straight cruise
   
    if ROBOT_STATE['junction_cooldown'] > 0:
        ROBOT_STATE['junction_cooldown'] -= 1
        
    if ROBOT_STATE['drop_prevent_ticks'] > 0:
        ROBOT_STATE['drop_prevent_ticks'] -= 1
        
    if ROBOT_STATE['pick_prevent_ticks'] > 0:
        ROBOT_STATE['pick_prevent_ticks'] -= 1

    if ROBOT_STATE['grid_lockout_ticks'] > 0:
        ROBOT_STATE['grid_lockout_ticks'] -= 1

    # --- NORTHBOUND TRANSITION TIMERS ---
    if ROBOT_STATE['northbound_transition_blind'] > 0:
        ROBOT_STATE['northbound_transition_blind'] -= 1
        if ROBOT_STATE['northbound_transition_blind'] == 0:
            print("[NORTH TRANSITION] Blind phase completed. Entering active left sweep...")
            ROBOT_STATE['mode'] = 'recover_northbound_line'
        return 4.5, 4.5

    if ROBOT_STATE['mode'] == 'recover_northbound_line':
        if vals[2] > 0.55:
            print("[NORTH TRANSITION] Upper loop track locked! Resuming maximum speed PID tracking.")
            ROBOT_STATE['mode'] = 'forward'
            _reset_pid()
        else:
            return -2.4, 2.4  

    # --- SOUTHBOUND TRANSITION TIMERS ---
    if ROBOT_STATE['transition_straight_ticks'] > 0:
        ROBOT_STATE['transition_straight_ticks'] -= 1
        if ROBOT_STATE['transition_straight_ticks'] == 0:
            print("[TRANSITION LOCK] Blind phase completed. Entering active line sweep re-acquisition...")
            ROBOT_STATE['mode'] = 'recover_transition_line'
        return 4.5, 4.5

    if ROBOT_STATE['mode'] == 'recover_transition_line':
        if vals[2] > 0.55:
            print("[TRANSITION LOCK] Track successfully locked! Blasting speed boost (20.0).")
            ROBOT_STATE['mode'] = 'forward'
            _reset_pid()
            if ROBOT_STATE['strict_left_only_phase']:
                return 20.0, 20.0
        else:
            return 2.4, -2.4

    if ROBOT_STATE['mode'] == 'forward' and ROBOT_STATE['post_turn_drop_ticks'] > 0:
        ROBOT_STATE['post_turn_drop_ticks'] -= 1
        if ROBOT_STATE['post_turn_drop_ticks'] == 0:
            print("[STABILIZED RUN] Ready to drop box.")

    left_sensor = vals[1]
    middle_sensor = vals[2]
    right_sensor = vals[3]
    left_corner = vals[0]
    right_corner = vals[4]

    # --- STATE 0: INITIAL STARTUP FORWARD DRIVE ---
    if ROBOT_STATE['mode'] == 'startup_drive':
        if ROBOT_STATE['startup_ticks'] > 0:
            ROBOT_STATE['startup_ticks'] -= 1
            return 5.0, 5.0  
        else:
            print("[STARTUP COMPLETE] Tracking line forward.")
            ROBOT_STATE['mode'] = 'forward'

    # --- STATE 1: STANDARD FORWARD TRACKING ---
    elif ROBOT_STATE['mode'] == 'forward':
        raw_left_corner = sensors.get('left_corner', 0.0)
        raw_right_corner = sensors.get('right_corner', 0.0)
        
        if not ROBOT_STATE['grid_switched']:
            if ROBOT_STATE['active_routing_color'] is not None and ROBOT_STATE['grid_lockout_ticks'] == 0:
                if raw_left_corner < 0.25 and raw_right_corner < 0.25:
                    ROBOT_STATE['grid_switch_confirm_ticks'] += 1
                    if ROBOT_STATE['grid_switch_confirm_ticks'] >= 4:  
                        ROBOT_STATE['grid_switched'] = True
                        ROBOT_STATE['grid_switch_confirm_ticks'] = 0
                        ROBOT_STATE['grid_lockout_ticks'] = 70  
                        ROBOT_STATE['heading_home'] = False
                        ROBOT_STATE['post_drop_junctions_passed'] = 0
                        
                        ROBOT_STATE['northbound_transition_blind'] = TUNING_CONFIG['transition_blind_ticks']
                        _reset_pid()
                        print(f"[GRID SWITCH] Entered 2nd Half floor. Running blind push phase...")
                        return 5.0, 5.0
                else:
                    ROBOT_STATE['grid_switch_confirm_ticks'] = 0
        else:
            if (ROBOT_STATE['heading_home'] 
                    and ROBOT_STATE['post_drop_junctions_passed'] >= 3 
                    and raw_left_corner > 0.78 
                    and raw_right_corner > 0.78 
                    and ROBOT_STATE['grid_lockout_ticks'] == 0):
                
                ROBOT_STATE['grid_switch_confirm_ticks'] += 1
                if ROBOT_STATE['grid_switch_confirm_ticks'] >= 4:  
                    ROBOT_STATE['grid_switched'] = False
                    ROBOT_STATE['heading_home'] = False
                    ROBOT_STATE['active_routing_color'] = None  
                    ROBOT_STATE['junction_cooldown'] = 30  
                    ROBOT_STATE['second_half_junctions'] = 0 
                    ROBOT_STATE['grid_switch_confirm_ticks'] = 0
                    ROBOT_STATE['grid_lockout_ticks'] = 70
                    
                    ROBOT_STATE['bypass_first_junc_after_blue_return'] = True
                    ROBOT_STATE['strict_left_only_phase'] = True
                    ROBOT_STATE['red_run_junction_count'] = 0  # Reset Red run 1st half junction counter
                    print("[RETURN TO 1ST HALF] Arming 1st junction bypass & Strict Left-Only logic until Red box pickup.")
                    
                    ROBOT_STATE['transition_straight_ticks'] = TUNING_CONFIG['transition_blind_ticks']
                    _reset_pid()
                    print(f"[GRID SWITCH] Returned to 1st Half. Arming blind push zone...")
                    return 5.0, 5.0
            else:
                ROBOT_STATE['grid_switch_confirm_ticks'] = 0

        # High-sensitivity Junction Detection
        thresh = TUNING_CONFIG['junction_threshold']
        if (ROBOT_STATE['junction_cooldown'] == 0 and
            middle_sensor > thresh and (left_corner > thresh and right_corner > thresh)):
           
            print("[JUNCTION FOUND] Aligning over node...")
            ROBOT_STATE['mode'] = 'node_align'

            # Track junctions in 1st half specifically during strict_left_only_phase
            if ROBOT_STATE['strict_left_only_phase']:
                ROBOT_STATE['red_run_junction_count'] += 1
                print(f"[RED RUN] 1st Half Junction #{ROBOT_STATE['red_run_junction_count']} encountered.")
                
                # Check for 3rd junction on Red pickup run -> Extra forward inching duration
                if ROBOT_STATE['red_run_junction_count'] == 3:
                    ROBOT_STATE['align_ticks'] = TUNING_CONFIG['node_align_extra_ticks']
                    print(f"[RED RUN 3RD JUNCTION] Inching extra forward past 3rd node before strict left turn...")
                else:
                    ROBOT_STATE['align_ticks'] = TUNING_CONFIG['node_align_ticks']
            else:
                ROBOT_STATE['align_ticks'] = TUNING_CONFIG['node_align_ticks']

            if ROBOT_STATE['grid_switched']:
                ROBOT_STATE['second_half_junctions'] += 1

            if ROBOT_STATE['carrying_color'] is not None:
                ROBOT_STATE['carrying_junctions_passed'] += 1
                print(f"[CARRYING TRACKER] Passed junction #{ROBOT_STATE['carrying_junctions_passed']} while holding {ROBOT_STATE['carrying_color']}.")
                
            if ROBOT_STATE['heading_home']:
                ROBOT_STATE['post_drop_junctions_passed'] += 1
                print(f"[POST-DROP TRACKER] Encountered junction #{ROBOT_STATE['post_drop_junctions_passed']} after drop.")

    # --- STATE 2: CENTER DEEP OVER NODE ---
    elif ROBOT_STATE['mode'] == 'node_align':
        # 1. Post-blue return 1st junction bypass
        if ROBOT_STATE['bypass_first_junc_after_blue_return']:
            ROBOT_STATE['bypass_first_junc_after_blue_return'] = False
            print("[BYPASS] First junction after returning to 1st half bypassed! Cruising straight.")
            ROBOT_STATE['mode'] = 'forward'
            ROBOT_STATE['junction_cooldown'] = TUNING_CONFIG['junction_cooldown']
            _reset_pid()
            return base_speed, base_speed

        # 2. Post-drop bypass counter
        if ROBOT_STATE['bypass_junction_count'] > 0:
            ROBOT_STATE['bypass_junction_count'] -= 1 
            print(f"[BYPASS] Post-drop bypass active. Remaining skips: {ROBOT_STATE['bypass_junction_count']}. Cruising straight.")
            ROBOT_STATE['mode'] = 'forward'
            ROBOT_STATE['junction_cooldown'] = TUNING_CONFIG['junction_cooldown']
            _reset_pid()
            return base_speed, base_speed

        # 3. Third junction post-pickup for RED or BLUE
        if is_3rd_junc_post_pickup:
            if is_blue_carry_3rd_junc or is_red_carry_3rd_junc:
                print(f"[BYPASS BOOST] 3rd junction post-pickup for {ROBOT_STATE['carrying_color'].upper()} box! Blasting speed boost (20.0) straight.")
                ROBOT_STATE['mode'] = 'forward'
                ROBOT_STATE['junction_cooldown'] = TUNING_CONFIG['junction_cooldown']
                _reset_pid()
                return 20.0, 20.0

        # 4. Second junction in 2nd half when carrying RED
        if is_red_2nd_junc_2nd_half:
            print(f"[BYPASS] 2nd junction in 2nd half for RED box. Cruising straight.")
            ROBOT_STATE['mode'] = 'forward'
            ROBOT_STATE['junction_cooldown'] = TUNING_CONFIG['junction_cooldown']
            _reset_pid()
            return base_speed, base_speed

        if ROBOT_STATE['align_ticks'] > 0:
            ROBOT_STATE['align_ticks'] -= 1
            return TUNING_CONFIG['node_align_speed'], TUNING_CONFIG['node_align_speed']  
        else:
            if ROBOT_STATE['strict_left_only_phase']:
                use_left_priority = True
            elif ROBOT_STATE['grid_switched']:
                use_left_priority = (ROBOT_STATE['box_sequence_count'] < 2)
            else:
                use_left_priority = (ROBOT_STATE['box_sequence_count'] == 1 and ROBOT_STATE['carrying_color'] is None) or (ROBOT_STATE['box_sequence_count'] == 2)

            if use_left_priority:
                print(f"[ROUTING] Priority selection: Left -> Right (Strict Left Phase: {ROBOT_STATE['strict_left_only_phase']})")
                ROBOT_STATE['mode'] = 'search_priority_1'
                ROBOT_STATE['turn_ticks'] = TUNING_CONFIG['node_turn_ticks_first']
                ROBOT_STATE['search_ticks'] = TUNING_CONFIG['search_limit_ticks_first']
                return TUNING_CONFIG['pivot_speed_stop'], TUNING_CONFIG['pivot_speed_fast'] 
            else:
                print(f"[ROUTING] Priority selection: Right -> Left (Seq: {ROBOT_STATE['box_sequence_count']})")
                ROBOT_STATE['mode'] = 'search_priority_1'
                ROBOT_STATE['turn_ticks'] = TUNING_CONFIG['node_turn_ticks_first']
                ROBOT_STATE['search_ticks'] = TUNING_CONFIG['search_limit_ticks_first']
                return TUNING_CONFIG['pivot_speed_fast'], TUNING_CONFIG['pivot_speed_stop'] 

    # --- STATE 3: PIVOT SWEEP FOR FIRST PRIORITY CHOICE ---
    elif ROBOT_STATE['mode'] == 'search_priority_1':
        if ROBOT_STATE['strict_left_only_phase']:
            use_left = True
        elif ROBOT_STATE['grid_switched']:
            use_left = (ROBOT_STATE['box_sequence_count'] < 2)
        else:
            use_left = (ROBOT_STATE['box_sequence_count'] == 1 and ROBOT_STATE['carrying_color'] is None) or (ROBOT_STATE['box_sequence_count'] == 2)
            
        motor_p1 = (TUNING_CONFIG['pivot_speed_stop'], TUNING_CONFIG['pivot_speed_fast']) if use_left else (TUNING_CONFIG['pivot_speed_fast'], TUNING_CONFIG['pivot_speed_stop'])
        motor_rev1 = (TUNING_CONFIG['pivot_speed_fast'], TUNING_CONFIG['pivot_speed_stop']) if use_left else (TUNING_CONFIG['pivot_speed_stop'], TUNING_CONFIG['pivot_speed_fast'])

        if ROBOT_STATE['turn_ticks'] > 0:
            ROBOT_STATE['turn_ticks'] -= 1
            return motor_p1
        else:
            ROBOT_STATE['search_ticks'] -= 1
            if middle_sensor > 0.5 or right_sensor > 0.5 or left_sensor > 0.5:
                print("[TRACK LOCKED] Branch acquired. Following line.")
                ROBOT_STATE['mode'] = 'forward'
                ROBOT_STATE['junction_cooldown'] = TUNING_CONFIG['junction_cooldown']
                
                if ROBOT_STATE['carrying_color'] in ['blue', 'red'] and ROBOT_STATE['carrying_junctions_passed'] == 2:
                    print(f"[SPEED BOOST] Branch acquired after 2nd junction carrying {ROBOT_STATE['carrying_color'].upper()} box! Blasting speed boost (20.0).")
                    _reset_pid()
                    return 20.0, 20.0
                
                if ROBOT_STATE['second_half_junctions'] == 3 and (ROBOT_STATE['carrying_color'] in ['blue', 'red']):
                    ROBOT_STATE['post_turn_drop_ticks'] = TUNING_CONFIG['post_turn_straight_ticks']
                    print(f"[DROP ENGINE] Stabilizer armed for {ROBOT_STATE['carrying_color']} drop run.")
                
                if ROBOT_STATE['box_sequence_count'] == 2 and not ROBOT_STATE['grid_switched'] and not ROBOT_STATE['strict_left_only_phase']:
                    ROBOT_STATE['box_sequence_count'] = 3
                    print("[SEQUENCE CONTROL] Left priority junction cleared. Restoring default Right routing logic.")
                    
            elif ROBOT_STATE['search_ticks'] <= 0:
                ROBOT_STATE['mode'] = 'recover_to_junction'
                ROBOT_STATE['recovery_ticks'] = TUNING_CONFIG['recovery_ticks_first']
                return motor_rev1
            else:
                return motor_p1

    elif ROBOT_STATE['mode'] == 'bot_died':
        return 0.0, 0.0

    # --- STATE 4: RECOVER BACK TO JUNCTION CENTER ---
    elif ROBOT_STATE['mode'] == 'recover_to_junction':
        if ROBOT_STATE['strict_left_only_phase']:
            use_left = True
        elif ROBOT_STATE['grid_switched']:
            use_left = (ROBOT_STATE['box_sequence_count'] < 2)
        else:
            use_left = (ROBOT_STATE['box_sequence_count'] == 1 and ROBOT_STATE['carrying_color'] is None) or (ROBOT_STATE['box_sequence_count'] == 2)
            
        motor_rev1 = (TUNING_CONFIG['pivot_speed_fast'], TUNING_CONFIG['pivot_speed_stop']) if use_left else (TUNING_CONFIG['pivot_speed_stop'], TUNING_CONFIG['pivot_speed_fast'])
        motor_p2 = (TUNING_CONFIG['pivot_speed_fast'], TUNING_CONFIG['pivot_speed_stop']) if use_left else (TUNING_CONFIG['pivot_speed_stop'], TUNING_CONFIG['pivot_speed_fast'])

        if ROBOT_STATE['recovery_ticks'] > 0:
            ROBOT_STATE['recovery_ticks'] -= 1
            return motor_rev1
        else:
            if middle_sensor > 0.45 or left_sensor > 0.45 or right_sensor > 0.45:
                print("[RECOVERY CATCH] Intercepted track during recovery sweep! Normalizing...")
                ROBOT_STATE['mode'] = 'forward'
                ROBOT_STATE['junction_cooldown'] = TUNING_CONFIG['junction_cooldown']
                _reset_pid()
                return base_speed, base_speed
            
            ROBOT_STATE['mode'] = 'search_priority_2'
            ROBOT_STATE['turn_ticks'] = TUNING_CONFIG['node_turn_ticks_second']
            ROBOT_STATE['search_ticks'] = TUNING_CONFIG['search_limit_ticks_second']
            return motor_p2

    # --- STATE 5: PIVOT SWEEP FOR SECOND PRIORITY CHOICE ---
    elif ROBOT_STATE['mode'] == 'search_priority_2':
        if ROBOT_STATE['strict_left_only_phase']:
            use_left = True
        elif ROBOT_STATE['grid_switched']:
            use_left = (ROBOT_STATE['box_sequence_count'] < 2)
        else:
            use_left = (ROBOT_STATE['box_sequence_count'] == 1 and ROBOT_STATE['carrying_color'] is None) or (ROBOT_STATE['box_sequence_count'] == 2)
            
        motor_p2 = (TUNING_CONFIG['pivot_speed_fast'], TUNING_CONFIG['pivot_speed_stop']) if use_left else (TUNING_CONFIG['pivot_speed_stop'], TUNING_CONFIG['pivot_speed_fast'])
        motor_rev2 = (TUNING_CONFIG['pivot_speed_stop'], TUNING_CONFIG['pivot_speed_fast']) if use_left else (TUNING_CONFIG['pivot_speed_fast'], TUNING_CONFIG['pivot_speed_stop'])

        if ROBOT_STATE['turn_ticks'] > 0:
            ROBOT_STATE['turn_ticks'] -= 1
            return motor_p2
        else:
            ROBOT_STATE['search_ticks'] -= 1
            if middle_sensor > 0.5 or left_sensor > 0.5 or right_sensor > 0.5:
                print("[TRACK LOCKED] Priority 2 branch acquired.")
                ROBOT_STATE['mode'] = 'forward'
                ROBOT_STATE['junction_cooldown'] = TUNING_CONFIG['junction_cooldown']
                
                if ROBOT_STATE['carrying_color'] in ['blue', 'red'] and ROBOT_STATE['carrying_junctions_passed'] == 2:
                    print(f"[SPEED BOOST] Branch acquired after 2nd junction carrying {ROBOT_STATE['carrying_color'].upper()} box! Blasting speed boost (20.0).")
                    _reset_pid()
                    return 20.0, 20.0
                
                if ROBOT_STATE['second_half_junctions'] == 3 and (ROBOT_STATE['carrying_color'] in ['blue', 'red']):
                    ROBOT_STATE['post_turn_drop_ticks'] = TUNING_CONFIG['post_turn_straight_ticks']
                    
                if ROBOT_STATE['box_sequence_count'] == 2 and not ROBOT_STATE['grid_switched'] and not ROBOT_STATE['strict_left_only_phase']:
                    ROBOT_STATE['box_sequence_count'] = 3
                    print("[SEQUENCE CONTROL] Left priority junction cleared. Restoring default Right routing logic.")
                    
            elif ROBOT_STATE['search_ticks'] <= 0:
                ROBOT_STATE['mode'] = 'recover_straight'
                ROBOT_STATE['recovery_ticks'] = TUNING_CONFIG['recovery_ticks_second']
                return motor_rev2
            else:
                return motor_p2

    # --- STATE 6: RECOVER STRAIGHT ---
    elif ROBOT_STATE['mode'] == 'recover_straight':
        if ROBOT_STATE['strict_left_only_phase']:
            use_left = True
        elif ROBOT_STATE['grid_switched']:
            use_left = (ROBOT_STATE['box_sequence_count'] < 2)
        else:
            use_left = (ROBOT_STATE['box_sequence_count'] == 1 and ROBOT_STATE['carrying_color'] is None) or (ROBOT_STATE['box_sequence_count'] == 2)
            
        motor_rev2 = (TUNING_CONFIG['pivot_speed_stop'], TUNING_CONFIG['pivot_speed_fast']) if use_left else (TUNING_CONFIG['pivot_speed_fast'], TUNING_CONFIG['pivot_speed_stop'])
        motor_p1_alternate = (TUNING_CONFIG['pivot_speed_stop'], TUNING_CONFIG['pivot_speed_fast']) if not use_left else (TUNING_CONFIG['pivot_speed_fast'], TUNING_CONFIG['pivot_speed_stop'])

        if middle_sensor > 0.45 or left_sensor > 0.45 or right_sensor > 0.45:
            print("[INFINITE LOOP OVERRIDE] Track caught successfully during straight alignment sweep!")
            ROBOT_STATE['mode'] = 'forward'
            ROBOT_STATE['junction_cooldown'] = TUNING_CONFIG['junction_cooldown']
            if ROBOT_STATE['box_sequence_count'] == 2 and not ROBOT_STATE['grid_switched'] and not ROBOT_STATE['strict_left_only_phase']:
                ROBOT_STATE['box_sequence_count'] = 3
            _reset_pid()
            return base_speed, base_speed

        if ROBOT_STATE['recovery_ticks'] > 0:
            ROBOT_STATE['recovery_ticks'] -= 1
            return motor_rev2
        else:
            ROBOT_STATE['recovery_ticks'] = TUNING_CONFIG['recovery_ticks_second'] * 2
            print("[JUNCTION HUNT] Track missed. Executing safety counter-sweep...")
            return motor_p1_alternate

    # --- HIGH-SPEED PID LINE FOLLOWER PROCESSOR ---
    weights = [-2.5, -1.0, 0.0, 1.0, 2.5]
    total_weight = sum(vals)
    if total_weight > 0.1:
        error = sum(w * v for w, v in zip(weights, vals)) / total_weight
    else:
        error = PID_DATA['prev_error']
        
    PID_DATA['integral'] += error
    PID_DATA['integral'] = max(min(PID_DATA['integral'], 4.0), -4.0)
    derivative = error - PID_DATA['prev_error']
    PID_DATA['prev_error'] = error
   
    steering = (PID_DATA['kp'] * error) + (PID_DATA['ki'] * PID_DATA['integral']) + (PID_DATA['kd'] * derivative)
    
    # Force heavy left steering nudge during strict_left_only_phase line tracking
    if ROBOT_STATE['strict_left_only_phase']:
        steering -= TUNING_CONFIG['continuous_bias_strength']
    elif ROBOT_STATE['grid_switched']:
        if ROBOT_STATE['box_sequence_count'] < 2:
            steering -= TUNING_CONFIG['continuous_bias_strength']
        else:
            steering += TUNING_CONFIG['continuous_bias_strength']

    left_speed = base_speed + steering
    right_speed = base_speed - steering
    
    left_speed = max(min(left_speed, 14.0), -5.0)
    right_speed = max(min(right_speed, 14.0), -5.0)
    return float(left_speed), float(right_speed)

def detect_color(sensors):
    """Identify the colour of the box/zone from the RGB sensor with wider sensitivity."""
    r = sensors.get('color_r', 0.0)
    g = sensors.get('color_g', 0.0)
    b = sensors.get('color_b', 0.0)

    if max(r, g, b) < 0.12:
        return None
    if r >= g and r >= b:
        return "red"
    elif b >= r and b >= g:
        return "blue"
    elif g >= r and g >= b:
        return "green"
    return None

def should_pick(sensors, carrying_color):
    """Decide whether to send a PICK this cycle based on proximity."""
    if carrying_color is not None or ROBOT_STATE['pick_prevent_ticks'] > 0:
        return False
    proximity = sensors.get('proximity', 1.0)
    vals = get_normalized_readings(sensors)
    middle_sensor = vals[2]
    if proximity < 0.25 and middle_sensor > 0.35:
        return True
    return False

def should_drop(sensors, carrying_color):
    """Decide whether to execute a drop action based on completing the straight line timer."""
    if carrying_color is None or not ROBOT_STATE['grid_switched']:
        return False

    if (carrying_color in ["blue", "red"] 
            and ROBOT_STATE['second_half_junctions'] >= 3 
            and ROBOT_STATE['post_turn_drop_ticks'] == 0):
        if ROBOT_STATE['mode'] == 'forward':
            return True
            
    return False

def main():
    client = CoppeliaClient(host="127.0.0.1", port=50002)
    client.connect()
    print("Bridge connection established.")
    
    last_sensors   = None
    carrying_color = None  
    
    try:
        print("Starting main execution loop...")
        while True:
            sensors = client.receive_sensor_data()
            if sensors is not None:
                last_sensors = sensors
            if last_sensors is None:
                time.sleep(0.02)
                continue

            left, right = control_loop(last_sensors)

            # --- DYNAMIC PICK ENGINE ---
            if carrying_color is None and should_pick(last_sensors, carrying_color):
                print("[ACTION] Target detected. Executing picking alignment...")
                inch_timeout = 0
                while inch_timeout < 25:
                    fresh_sensors = client.receive_sensor_data()
                    if fresh_sensors is not None:
                        last_sensors = fresh_sensors
                    prox = last_sensors.get('proximity', 1.0)
                    if prox <= 0.12:
                        break
                    client.send_motor_command(1.5, 1.5)
                    time.sleep(0.02)
                    inch_timeout += 1
                
                client.send_motor_command(0.0, 0.0)
                time.sleep(0.15)
                
                for _ in range(3):
                    fresh_sensors = client.receive_sensor_data()
                    if fresh_sensors is not None:
                        last_sensors = fresh_sensors
                    time.sleep(0.03)

                colour_seen = detect_color(last_sensors)
                
                if colour_seen is None and ROBOT_STATE['strict_left_only_phase']:
                    colour_seen = "red"
                    print("[PICK ENGINE] Color reading low; applying Red fallback based on route position.")

                if colour_seen is not None:
                    success = client.send_pick()
                    print(f"[PICK] Color: {colour_seen!r} — Success={success}")
                    if success:
                        carrying_color = colour_seen
                        ROBOT_STATE['carrying_color'] = carrying_color
                        ROBOT_STATE['active_routing_color'] = carrying_color 
                        ROBOT_STATE['grid_switched'] = False 
                        ROBOT_STATE['second_half_junctions'] = 0 
                        ROBOT_STATE['carrying_junctions_passed'] = 0
                        ROBOT_STATE['post_turn_drop_ticks'] = -1
                        ROBOT_STATE['bypass_junction_count'] = 0
                        ROBOT_STATE['heading_home'] = False
                        ROBOT_STATE['post_drop_junctions_passed'] = 0
                        ROBOT_STATE['grid_lockout_ticks'] = 0
                        
                        # TURN OFF STRICT LEFT PHASE ON SUCCESSFUL RED PICKUP & RESTORE OLD LOGIC
                        ROBOT_STATE['strict_left_only_phase'] = False
                        ROBOT_STATE['red_run_junction_count'] = 0
                        print("[PICK SUCCESS] Red box acquired! Strict Left phase terminated. Standard old routing logic restored.")
                        
                        ROBOT_STATE['box_sequence_count'] += 1
                        print(f"[SEQUENCE COUNTER] Advanced stage. Current Box Count: {ROBOT_STATE['box_sequence_count']}")
                        
                        _reset_pid()
                        continue
                else:
                    ROBOT_STATE['pick_prevent_ticks'] = 15

            # --- LINE-VALIDATED DROP ENGINE ---
            if carrying_color is not None and should_drop(last_sensors, carrying_color):
                print(f"[ACTION] Straight line run stabilized. Dropping {carrying_color} box...")
                client.send_motor_command(0.0, 0.0)
                time.sleep(0.2)
                
                is_blue_drop = (carrying_color == "blue")
                
                success = client.send_drop()
                print(f"[DROP] Sequence completed — Success={success}")
                if success:
                    carrying_color = None
                    ROBOT_STATE['carrying_color'] = None
                    ROBOT_STATE['carrying_junctions_passed'] = 0
                    ROBOT_STATE['post_turn_drop_ticks'] = -1
                    ROBOT_STATE['pick_prevent_ticks'] = 60 
                    
                    if is_blue_drop:
                        ROBOT_STATE['bypass_junction_count'] = 3
                        ROBOT_STATE['heading_home'] = True
                        ROBOT_STATE['post_drop_junctions_passed'] = 0
                        print("[BYPASS ARMED] 3 junction bypasses assigned for Blue drop return.")
                    else:
                        ROBOT_STATE['bypass_junction_count'] = 0
                        print("[BYPASS INACTIVE] Red box drop completed. Maintaining track lock configuration.")
                    
                    post_drop_speed = 20.0 if is_blue_drop else 2.5
                    print(f"[RECOVERY] Inching forward past drop node with speed {post_drop_speed}...")
                    for _ in range(18):
                        client.send_motor_command(post_drop_speed, post_drop_speed)
                        time.sleep(0.02)
                    
                    ROBOT_STATE['junction_cooldown'] = 50  
                    _reset_pid()
                    continue

            client.send_motor_command(left, right)
            time.sleep(0.02)  
            
    except KeyboardInterrupt:
        print("\nTerminating process...")
    finally:
        try:
            client.send_motor_command(0.0, 0.0)
        except Exception:
            pass
        client.close()

if __name__ == "__main__":
    main()