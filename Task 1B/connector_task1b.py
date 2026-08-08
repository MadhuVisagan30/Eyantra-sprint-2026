# connector_task1b.py - communication layer for Task 1B (Q-Learning).
#
# Talks to bridge_task1b.py over TCP (127.0.0.1:50002) using its JSON protocol.
# The bridge starts the simulation when it launches and stops it on completion
#  Ctrl+C
# Don't Edit this File.

import socket
import json


class CoppeliaClient:
    def __init__(self, host='127.0.0.1', port=50002):
        self.host = host
        self.port = port
        self.sock = None
        self.buffer = ""
        self._send_count = 0
        self._recv_count = 0
        self._freq_warned = False

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self.sock.settimeout(0.1)

    def send_motor_command(self, left_speed, right_speed, state=0, reward=0, action=0):
        """Send wheel speeds plus the RL triple (state, reward, action)."""
        cmd = {
            "command": "set_speed",
            "L": left_speed,
            "R": right_speed,
            "State": state,
            "Reward": reward,
            "Action": action,
        }
        self.sock.sendall((json.dumps(cmd) + "\n").encode())
        self._send_count += 1
        self._check_frequency()

    def start_simulation(self):
        cmd = {"command": "start_simulation"}
        self.sock.sendall((json.dumps(cmd) + "\n").encode())

    def stop_simulation(self):
        cmd = {"command": "stop_simulation"}
        self.sock.sendall((json.dumps(cmd) + "\n").encode())

    def receive_sensor_data(self):
        """Return the latest sensor dict, or None if no full packet yet.

        Sensors: {'left_corner','left','middle','right','right_corner'} in [0,1].
        """
        try:
            data = self.sock.recv(1024).decode()
            if not data:
                return None
            self.buffer += data
            if "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)
                sensor_msg = json.loads(line)
                if sensor_msg.get("type") == "sensor_update":
                    self._recv_count += 1
                    return sensor_msg["sensors"]
        except socket.timeout:
            pass
        except Exception as e:
            print(f"[CoppeliaClient] Error receiving sensor data: {e}")
        return None

    def _check_frequency(self):
        if self._freq_warned:
            return
        if self._send_count >= 40 and self._send_count > 2 * max(self._recv_count, 1):
            print("[CoppeliaClient] WARNING: your control loop is sending "
                  "commands FASTER than the bridge (~20 Hz) can read them "
                  f"(sent {self._send_count}, received {self._recv_count} sensor "
                  "packets). Add/lengthen a delay (e.g. time.sleep(0.05)) or send "
                  "one command per received sensor packet, otherwise the bridge "
                  "may fail to parse your commands.")
            self._freq_warned = True

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None
