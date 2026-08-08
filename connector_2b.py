# connector_2b.py — communication layer for bridge_v1_2b.py (Task 2B).
#
# Talks to bridge_v1_2b.py over TCP (127.0.0.1:50002) using its newline-delimited
# JSON protocol. The bridge starts the simulation when it launches and stops it
# on completion / Ctrl+C.
# One connector serves BOTH templates:
#   • PID template — calls send_motor_command(L, R).
#   • RL  template — calls send_motor_command(L, R, state, reward, action).
# Don't Edit this File.

import socket
import json
import time


class CoppeliaClient:
    """
    JSON-protocol client for bridge_v1_2b.py.

    Bridge → Client (one per sim step):
        {"type":"sensor_update","sensors":{
            "left_corner","left","middle","right","right_corner"  — [0,1]
            "proximity"                       — metres (1.0 = nothing in range)
            "color_r","color_g","color_b"     — [0,1]
        }}

    Client → Bridge:
        {"command":"set_speed","L":..,"R":..,"State":..,"Reward":..,"Action":..}
        {"command":"pick"}      — pick the nearest in-range box
        {"command":"drop"}      — drop the held box
        {"command":"stop"}

    Bridge → Client (replies):
        {"type":"pick_result","success":true/false,"color":"red"/"blue"/null}
        {"type":"drop_result","success":true/false,"color":"red"/"blue"/null}
    """

    def __init__(self, host="127.0.0.1", port=50002):
        self.host = host
        self.port = port
        self.sock = None
        self.buffer = ""
        self._send_count = 0
        self._recv_count = 0
        self._freq_warned = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def connect(self):
        """Open TCP connection to the bridge."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self.sock.settimeout(0.1)

    def close(self):
        """Close the connection."""
        if self.sock:
            self.sock.close()
            self.sock = None

    # ------------------------------------------------------------------
    # Sending commands
    # ------------------------------------------------------------------
    def send_motor_command(self, left_speed, right_speed, state=0, reward=0, action=0):
        """Send wheel speeds (rad/s). state/reward/action are for the RL template
        and are ignored by the PID flow (leave them at their defaults)."""
        cmd = {
            "command": "set_speed",
            "L": float(left_speed),
            "R": float(right_speed),
            "State": state,
            "Reward": reward,
            "Action": action,
        }
        self.sock.sendall((json.dumps(cmd) + "\n").encode())
        self._send_count += 1
        self._check_frequency()

    def send_pick(self):
        """Ask the bridge to pick the nearest in-range box.

        Returns:
            bool | None: True on success, False on failure, None on timeout.
        """
        self.sock.sendall((json.dumps({"command": "pick"}) + "\n").encode())
        return self._wait_for_reply("pick_result", timeout=2.0)

    def send_drop(self):
        """Ask the bridge to drop the held box.

        Returns:
            bool | None: True on success, False on failure, None on timeout.
        """
        self.sock.sendall((json.dumps({"command": "drop"}) + "\n").encode())
        return self._wait_for_reply("drop_result", timeout=2.0)

    def stop(self):
        """Tell the bridge to zero the motors."""
        self.sock.sendall((json.dumps({"command": "stop"}) + "\n").encode())

    # ------------------------------------------------------------------
    # Receiving sensor data
    # ------------------------------------------------------------------
    def receive_sensor_data(self):
        """Return the NEWEST sensor dict, or None if no packet yet.

        Drains every buffered line and returns the most recent sensor_update,
        discarding older ones, so the control loop always acts on fresh data
        rather than a growing backlog.

        Keys: 'left_corner','left','middle','right','right_corner' — [0,1];
              'proximity' — metres; 'color_r','color_g','color_b' — [0,1].
        """
        try:
            data = self.sock.recv(4096).decode("utf-8", errors="ignore")
            if not data:
                return None
            self.buffer += data
        except socket.timeout:
            pass
        except Exception as e:
            print(f"[CoppeliaClient] Error receiving sensor data: {e}")
            return None

        latest = None
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "sensor_update":
                self._recv_count += 1
                latest = msg["sensors"]
            # pick_result / drop_result lines are handled by _wait_for_reply;
            # if one arrives here (no pending wait) just ignore it.
        return latest

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _wait_for_reply(self, reply_type, timeout=2.0):
        """Block until a pick_result/drop_result arrives; return its 'success'.

        sensor_update lines seen while waiting are set aside and re-queued into
        self.buffer afterwards (they must NOT be pushed back mid-scan, which
        would re-split the same line forever and hang). Returns None on timeout.
        """
        deadline = time.time() + timeout
        leftover = []
        while time.time() < deadline:
            while "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("type") == reply_type:
                    if leftover:
                        self.buffer = "\n".join(leftover) + "\n" + self.buffer
                    return bool(msg.get("success", False))
                # keep sensor packets (and anything else) for later reads
                leftover.append(line)
            try:
                chunk = self.sock.recv(4096).decode("utf-8", errors="ignore")
                if chunk:
                    self.buffer += chunk
            except socket.timeout:
                pass
            except Exception:
                break
        if leftover:
            self.buffer = "\n".join(leftover) + "\n" + self.buffer
        return None

    def _check_frequency(self):
        if self._freq_warned:
            return
        if self._send_count >= 40 and self._send_count > 2 * max(self._recv_count, 1):
            print("[CoppeliaClient] WARNING: your control loop is sending "
                  "commands FASTER than the bridge can read them "
                  f"(sent {self._send_count}, received {self._recv_count} sensor "
                  "packets). Send one command per received sensor packet, or add "
                  "a small delay (e.g. time.sleep(0.05)).")
            self._freq_warned = True
