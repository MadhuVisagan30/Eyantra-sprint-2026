# connector_2a_wrapper_pid.py — communication layer for 2a_wrapper.py (PID + Pick & Place).
#
# Talks to 2a_wrapper.py over TCP (127.0.0.1:50002) using its text protocol.
# The bridge starts the simulation when it launches and stops it on completion / Ctrl+C.
# Don't Edit this File.

import socket
import time


class CoppeliaClient:
    """
    Text-protocol client for 2a_wrapper.py.

    Bridge → Client (one line per step):
        S:lc,l,m,r,rc;P:dist;C:r,g,b\n
            lc,l,m,r,rc  — line sensors (left_corner … right_corner) in [0,1]
            dist         — proximity sensor distance (metres; 1.0 = nothing)
            r,g,b        — RGB from colour sensor [0,1]

    Client → Bridge:
        L:<float>;R:<float>\n   — set left / right motor speeds (rad/s)
        PICK\n                  — pick the box
        DROP\n                  — drop the box

    Bridge → Client (replies):
        PICK:True\n  or  PICK:False\n
        DROP:True\n  or  DROP:False\n
    """

    def __init__(self, host='127.0.0.1', port=50002):
        self.host   = host
        self.port   = port
        self.sock   = None
        self.buffer = ""

        self._send_count  = 0
        self._recv_count  = 0
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
    def send_motor_command(self, left_speed, right_speed):
        """Send wheel speeds to the robot.

        Args:
            left_speed  (float): target velocity for the left  wheel (rad/s).
            right_speed (float): target velocity for the right wheel (rad/s).
        """
        msg = f"L:{left_speed:.4f};R:{right_speed:.4f}\n"
        self.sock.sendall(msg.encode())
        self._send_count += 1
        self._check_frequency()

    def send_pick(self):
        """Ask the bridge to pick the box.

        Returns:
            bool | None: True on success, False on failure, None on timeout.
        """
        self.sock.sendall(b"PICK\n")
        return self._wait_for_reply("PICK", timeout=2.0)

    def send_drop(self):
        """Ask the bridge to drop the currently held box.

        Returns:
            bool | None: True on success, False on failure, None on timeout.
        """
        self.sock.sendall(b"DROP\n")
        return self._wait_for_reply("DROP", timeout=2.0)

    # ------------------------------------------------------------------
    # Receiving sensor data
    # ------------------------------------------------------------------
    def receive_sensor_data(self):
        """Return the latest parsed sensor dict, or None if no packet yet.

        Returned dict keys:
            'left_corner', 'left', 'middle', 'right', 'right_corner'  — [0,1]
            'proximity'    — metres  (1.0 = nothing in range)
            'color_r', 'color_g', 'color_b'                          — [0,1]
        """
        try:
            data = self.sock.recv(4096).decode("utf-8", errors="ignore")
            if not data:
                return None
            self.buffer += data
            while "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                # Skip PICK/DROP reply lines — they are handled by _wait_for_reply
                if line.startswith("PICK:") or line.startswith("DROP:"):
                    continue
                parsed = self._parse_sensor_line(line)
                if parsed is not None:
                    self._recv_count += 1
                    return parsed
        except socket.timeout:
            pass
        except Exception as e:
            print(f"[CoppeliaClient] Error receiving sensor data: {e}")
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _parse_sensor_line(self, line):
        """Parse  S:lc,l,m,r,rc;P:dist;C:r,g,b  into a sensor dict."""
        try:
            sensors = {}
            parts = line.split(";")
            for part in parts:
                if part.startswith("S:"):
                    vals = [float(v) for v in part[2:].split(",")]
                    keys = ["left_corner", "left", "middle", "right", "right_corner"]
                    for k, v in zip(keys, vals):
                        sensors[k] = v
                elif part.startswith("P:"):
                    sensors["proximity"] = float(part[2:])
                elif part.startswith("C:"):
                    r, g, b = [float(v) for v in part[2:].split(",")]
                    sensors["color_r"] = r
                    sensors["color_g"] = g
                    sensors["color_b"] = b
            return sensors if sensors else None
        except Exception:
            return None

    def _wait_for_reply(self, prefix, timeout=2.0):
        """Block until a PICK: or DROP: reply arrives."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                chunk = self.sock.recv(4096).decode("utf-8", errors="ignore")
                if chunk:
                    self.buffer += chunk
            except socket.timeout:
                pass
            except Exception as e:
                print(f"[CoppeliaClient] _wait_for_reply error: {e}")
                break
            while "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                if line.startswith(f"{prefix}:"):
                    val = line.split(":", 1)[1].strip().lower()
                    return val == "true"
                # Re-buffer sensor lines so receive_sensor_data can see them
                if line.startswith("S:"):
                    self.buffer = line + "\n" + self.buffer
        return None

    def _check_frequency(self):
        if self._freq_warned:
            return
        if self._send_count >= 40 and self._send_count > 2 * max(self._recv_count, 1):
            print("[CoppeliaClient] WARNING: your control loop is sending "
                  "commands FASTER than the bridge (~20 Hz) can read them "
                  f"(sent {self._send_count}, received {self._recv_count} sensor "
                  "packets). Add a time.sleep(0.05) or send one command per "
                  "received sensor packet.")
            self._freq_warned = True
