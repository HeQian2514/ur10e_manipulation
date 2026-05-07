import socket
import time

class AG95TCPClient:
    '''Instruction format should refer to DH-30A protocol manual'''
    def __init__(self, ip='192.168.0.30', port=8888):
        self.ip = ip
        self.port = port
        self.sock = None

    def connect(self):
        """Connect to AG95 Communication Box TCP Server"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.ip, self.port))
        self.sock.settimeout(2.0)  # ← Set timeout
        print(f"✅ Connected to AG95: {self.ip}:{self.port}")

    def is_autofeedback_enabled(self):
        """Check if auto-feedback is enabled"""
        cmd = "FFFEFDFC010801000000000000FB"
        return self.send_cmd(cmd, wait=True)

    def disconnect(self):
        """Disconnect"""
        if self.sock:
            self.sock.close()
            print("🔌 Connection disconnected")

    def send_cmd(self, hex_cmd_str, wait=False):
        """Send arbitrary AG95 protocol command hex string"""
        # print("🚀 Send command:", hex_cmd_str)
        cmd_bytes = bytes.fromhex(hex_cmd_str)
        self.sock.send(cmd_bytes)
        if wait: time.sleep(0.1)
        while wait:
            resp = self.sock.recv(32)
            # print("📥 Received response:", resp.hex().upper())
            return resp
        return None

    def initialize(self, feedback=True):
        """Initialize gripper"""
        cmd = "FFFEFDFC010802010000000000FB" if feedback else "FFFEDFDC0108010000000000FB"
        return self.send_cmd(cmd, wait=True)


    def set_mode_params(self, position: int, force: int):
        """Set I/O mode position and force"""

        base_code = 0x0602
        cmd1 = f"FFFEFDFC01{base_code:04X}01{position:04X}000000FB"
        base_code = 0x0502
        cmd2 = f"FFFEFDFC01{base_code:04X}01{force:04X}000000FB"
        self.send_cmd(cmd1)
        # time.sleep(0.001)
        self.send_cmd(cmd2)

    def close_gripper(self):
        """Close gripper position 0 force 100"""
        self.set_mode_params(position=0, force=100)

    def open_gripper(self):
        """Open gripper position 100 force 100"""
        self.set_mode_params(position=100, force=100)



if __name__ == "__main__":
    client = AG95TCPClient(ip='192.168.1.29', port=8888)

    try:
        client.connect()
        client.is_autofeedback_enabled()
        # client.set_gripper_id(1)
        # client.set_can_baudrate(500)

        client.initialize()
        time.sleep(5)
        client.set_mode_params(position=20, force=100)  # Should close
        time.sleep(3)
        client.set_mode_params(position=100, force=100) # Should open
        time.sleep(3)
        client.close_gripper()
        time.sleep(3)

        client.open_gripper()
        time.sleep(3)

        client.disconnect()

    except Exception as e:
        print("❌ Program error:", e)
