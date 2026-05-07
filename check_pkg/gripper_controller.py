import socket
import time

class AG95TCPClient:
    '''指令格式应参照DH-30A协议手册 连接 TCP Server 地址'''
    def __init__(self, ip='192.168.1.29', port=8888):
        self.ip = ip
        self.port = port
        self.sock = None

    def connect(self):
        """连接到 AG95 通讯盒 TCP Server """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.ip, self.port))
        self.sock.settimeout(2.0)  # ← 设置超时
        print(f"✅ 已连接到 AG95: {self.ip}:{self.port}")

    def is_autofeedback_enabled(self):
        """检查是否启用了自动反馈"""
        cmd = "FFFEFDFC010801000000000000FB"
        return self.send_cmd(cmd, wait=True)

    def disconnect(self):
        """断开连接"""
        if self.sock:
            self.sock.close()
            print("🔌 连接已断开")

    def send_cmd(self, hex_cmd_str, wait=True):
        """发送任意 AG95 协议指令 16进制字符串 """
        # print("🚀 发送指令:", hex_cmd_str)
        cmd_bytes = bytes.fromhex(hex_cmd_str)
        self.sock.send(cmd_bytes)
        time.sleep(0.1)
        while wait:
            resp = self.sock.recv(32)
            # print("📥 收到响应:", resp.hex().upper())
            return resp
        return None

    def initialize(self, feedback=True):
        """初始化夹爪"""
        cmd = "FFFEFDFC010802010000000000FB" if feedback else "FFFEDFDC0108010000000000FB"
        return self.send_cmd(cmd, wait=True)


    def set_mode_params(self, position: int, force: int):
        """设置 I/O 模式位置和力"""

        base_code = 0x0602
        cmd1 = f"FFFEFDFC01{base_code:04X}01{position:04X}000000FB"
        base_code = 0x0502
        cmd2 = f"FFFEFDFC01{base_code:04X}01{force:04X}000000FB"
        self.send_cmd(cmd1)
        time.sleep(0.05)
        self.send_cmd(cmd2)

    def close_gripper(self):
        """闭合夹爪 位置0 力100 """
        self.set_mode_params(position=0, force=100)

    def open_gripper(self):
        """张开夹爪 位置100 力100 """
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
        client.set_mode_params(position=20, force=100)  # 应闭合
        time.sleep(5)
        client.set_mode_params(position=100, force=100) # 应张开
        time.sleep(5)
        client.close_gripper()
        time.sleep(5)

        client.open_gripper()
        time.sleep(5)

        client.disconnect()

    except Exception as e:
        print("❌ 程序出错:", e)
