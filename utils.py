
import socket
import pickle
import struct

def send_tensor(tensor, host, port):
    data = pickle.dumps(tensor)
    s = socket.socket()
    s.connect((host, port))
    s.sendall(struct.pack('>I', len(data)) + data)
    s.close()

def receive_tensor(port):
    s = socket.socket()
    s.bind(('', port))
    s.listen(1)
    conn, _ = s.accept()

    raw_len = recvall(conn, 4)
    if not raw_len:
        return None
    data_len = struct.unpack('>I', raw_len)[0]
    data = recvall(conn, data_len)
    conn.close()
    return pickle.loads(data)

def recvall(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data
