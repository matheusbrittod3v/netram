import socket
import struct
import time

MAGIC = b'NR'
HEADER_SIZE = 16

class NetRamPythonClient:
    """Cliente Python de Alta Performance para o protocolo NetRAM."""

    OP_PING = 0x00
    OP_ALLOC = 0x01
    OP_WRITE = 0x02
    OP_READ = 0x03
    OP_FREE = 0x04
    OP_INFO = 0x05

    def __init__(self, host: str = "127.0.0.1", port: int = 9876):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((host, port))

    def ping(self) -> float:
        """Retorna o tempo de ida e volta (RTT) em microssegundos."""
        header = MAGIC + struct.pack(">B B Q I", self.OP_PING, 0, 0, 0)
        start = time.perf_counter()
        self.sock.sendall(header)
        resp = self._recv_exact(HEADER_SIZE)
        elapsed_us = (time.perf_counter() - start) * 1_000_000
        return elapsed_us

    def get_remote_ram_size(self) -> int:
        """Retorna o tamanho total da RAM física no servidor (em bytes)."""
        header = MAGIC + struct.pack(">B B Q I", self.OP_INFO, 0, 0, 0)
        self.sock.sendall(header)
        resp = self._recv_exact(HEADER_SIZE)
        _, _, _, offset, _ = struct.unpack(">2s B B Q I", resp)
        return offset

    def write_bytes(self, offset: int, data: bytes | bytearray | memoryview):
        """Grava um buffer de bytes ou tensor na RAM do notebook Windows."""
        header = MAGIC + struct.pack(">B B Q I", self.OP_WRITE, 0, offset, len(data))
        self.sock.sendall(header)
        self.sock.sendall(data)
        resp = self._recv_exact(HEADER_SIZE)
        _, _, status, _, _ = struct.unpack(">2s B B Q I", resp)
        if status != 0:
            raise RuntimeError(f"Erro na gravação remota: status {status}")

    def read_bytes(self, offset: int, length: int) -> bytes:
        """Lê bytes da RAM do notebook Windows diretamente para o Mac."""
        header = MAGIC + struct.pack(">B B Q I", self.OP_READ, 0, offset, length)
        self.sock.sendall(header)
        resp = self._recv_exact(HEADER_SIZE)
        _, _, status, _, payload_len = struct.unpack(">2s B B Q I", resp)
        if status != 0:
            raise RuntimeError(f"Erro na leitura remota: status {status}")
        return self._recv_exact(payload_len)

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray(n)
        view = memoryview(buf)
        pos = 0
        while pos < n:
            received = self.sock.recv_into(view[pos:])
            if received == 0:
                raise ConnectionResetError("Conexão com o servidor NetRAM perdida.")
            pos += received
        return bytes(buf)

    def close(self):
        self.sock.close()

if __name__ == "__main__":
    import numpy as np

    print("--- Teste do NetRAM Client em Python ---")
    client = NetRamPythonClient("127.0.0.1", 9876)
    rtt = client.ping()
    size_gb = client.get_remote_ram_size() / (1024**3)
    print(f"RTT: {rtt:.2f} µs | RAM Remota: {size_gb:.2f} GB")

    # Exemplo com Arrays NumPy / Tensores
    print("Enviando array de 10 milhões de floats (40MB)...")
    array = np.random.randn(10_000_000).astype(np.float32)
    start = time.perf_counter()
    client.write_bytes(0, array.tobytes())
    write_time = (time.perf_counter() - start) * 1000
    print(f"Gravado na RAM remota em {write_time:.2f} ms")

    # Recuperando da RAM remota
    start = time.perf_counter()
    raw = client.read_bytes(0, array.nbytes)
    recovered = np.frombuffer(raw, dtype=np.float32)
    read_time = (time.perf_counter() - start) * 1000
    print(f"Lido da RAM remota em {read_time:.2f} ms")

    assert np.allclose(array, recovered)
    print("✅ Integridade NumPy validada com perfeição!")
    client.close()
