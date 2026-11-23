# 🚀 NetRAM & NetCompute

> **Protocolo de Rede Distribuído de Alto Desempenho em Rust para Expansão de Memória RAM e Computação In-Situ Heterogênea (CPU/GPU) entre Máquinas Locais.**

Desenvolvido para conectar a memória RAM física e os processadores de um **notebook Windows** como uma extensão ultra-rápida e resiliente para o **Mac (Apple Silicon)** via rede local (Wi-Fi / Ethernet).

---

## 🌟 Principais Recursos

### 1. Memória Distribuída Zero-Copy (NetRAM)
- **Direct Pinned Memory**: Alocação de memória física contígua com `VirtualAlloc` e `VirtualLock` no Windows (sem swap de disco).
- **Detecção SIMD Zero-Page**: Detecção por hardware de blocos zerados com envio de payload vazio (0 bytes trafegados).
- **Delta-XOR Compression**: Transmissão apenas dos bytes que sofreram mutação em relação ao snapshot anterior (>99% de economia).
- **Zstandard Dictionary & LZ4 Flex**: Compressão adaptativa por análise de entropia em tempo real e dicionários treinados de 64 KB.
- **Hybrid Sub-Page Chunking**: Divisão de blocos grandes em sub-páginas de 64 KB com estratégias heterogêneas por sub-bloco.

### 2. Sincronização & Atomics Remotos de Hardware
- **`OpCode::AtomicAdd` (`lock xadd`)**: Execução de Fetch-And-Add atômico direto no silício do processador Windows em ~5 ns.
- **`OpCode::AtomicCas` (`lock cmpxchg`)**: Operação atômica Compare-And-Swap distribuída para construção de Spinlocks e Mutexes remotos.

### 3. Computação In-Situ na RAM (NetCompute)
- **"Mova o Código, Não os Dados"**: Processamento direto onde os dados residem na RAM física do Windows a 60 GB/s.
- **Motor CPU Intel Multi-core (Rayon + AVX SIMD)**: Paralelização automática com redução vetorial (`VectorSum`) e escala in-place (`VectorScale`).
- **Motor GPU Intel HD 4000 (OpenCL 1.2)**: Acesso direto por buffer de host compartilhado (`CL_MEM_USE_HOST_PTR`) para convolução 2D Sobel e filtros matriciais a 332 GFLOPS.

### 4. Alta Disponibilidade & Resiliência
- **Forward Error Correction (FEC)**: Paridade XOR de reconstrução instantânea para perdas de pacotes no Wi-Fi.
- **Checksum Espacial CRC32C**: Integridade validada por instruções de silício a 30 GB/s.
- **Auto-Reconnect & Journaling**: Tolerância a oscilações de sinal de rede com reconexão exponencial e envio de deltas pendentes.

---

## 📁 Estrutura do Workspace

```
.
├── netram-proto/       # Especificação do Wire Protocol (Header de 24B, Atomics, NetCompute)
├── netram-server/      # Servidor Rust nativo para Windows (VirtualAlloc, Rayon, OpenCL)
├── netram-client/      # Biblioteca e CLI client para macOS (Tiered Memory, LRU, Resiliência)
└── netram_python_example.py # Cliente interoperável em Python com NumPy
```

---

## ⚡ Como Executar

### 1. Iniciar o Servidor no Windows
```bash
cargo run --release -p netram-server -- 9876 3
```
*(Escuta na porta 9876 e aloca 3 GB de RAM física Pinned).*

### 2. Rodar o Benchmark no Mac
```bash
cargo run --release -p netram-client -- 192.168.1.6:9876
```

### 3. Exemplo em Python (NumPy)
```python
import numpy as np
from netram_python_example import NetRamClient

client = NetRamClient("192.168.1.6", 9876)
data = np.arange(1000000, dtype=np.float32).tobytes()

# Gravação adaptativa na RAM física do Windows
client.write(offset=0, data=data)

# Leitura com descompressão automática
recovered = client.read(offset=0, size=len(data))
```

---

## 📜 Licença
MIT License.
