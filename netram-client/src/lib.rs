pub mod tiered;
pub use tiered::{TieredMemory, TieredStats};

pub mod resilient;
pub use resilient::{QueuedWrite, ResilientNetRamClient};

use netram_proto::{
    compute_crc32c, compute_delta_xor, is_all_zeros, select_best_strategy, zstd_dict,
    AdaptiveStrategy, ComputeEngine, ComputeRequest, ComputeTask, Header, OpCode,
    Status, SubPageMode, FLAG_DELTA_XOR, FLAG_LZ4, FLAG_PIPELINED, FLAG_RAW, FLAG_ZERO_PAGE,
    FLAG_ZSTD_DICT, HEADER_SIZE, SUB_PAGE_SIZE,
};
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

pub struct NetRamClient {
    pub stream: TcpStream,
}

fn tune_client_socket(stream: &TcpStream) {
    let _ = stream.set_nodelay(true);

    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let fd = stream.as_raw_fd();
        let buf_size = (8 * 1024 * 1024) as libc::c_int;
        unsafe {
            libc::setsockopt(
                fd,
                libc::SOL_SOCKET,
                libc::SO_RCVBUF,
                &buf_size as *const _ as *const libc::c_void,
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            );
            libc::setsockopt(
                fd,
                libc::SOL_SOCKET,
                libc::SO_SNDBUF,
                &buf_size as *const _ as *const libc::c_void,
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            );
        }
    }

    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawSocket;
        #[link(name = "ws2_32")]
        extern "system" {
            fn setsockopt(s: usize, level: i32, optname: i32, optval: *const i8, optlen: i32) -> i32;
        }
        let sock = stream.as_raw_socket() as usize;
        let buf_size = 8 * 1024 * 1024 as i32;
        const SOL_SOCKET: i32 = 0xffff;
        const SO_SNDBUF: i32 = 0x1001;
        const SO_RCVBUF: i32 = 0x1002;
        unsafe {
            setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &buf_size as *const _ as *const i8, 4);
            setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &buf_size as *const _ as *const i8, 4);
        }
    }
}

impl NetRamClient {
    pub async fn connect(addr: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let stream = TcpStream::connect(addr).await?;
        tune_client_socket(&stream);
        Ok(Self { stream })
    }

    pub async fn ping(&mut self) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        let req = Header::new(OpCode::Ping, 0, 0, 0, 0, 0);
        let start = Instant::now();

        self.stream.write_all(&req.encode()).await?;
        let mut resp_buf = [0u8; HEADER_SIZE];
        self.stream.read_exact(&mut resp_buf).await?;

        let elapsed_us = start.elapsed().as_nanos() as f64 / 1_000.0;
        let resp = Header::decode(&resp_buf)?;
        if resp.status_or_flags != Status::Ok as u8 {
            return Err("Ping falhou".into());
        }

        Ok(elapsed_us)
    }

    pub async fn get_info(&mut self) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let req = Header::new(OpCode::Info, 0, 0, 0, 0, 0);
        self.stream.write_all(&req.encode()).await?;

        let mut resp_buf = [0u8; HEADER_SIZE];
        self.stream.read_exact(&mut resp_buf).await?;
        let resp = Header::decode(&resp_buf)?;

        Ok(resp.offset)
    }

    /// REMOTE MEMORY ATOMIC ADD (lock xadd no hardware em 5ns)
    pub async fn atomic_add(&mut self, offset: u64, val: u64) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let req = Header::new(OpCode::AtomicAdd, 0, 0, offset, 8, 0);
        self.stream.write_all(&req.encode()).await?;
        self.stream.write_all(&val.to_be_bytes()).await?;

        let mut resp_buf = [0u8; HEADER_SIZE];
        self.stream.read_exact(&mut resp_buf).await?;
        let resp = Header::decode(&resp_buf)?;
        if resp.status_or_flags != Status::Ok as u8 {
            return Err(format!("Erro no AtomicAdd: status {:?}", resp.status_or_flags).into());
        }

        let mut prev_buf = [0u8; 8];
        self.stream.read_exact(&mut prev_buf).await?;
        Ok(u64::from_be_bytes(prev_buf))
    }

    /// REMOTE MEMORY ATOMIC COMPARE-AND-SWAP (lock cmpxchg no hardware em 5ns)
    pub async fn atomic_cas(&mut self, offset: u64, expected: u64, desired: u64) -> Result<Result<u64, u64>, Box<dyn std::error::Error + Send + Sync>> {
        let req = Header::new(OpCode::AtomicCas, 0, 0, offset, 16, 0);
        self.stream.write_all(&req.encode()).await?;
        self.stream.write_all(&expected.to_be_bytes()).await?;
        self.stream.write_all(&desired.to_be_bytes()).await?;

        let mut resp_buf = [0u8; HEADER_SIZE];
        self.stream.read_exact(&mut resp_buf).await?;
        let resp = Header::decode(&resp_buf)?;

        let mut val_buf = [0u8; 8];
        self.stream.read_exact(&mut val_buf).await?;
        let val = u64::from_be_bytes(val_buf);

        if resp.status_or_flags == Status::Ok as u8 {
            Ok(Ok(val))
        } else if resp.status_or_flags == Status::ErrCasFailed as u8 {
            Ok(Err(val))
        } else {
            Err(format!("Erro no AtomicCAS: status {:?}", resp.status_or_flags).into())
        }
    }

    /// HYBRID SUB-PAGE CHUNKING (64 KB heterogeneous tiles)
    pub async fn write_hybrid(&mut self, offset: u64, data: &[u8]) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let num_pages = (data.len() + SUB_PAGE_SIZE - 1) / SUB_PAGE_SIZE;
        let mut modes = Vec::with_capacity(num_pages);
        let mut chunks_payload = Vec::new();

        for i in 0..num_pages {
            let start = i * SUB_PAGE_SIZE;
            let end = (start + SUB_PAGE_SIZE).min(data.len());
            let slice = &data[start..end];

            if is_all_zeros(slice) {
                modes.push(SubPageMode::Zero as u8);
            } else {
                let ent = netram_proto::estimate_shannon_entropy(slice);
                if ent > 7.4 {
                    modes.push(SubPageMode::Raw as u8);
                    chunks_payload.extend_from_slice(slice);
                } else if ent > 5.5 {
                    modes.push(SubPageMode::ZstdDict as u8);
                    let comp = zstd_dict::compress_with_dict(slice)?;
                    chunks_payload.extend_from_slice(&(comp.len() as u32).to_be_bytes());
                    chunks_payload.extend_from_slice(&comp);
                } else {
                    modes.push(SubPageMode::Lz4 as u8);
                    let comp = netram_proto::compress::compress_prepend_size(slice);
                    chunks_payload.extend_from_slice(&(comp.len() as u32).to_be_bytes());
                    chunks_payload.extend_from_slice(&comp);
                }
            }
        }

        let total_wire_len = 4 + modes.len() + chunks_payload.len();
        let req = Header::new(OpCode::WriteHybrid, 0, 0, offset, total_wire_len as u32, 0);
        self.stream.write_all(&req.encode()).await?;
        self.stream.write_all(&(num_pages as u32).to_be_bytes()).await?;
        self.stream.write_all(&modes).await?;
        self.stream.write_all(&chunks_payload).await?;

        let mut resp_buf = [0u8; HEADER_SIZE];
        self.stream.read_exact(&mut resp_buf).await?;
        let resp = Header::decode(&resp_buf)?;
        if resp.status_or_flags != Status::Ok as u8 {
            return Err("Falha no WriteHybrid".into());
        }

        Ok(total_wire_len)
    }

    pub async fn write_zstd_dict(&mut self, offset: u64, stream_id: u16, data: &[u8]) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let compressed = zstd_dict::compress_with_dict(data)?;
        let crc = compute_crc32c(&compressed);
        let comp_len = compressed.len();

        let req = Header::new(OpCode::Write, FLAG_ZSTD_DICT, stream_id, offset, comp_len as u32, crc);
        self.stream.write_all(&req.encode()).await?;
        self.stream.write_all(&compressed).await?;

        let mut resp_buf = [0u8; HEADER_SIZE];
        self.stream.read_exact(&mut resp_buf).await?;
        let resp = Header::decode(&resp_buf)?;
        if resp.status_or_flags != Status::Ok as u8 {
            return Err(format!("Falha na gravação ZSTD: status {:?}", resp.status_or_flags).into());
        }

        Ok(comp_len)
    }

    pub async fn write_adaptive(&mut self, offset: u64, data: &[u8]) -> Result<(usize, &'static str), Box<dyn std::error::Error + Send + Sync>> {
        let strategy = select_best_strategy(data);

        match strategy {
            AdaptiveStrategy::ZeroPage => {
                let req = Header::new(OpCode::Write, FLAG_ZERO_PAGE, 0, offset, data.len() as u32, 0);
                self.stream.write_all(&req.encode()).await?;

                let mut resp_buf = [0u8; HEADER_SIZE];
                self.stream.read_exact(&mut resp_buf).await?;
                let resp = Header::decode(&resp_buf)?;
                if resp.status_or_flags != Status::Ok as u8 {
                    return Err("Falha no ZeroPage".into());
                }
                Ok((0, "ADAPTIVE: ZERO-PAGE-SIMD (0 bytes)"))
            }

            AdaptiveStrategy::ZstdDict => {
                let comp_len = self.write_zstd_dict(offset, 1, data).await?;
                Ok((comp_len, "ADAPTIVE: ZSTD-64KB-DICTIONARY"))
            }

            AdaptiveStrategy::Lz4 => {
                let compressed = netram_proto::compress::compress_prepend_size(data);
                let crc = compute_crc32c(&compressed);
                let comp_len = compressed.len();
                let req = Header::new(OpCode::Write, FLAG_LZ4, 0, offset, comp_len as u32, crc);
                self.stream.write_all(&req.encode()).await?;
                self.stream.write_all(&compressed).await?;

                let mut resp_buf = [0u8; HEADER_SIZE];
                self.stream.read_exact(&mut resp_buf).await?;
                let resp = Header::decode(&resp_buf)?;
                if resp.status_or_flags != Status::Ok as u8 {
                    return Err("Falha no LZ4".into());
                }
                Ok((comp_len, "ADAPTIVE: LZ4-COMPRESSED"))
            }

            AdaptiveStrategy::Raw => {
                let crc = compute_crc32c(data);
                let req = Header::new(OpCode::Write, FLAG_RAW, 0, offset, data.len() as u32, crc);
                self.stream.write_all(&req.encode()).await?;
                self.stream.write_all(data).await?;

                let mut resp_buf = [0u8; HEADER_SIZE];
                self.stream.read_exact(&mut resp_buf).await?;
                let resp = Header::decode(&resp_buf)?;
                if resp.status_or_flags != Status::Ok as u8 {
                    return Err("Falha no Raw".into());
                }
                Ok((data.len(), "ADAPTIVE: RAW-ZERO-COPY (Alta Entropia)"))
            }
        }
    }

    /// Envia uma rajada assíncrona de páginas com FLAG_PIPELINED (saturação total do Wi-Fi a 50-70 MB/s)
    pub async fn write_pipelined_stream(
        &mut self,
        offset: u64,
        data: &[u8],
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let is_zeros = is_all_zeros(data);
        if is_zeros {
            let req = Header::new(OpCode::Write, FLAG_ZERO_PAGE | FLAG_PIPELINED, 0, offset, data.len() as u32, 0);
            self.stream.write_all(&req.encode()).await?;
            return Ok(0);
        }

        // Tenta compressão rápida LZ4
        let compressed = netram_proto::compress::compress_prepend_size(data);
        if compressed.len() < data.len() {
            let crc = compute_crc32c(&compressed);
            let comp_len = compressed.len();
            let req = Header::new(OpCode::Write, FLAG_LZ4 | FLAG_PIPELINED, 0, offset, comp_len as u32, crc);
            self.stream.write_all(&req.encode()).await?;
            self.stream.write_all(&compressed).await?;
            Ok(comp_len)
        } else {
            let crc = compute_crc32c(data);
            let req = Header::new(OpCode::Write, FLAG_RAW | FLAG_PIPELINED, 0, offset, data.len() as u32, crc);
            self.stream.write_all(&req.encode()).await?;
            self.stream.write_all(data).await?;
            Ok(data.len())
        }
    }

    pub async fn write_delta_xor(
        &mut self,
        offset: u64,
        old_data: &[u8],
        new_data: &[u8],
    ) -> Result<(usize, &'static str), Box<dyn std::error::Error + Send + Sync>> {
        let delta = compute_delta_xor(old_data, new_data);

        if is_all_zeros(&delta) {
            return Ok((0, "DELTA-ZERO (0 bytes, dados idênticos)"));
        }

        let compressed_delta = netram_proto::compress::compress_prepend_size(&delta);
        let crc = compute_crc32c(&compressed_delta);
        let comp_len = compressed_delta.len();

        let req = Header::new(OpCode::Write, FLAG_DELTA_XOR | FLAG_LZ4, 0, offset, comp_len as u32, crc);
        self.stream.write_all(&req.encode()).await?;
        self.stream.write_all(&compressed_delta).await?;

        let mut resp_buf = [0u8; HEADER_SIZE];
        self.stream.read_exact(&mut resp_buf).await?;
        let resp = Header::decode(&resp_buf)?;
        if resp.status_or_flags != Status::Ok as u8 {
            return Err("Falha na gravação do Delta-XOR".into());
        }

        Ok((comp_len, "DELTA-XOR + LZ4"))
    }

    pub async fn read_adaptive(&mut self, offset: u64, dest_len: usize) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        let req = Header::new(OpCode::Read, FLAG_ZSTD_DICT, 0, offset, dest_len as u32, 0);
        self.stream.write_all(&req.encode()).await?;

        let mut resp_buf = [0u8; HEADER_SIZE];
        self.stream.read_exact(&mut resp_buf).await?;
        let resp = Header::decode(&resp_buf)?;

        if resp.status_or_flags != Status::Ok as u8 && resp.status_or_flags != FLAG_ZSTD_DICT && resp.status_or_flags != FLAG_LZ4 && resp.status_or_flags != FLAG_RAW {
            return Err(format!("Erro na leitura: status {:?}", resp.status_or_flags).into());
        }

        let mut payload = vec![0u8; resp.payload_len as usize];
        self.stream.read_exact(&mut payload).await?;

        if resp.checksum != 0 && compute_crc32c(&payload) != resp.checksum {
            return Err("Corrupção de dados detectada! Checksum CRC32C divergente".into());
        }

        if (resp.status_or_flags & FLAG_ZSTD_DICT) != 0 {
            let decompressed = zstd_dict::decompress_with_dict(&payload)?;
            Ok(decompressed)
        } else if (resp.status_or_flags & FLAG_LZ4) != 0 {
            let decompressed = netram_proto::compress::decompress_size_prepended(&payload)
                .map_err(|e| format!("Falha de descompressão LZ4: {:?}", e))?;
            Ok(decompressed)
        } else {
            Ok(payload)
        }
    }

    pub async fn read_lz4(&mut self, offset: u64, dest_len: usize) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        self.read_adaptive(offset, dest_len).await
    }

    /// Gravação transparente de memória remota via estratégia adaptativa (ZeroCopy / LZ4 / ZSTD)
    pub async fn write(&mut self, offset: u64, data: &[u8]) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        self.write_adaptive(offset, data).await.map(|(len, _)| len)
    }

    /// Leitura transparente de memória remota com descompressão adaptativa automática
    pub async fn read(&mut self, offset: u64, dest_len: usize) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        self.read_adaptive(offset, dest_len).await
    }

    /// Executa computação in-situ nos dados que já moram na RAM física do notebook Windows
    pub async fn remote_compute(
        &mut self,
        engine: ComputeEngine,
        task: ComputeTask,
        offset: u64,
        length: usize,
        param_a: f32,
        param_b: f32,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        let req = ComputeRequest {
            engine: engine as u8,
            _reserved: 0,
            task: task as u16,
            offset,
            length: length as u32,
            param_a,
            param_b,
            extra: 0,
        };
        let req_bytes = req.to_bytes();
        let checksum = compute_crc32c(&req_bytes);

        let header = Header::new(
            OpCode::RemoteCompute,
            0,
            0,
            offset,
            req_bytes.len() as u32,
            checksum,
        );
        self.stream.write_all(&header.encode()).await?;
        self.stream.write_all(&req_bytes).await?;

        let mut resp_buf = [0u8; HEADER_SIZE];
        self.stream.read_exact(&mut resp_buf).await?;
        let resp = Header::decode(&resp_buf)?;

        if resp.status_or_flags != Status::Ok as u8 {
            return Err(format!("Falha na execução remota: status {:?}", resp.status_or_flags).into());
        }

        let mut result = vec![0u8; resp.payload_len as usize];
        if resp.payload_len > 0 {
            self.stream.read_exact(&mut result).await?;
            if resp.checksum != 0 && compute_crc32c(&result) != resp.checksum {
                return Err("Corrupção detectada no resultado da computação remota!".into());
            }
        }

        Ok(result)
    }
}
