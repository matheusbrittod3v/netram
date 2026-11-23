//! Protocolo Binário de Alta Performance NetRAM v9.0 (HPC Atomics & Hybrid Sub-Page Chunking)
//!
//! Layout do Header (24 bytes fixos alinhados a 64 bits):
//!
//!  0                   1                   2                   3
//!  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |  Magic (0x4E 0x52) 'NR'       | OpCode (1B)   | Status/Flags  |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |          StreamID (u16)       |          Reserved (u16)       |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |                      Offset / Block ID (64-bit)               |
//! |                                                               |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |                      Payload Length (32-bit)                  |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |                      Checksum CRC32C (32-bit)                 |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

pub const MAGIC: [u8; 2] = [0x4E, 0x52]; // 'N', 'R'
pub const HEADER_SIZE: usize = 24;
pub const SUB_PAGE_SIZE: usize = 64 * 1024; // 64 KB por sub-página no Chunking Heterogêneo

// Flags de Payload, Otimização e Resiliência
pub const FLAG_RAW: u8 = 0x00;
pub const FLAG_LZ4: u8 = 0x01;
pub const FLAG_ZERO_PAGE: u8 = 0x02;
pub const FLAG_PIPELINED: u8 = 0x04;
pub const FLAG_DELTA_XOR: u8 = 0x08;
pub const FLAG_SESSION_RESUME: u8 = 0x10;
pub const FLAG_ZSTD_DICT: u8 = 0x20;
pub const FLAG_FEC_PARITY: u8 = 0x40;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpCode {
    Ping = 0x00,
    Alloc = 0x01,
    Write = 0x02,
    Read = 0x03,
    Free = 0x04,
    Info = 0x05,
    WriteV = 0x06,
    ReadV = 0x07,
    Heartbeat = 0x08,
    AtomicCas = 0x09,     // Remote Hardware Compare-And-Swap (lock cmpxchg)
    AtomicAdd = 0x0A,     // Remote Hardware Fetch-And-Add (lock xadd)
    WriteHybrid = 0x0B,   // Hybrid Sub-Page Chunking (64 KB heterogeneous tiles)
    RemoteCompute = 0x0C, // In-Situ CPU (Rayon/AVX) & GPU (OpenCL 1.2) Execution
    PipelineForward = 0x0D, // Pipeline Parallelism: Forward Hidden State (Layers 14-27 on Windows)
    Unknown = 0xFF,
}

impl From<u8> for OpCode {
    fn from(val: u8) -> Self {
        match val {
            0x00 => OpCode::Ping,
            0x01 => OpCode::Alloc,
            0x02 => OpCode::Write,
            0x03 => OpCode::Read,
            0x04 => OpCode::Free,
            0x05 => OpCode::Info,
            0x06 => OpCode::WriteV,
            0x07 => OpCode::ReadV,
            0x08 => OpCode::Heartbeat,
            0x09 => OpCode::AtomicCas,
            0x0A => OpCode::AtomicAdd,
            0x0B => OpCode::WriteHybrid,
            0x0C => OpCode::RemoteCompute,
            0x0D => OpCode::PipelineForward,
            _ => OpCode::Unknown,
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Ok = 0x00,
    ErrInvalidHeader = 0x01,
    ErrOutOfMemory = 0x02,
    ErrOutOfBounds = 0x03,
    ErrUnknownOp = 0x04,
    ErrInternal = 0x05,
    ErrDecompression = 0x06,
    ErrSessionLost = 0x07,
    ErrChecksumMismatch = 0x08,
    ErrCasFailed = 0x09,
    ErrComputeFailed = 0x0A,
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for Status {}

impl From<u8> for Status {
    fn from(val: u8) -> Self {
        match val {
            0x00 => Status::Ok,
            0x01 => Status::ErrInvalidHeader,
            0x02 => Status::ErrOutOfMemory,
            0x03 => Status::ErrOutOfBounds,
            0x04 => Status::ErrUnknownOp,
            0x06 => Status::ErrDecompression,
            0x07 => Status::ErrSessionLost,
            0x08 => Status::ErrChecksumMismatch,
            0x09 => Status::ErrCasFailed,
            0x0A => Status::ErrComputeFailed,
            _ => Status::ErrInternal,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub op: OpCode,
    pub status_or_flags: u8,
    pub stream_id: u16,
    pub offset: u64,
    pub payload_len: u32,
    pub checksum: u32,
}

impl Header {
    pub fn new(op: OpCode, status_or_flags: u8, stream_id: u16, offset: u64, payload_len: u32, checksum: u32) -> Self {
        Self {
            op,
            status_or_flags,
            stream_id,
            offset,
            payload_len,
            checksum,
        }
    }

    #[inline]
    pub fn encode(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0] = MAGIC[0];
        buf[1] = MAGIC[1];
        buf[2] = self.op as u8;
        buf[3] = self.status_or_flags;
        buf[4..6].copy_from_slice(&self.stream_id.to_be_bytes());
        buf[6..8].copy_from_slice(&0u16.to_be_bytes());
        buf[8..16].copy_from_slice(&self.offset.to_be_bytes());
        buf[16..20].copy_from_slice(&self.payload_len.to_be_bytes());
        buf[20..24].copy_from_slice(&self.checksum.to_be_bytes());
        buf
    }

    #[inline]
    pub fn decode(buf: &[u8; HEADER_SIZE]) -> Result<Self, Status> {
        if buf[0] != MAGIC[0] || buf[1] != MAGIC[1] {
            return Err(Status::ErrInvalidHeader);
        }

        let op = OpCode::from(buf[2]);
        if op == OpCode::Unknown {
            return Err(Status::ErrUnknownOp);
        }

        let status_or_flags = buf[3];
        let stream_id = u16::from_be_bytes(buf[4..6].try_into().unwrap());
        let offset = u64::from_be_bytes(buf[8..16].try_into().unwrap());
        let payload_len = u32::from_be_bytes(buf[16..20].try_into().unwrap());
        let checksum = u32::from_be_bytes(buf[20..24].try_into().unwrap());

        Ok(Self {
            op,
            status_or_flags,
            stream_id,
            offset,
            payload_len,
            checksum,
        })
    }
}

#[inline]
pub fn compute_crc32c(data: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(data);
    hasher.finalize()
}

/// Modos de Sub-Página de 64 KB no Chunking Heterogêneo
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubPageMode {
    Zero = 0,     // 0 bytes de dados (memset 0)
    ZstdDict = 1, // Comprimido com ZSTD + Dicionário de 64KB
    Lz4 = 2,      // Comprimido com LZ4
    Raw = 3,      // Dados brutos sem compressão
}

impl From<u8> for SubPageMode {
    fn from(v: u8) -> Self {
        match v {
            0 => SubPageMode::Zero,
            1 => SubPageMode::ZstdDict,
            2 => SubPageMode::Lz4,
            _ => SubPageMode::Raw,
        }
    }
}

pub mod fec {
    pub fn generate_parity(blocks: &[&[u8]]) -> Vec<u8> {
        if blocks.is_empty() {
            return Vec::new();
        }
        let max_len = blocks.iter().map(|b| b.len()).max().unwrap_or(0);
        let mut parity = vec![0u8; max_len];

        for block in blocks {
            for (p, &b) in parity.iter_mut().zip(block.iter()) {
                *p ^= b;
            }
        }
        parity
    }

    pub fn reconstruct_lost_block(surviving_blocks: &[&[u8]], parity: &[u8]) -> Vec<u8> {
        let mut reconstructed = parity.to_vec();
        for block in surviving_blocks {
            for (r, &b) in reconstructed.iter_mut().zip(block.iter()) {
                *r ^= b;
            }
        }
        reconstructed
    }
}

pub mod zstd_dict {
    use std::sync::OnceLock;

    static MEMORY_DICTIONARY: OnceLock<Vec<u8>> = OnceLock::new();

    pub fn get_dictionary() -> &'static [u8] {
        MEMORY_DICTIONARY.get_or_init(|| {
            let mut dict = Vec::with_capacity(64 * 1024);
            for i in 0..(8 * 1024) {
                dict.extend_from_slice(&(i as u64).to_le_bytes());
            }
            dict
        })
    }

    pub fn compress_with_dict(data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
        let dict = get_dictionary();
        let mut encoder = zstd::stream::Encoder::with_dictionary(Vec::new(), 1, dict)?;
        use std::io::Write;
        encoder.write_all(data)?;
        encoder.finish()
    }

    pub fn decompress_with_dict(data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
        let dict = get_dictionary();
        let mut decoder = zstd::stream::Decoder::with_dictionary(data, dict)?;
        use std::io::Read;
        let mut result = Vec::new();
        decoder.read_to_end(&mut result)?;
        Ok(result)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IoSegment {
    pub offset: u64,
    pub len: u32,
}

impl IoSegment {
    pub const SIZE: usize = 12;

    pub fn encode(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..8].copy_from_slice(&self.offset.to_be_bytes());
        buf[8..12].copy_from_slice(&self.len.to_be_bytes());
        buf
    }

    pub fn decode(buf: &[u8; Self::SIZE]) -> Self {
        let offset = u64::from_be_bytes(buf[0..8].try_into().unwrap());
        let len = u32::from_be_bytes(buf[8..12].try_into().unwrap());
        Self { offset, len }
    }
}

#[inline]
pub fn is_all_zeros(data: &[u8]) -> bool {
    let (prefix, chunks, suffix) = unsafe { data.align_to::<u128>() };
    prefix.iter().all(|&b| b == 0)
        && chunks.iter().all(|&w| w == 0)
        && suffix.iter().all(|&b| b == 0)
}

#[inline]
pub fn compute_delta_xor(old: &[u8], new: &[u8]) -> Vec<u8> {
    assert_eq!(old.len(), new.len(), "Buffers de Delta-XOR devem ter o mesmo tamanho");
    let mut delta = vec![0u8; old.len()];

    let (old_prefix, old_chunks, old_suffix) = unsafe { old.align_to::<u128>() };
    let (new_prefix, new_chunks, new_suffix) = unsafe { new.align_to::<u128>() };
    let (delta_prefix, delta_chunks, delta_suffix) = unsafe { delta.align_to_mut::<u128>() };

    for i in 0..old_prefix.len() {
        delta_prefix[i] = old_prefix[i] ^ new_prefix[i];
    }
    for i in 0..old_chunks.len() {
        delta_chunks[i] = old_chunks[i] ^ new_chunks[i];
    }
    for i in 0..old_suffix.len() {
        delta_suffix[i] = old_suffix[i] ^ new_suffix[i];
    }

    delta
}

#[inline]
pub fn apply_delta_xor_in_place(target: &mut [u8], delta: &[u8]) {
    let min_len = target.len().min(delta.len());
    let (target_prefix, target_chunks, target_suffix) = unsafe { target[..min_len].align_to_mut::<u128>() };
    let (delta_prefix, delta_chunks, delta_suffix) = unsafe { delta[..min_len].align_to::<u128>() };

    for i in 0..target_prefix.len() {
        target_prefix[i] ^= delta_prefix[i];
    }
    for i in 0..target_chunks.len() {
        target_chunks[i] ^= delta_chunks[i];
    }
    for i in 0..target_suffix.len() {
        target_suffix[i] ^= delta_suffix[i];
    }
}

pub fn estimate_shannon_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let sample_len = data.len().min(256);
    let step = (data.len() / sample_len).max(1);
    let mut freq = [0u32; 256];
    let mut total_samples = 0;

    let mut i = 0;
    while i < data.len() && total_samples < sample_len {
        freq[data[i] as usize] += 1;
        total_samples += 1;
        i += step;
    }

    let mut entropy = 0.0;
    let total_f = total_samples as f64;
    for &count in &freq {
        if count > 0 {
            let p = count as f64 / total_f;
            entropy -= p * p.log2();
        }
    }

    entropy
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveStrategy {
    ZeroPage,
    Lz4,
    ZstdDict,
    Raw,
}

#[inline]
pub fn select_best_strategy(data: &[u8]) -> AdaptiveStrategy {
    if is_all_zeros(data) {
        return AdaptiveStrategy::ZeroPage;
    }

    let entropy = estimate_shannon_entropy(data);
    if entropy > 7.4 {
        AdaptiveStrategy::Raw
    } else if entropy > 5.5 {
        AdaptiveStrategy::ZstdDict
    } else {
        AdaptiveStrategy::Lz4
    }
}

// ============================================================================
// NETCOMPUTE: ESPECIFICAÇÃO DE EXECUÇÃO REMOTA IN-SITU (CPU & GPU)
// ============================================================================

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeEngine {
    CpuRayonAvx = 0x01,  // Multi-threading 4 cores + AVX SIMD
    GpuIntelOpenCl = 0x02, // Intel HD 4000 16 EUs (OpenCL 1.2 Zero-Copy)
    Auto = 0x03,         // Seleção automática
    Unknown = 0xFF,
}

impl From<u8> for ComputeEngine {
    fn from(val: u8) -> Self {
        match val {
            0x01 => ComputeEngine::CpuRayonAvx,
            0x02 => ComputeEngine::GpuIntelOpenCl,
            0x03 => ComputeEngine::Auto,
            _ => ComputeEngine::Unknown,
        }
    }
}

#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeTask {
    VectorSum        = 0x0001, // Redução rápida de f32
    VectorScale      = 0x0002, // Multiplicação in-place f32 por param_a
    ImageSobel       = 0x0003, // Filtro Sobel 2D de visão computacional
    MatrixMultiply   = 0x0004, // Multiplicação GEMM de blocos
    Unknown          = 0xFFFF,
}

impl From<u16> for ComputeTask {
    fn from(val: u16) -> Self {
        match val {
            0x0001 => ComputeTask::VectorSum,
            0x0002 => ComputeTask::VectorScale,
            0x0003 => ComputeTask::ImageSobel,
            0x0004 => ComputeTask::MatrixMultiply,
            _ => ComputeTask::Unknown,
        }
    }
}

/// Header de Requisição para OpCode::RemoteCompute (32 bytes enviados no payload)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct ComputeRequest {
    pub engine: u8,
    pub _reserved: u8,
    pub task: u16,
    pub offset: u64,
    pub length: u32,
    pub param_a: f32,
    pub param_b: f32,
    pub extra: u64,
}

impl ComputeRequest {
    pub fn to_bytes(&self) -> [u8; 32] {
        let mut buf = [0u8; 32];
        buf[0] = self.engine;
        buf[1] = self._reserved;
        buf[2..4].copy_from_slice(&self.task.to_le_bytes());
        buf[4..12].copy_from_slice(&self.offset.to_le_bytes());
        buf[12..16].copy_from_slice(&self.length.to_le_bytes());
        buf[16..20].copy_from_slice(&self.param_a.to_le_bytes());
        buf[20..24].copy_from_slice(&self.param_b.to_le_bytes());
        buf[24..32].copy_from_slice(&self.extra.to_le_bytes());
        buf
    }

    pub fn from_bytes(slice: &[u8]) -> Option<Self> {
        if slice.len() < 32 {
            return None;
        }
        let engine = slice[0];
        let _reserved = slice[1];
        let task = u16::from_le_bytes(slice[2..4].try_into().ok()?);
        let offset = u64::from_le_bytes(slice[4..12].try_into().ok()?);
        let length = u32::from_le_bytes(slice[12..16].try_into().ok()?);
        let param_a = f32::from_le_bytes(slice[16..20].try_into().ok()?);
        let param_b = f32::from_le_bytes(slice[20..24].try_into().ok()?);
        let extra = u64::from_le_bytes(slice[24..32].try_into().ok()?);

        Some(Self {
            engine,
            _reserved,
            task,
            offset,
            length,
            param_a,
            param_b,
            extra,
        })
    }
}

// ============================================================================
// PIPELINE PARALLELISM: FORWARDING DE HIDDEN STATES (QWEN 2.5 CODER)
// ============================================================================

/// Cabeçalho de Forwarding de Pipeline: Camadas 14 a 27 no Windows
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct PipelineHeader {
    pub start_layer: u8,   // Ex: 14
    pub end_layer: u8,     // Ex: 27
    pub hidden_dim: u16,   // Ex: 1536 floats para Qwen 1.5B (6.144 bytes)
    pub seq_len: u32,      // Número de tokens no batch
    pub temperature: f32,  // Temperatura de amostragem
    pub top_p: f32,        // Top-p sampling
    pub token_id_hint: u32,// ID estimado
    pub flags: u32,
}

impl PipelineHeader {
    pub const SIZE: usize = 24;

    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0] = self.start_layer;
        buf[1] = self.end_layer;
        buf[2..4].copy_from_slice(&self.hidden_dim.to_le_bytes());
        buf[4..8].copy_from_slice(&self.seq_len.to_le_bytes());
        buf[8..12].copy_from_slice(&self.temperature.to_le_bytes());
        buf[12..16].copy_from_slice(&self.top_p.to_le_bytes());
        buf[16..20].copy_from_slice(&self.token_id_hint.to_le_bytes());
        buf[20..24].copy_from_slice(&self.flags.to_le_bytes());
        buf
    }

    pub fn from_bytes(slice: &[u8]) -> Option<Self> {
        if slice.len() < Self::SIZE {
            return None;
        }
        let start_layer = slice[0];
        let end_layer = slice[1];
        let hidden_dim = u16::from_le_bytes(slice[2..4].try_into().ok()?);
        let seq_len = u32::from_le_bytes(slice[4..8].try_into().ok()?);
        let temperature = f32::from_le_bytes(slice[8..12].try_into().ok()?);
        let top_p = f32::from_le_bytes(slice[12..16].try_into().ok()?);
        let token_id_hint = u32::from_le_bytes(slice[16..20].try_into().ok()?);
        let flags = u32::from_le_bytes(slice[20..24].try_into().ok()?);

        Some(Self {
            start_layer,
            end_layer,
            hidden_dim,
            seq_len,
            temperature,
            top_p,
            token_id_hint,
            flags,
        })
    }
}

/// Logit de um candidato com probabilidade
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CandidateLogit {
    pub token_id: u32,
    pub logit: f32,
}

/// Resposta do Pipeline retornada pelo Windows para o Mac
#[derive(Debug, Clone)]
pub struct PipelineResponse {
    pub predicted_token: u32,
    pub compute_time_us: u32,
    pub candidates: Vec<CandidateLogit>,
}

impl PipelineResponse {
    pub fn to_bytes(&self) -> Vec<u8> {
        let count = self.candidates.len().min(64) as u16;
        let mut buf = Vec::with_capacity(12 + (count as usize * 8));
        buf.extend_from_slice(&self.predicted_token.to_le_bytes());
        buf.extend_from_slice(&self.compute_time_us.to_le_bytes());
        buf.extend_from_slice(&count.to_le_bytes());
        buf.extend_from_slice(&[0u8; 2]); // reserved

        for cand in &self.candidates[..count as usize] {
            buf.extend_from_slice(&cand.token_id.to_le_bytes());
            buf.extend_from_slice(&cand.logit.to_le_bytes());
        }
        buf
    }

    pub fn from_bytes(slice: &[u8]) -> Option<Self> {
        if slice.len() < 12 {
            return None;
        }
        let predicted_token = u32::from_le_bytes(slice[0..4].try_into().ok()?);
        let compute_time_us = u32::from_le_bytes(slice[4..8].try_into().ok()?);
        let count = u16::from_le_bytes(slice[8..10].try_into().ok()?) as usize;

        let expected_len = 12 + count * 8;
        if slice.len() < expected_len {
            return None;
        }

        let mut candidates = Vec::with_capacity(count);
        for i in 0..count {
            let offset = 12 + i * 8;
            let token_id = u32::from_le_bytes(slice[offset..offset + 4].try_into().ok()?);
            let logit = f32::from_le_bytes(slice[offset + 4..offset + 8].try_into().ok()?);
            candidates.push(CandidateLogit { token_id, logit });
        }

        Some(Self {
            predicted_token,
            compute_time_us,
            candidates,
        })
    }
}

pub mod compress {
    pub fn compress_prepend_size(input: &[u8]) -> Vec<u8> {
        lz4_flex::compress_prepend_size(input)
    }

    pub fn decompress_size_prepended(input: &[u8]) -> Result<Vec<u8>, lz4_flex::block::DecompressError> {
        lz4_flex::decompress_size_prepended(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_opcodes() {
        assert_eq!(OpCode::from(0x09), OpCode::AtomicCas);
        assert_eq!(OpCode::from(0x0A), OpCode::AtomicAdd);
        assert_eq!(OpCode::from(0x0B), OpCode::WriteHybrid);
    }
}
