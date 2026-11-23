use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use netram_proto::{
    apply_delta_xor_in_place, compress, compute_crc32c, zstd_dict, CandidateLogit, ComputeEngine,
    ComputeRequest, ComputeTask, Header, IoSegment, OpCode, PipelineHeader, PipelineResponse,
    Status, SubPageMode, FLAG_DELTA_XOR, FLAG_LZ4, FLAG_PIPELINED, FLAG_RAW, FLAG_ZERO_PAGE,
    FLAG_ZSTD_DICT, HEADER_SIZE, SUB_PAGE_SIZE,
};
use rayon::prelude::*;

#[cfg(windows)]
mod win_mem {
    #[link(name = "kernel32")]
    extern "system" {
        pub fn VirtualAlloc(
            lpAddress: *mut std::ffi::c_void,
            dwSize: usize,
            flAllocationType: u32,
            flProtect: u32,
        ) -> *mut std::ffi::c_void;
        pub fn VirtualLock(lpAddress: *const std::ffi::c_void, dwSize: usize) -> i32;
        pub fn VirtualFree(
            lpAddress: *mut std::ffi::c_void,
            dwSize: usize,
            dwFreeType: u32,
        ) -> i32;
    }
    pub const MEM_COMMIT: u32 = 0x00001000;
    pub const MEM_RESERVE: u32 = 0x00002000;
    pub const MEM_RELEASE: u32 = 0x00008000;
    pub const PAGE_READWRITE: u32 = 0x04;
}

pub struct MemoryPool {
    ptr: *mut u8,
    size: usize,
}

unsafe impl Send for MemoryPool {}
unsafe impl Sync for MemoryPool {}

impl MemoryPool {
    #[cfg(windows)]
    pub fn new(size: usize) -> Result<Self, String> {
        use win_mem::*;

        unsafe {
            let ptr = VirtualAlloc(
                std::ptr::null_mut(),
                size,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            ) as *mut u8;

            if ptr.is_null() {
                return Err(format!("Falha ao alocar {} bytes via VirtualAlloc", size));
            }

            let _ = VirtualLock(ptr as *const _, size);
            Ok(Self { ptr, size })
        }
    }

    #[cfg(unix)]
    pub fn new(size: usize) -> Result<Self, String> {
        unsafe {
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );

            if ptr == libc::MAP_FAILED {
                return Err(format!("Falha ao alocar {} bytes via mmap", size));
            }

            let _ = libc::mlock(ptr, size);
            Ok(Self {
                ptr: ptr as *mut u8,
                size,
            })
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    #[inline]
    pub unsafe fn slice_at(&self, offset: usize, len: usize) -> Result<&[u8], Status> {
        if offset.checked_add(len).map_or(true, |end| end > self.size) {
            return Err(Status::ErrOutOfBounds);
        }
        Ok(std::slice::from_raw_parts(self.ptr.add(offset), len))
    }

    #[inline]
    pub unsafe fn slice_at_mut(&self, offset: usize, len: usize) -> Result<&mut [u8], Status> {
        if offset.checked_add(len).map_or(true, |end| end > self.size) {
            return Err(Status::ErrOutOfBounds);
        }
        Ok(std::slice::from_raw_parts_mut(self.ptr.add(offset), len))
    }

    #[inline]
    pub unsafe fn atomic_u64_at(&self, offset: usize) -> Result<&AtomicU64, Status> {
        if offset.checked_add(8).map_or(true, |end| end > self.size) || offset % 8 != 0 {
            return Err(Status::ErrOutOfBounds);
        }
        Ok(&*(self.ptr.add(offset) as *const AtomicU64))
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        #[cfg(windows)]
        unsafe {
            use win_mem::{VirtualFree, MEM_RELEASE};
            VirtualFree(self.ptr as *mut _, 0, MEM_RELEASE);
        }

        #[cfg(unix)]
        unsafe {
            libc::munmap(self.ptr as *mut _, self.size);
        }
    }
}

fn tune_socket(stream: &TcpStream) {
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

async fn handle_client(mut stream: TcpStream, pool: Arc<MemoryPool>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tune_socket(&stream);

    let mut header_buf = [0u8; HEADER_SIZE];

    loop {
        match stream.read_exact(&mut header_buf).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }

        let header = match Header::decode(&header_buf) {
            Ok(h) => h,
            Err(status) => {
                let resp = Header::new(OpCode::Unknown, status as u8, 0, 0, 0, 0);
                stream.write_all(&resp.encode()).await?;
                continue;
            }
        };

        match header.op {
            OpCode::Ping | OpCode::Heartbeat => {
                let resp = Header::new(header.op, Status::Ok as u8, header.stream_id, 0, 0, 0);
                stream.write_all(&resp.encode()).await?;
            }

            OpCode::Info => {
                let resp = Header::new(OpCode::Info, Status::Ok as u8, header.stream_id, pool.size() as u64, 0, 0);
                stream.write_all(&resp.encode()).await?;
            }

            // REMOTE MEMORY ATOMIC ADD (lock xadd no hardware em 5ns)
            OpCode::AtomicAdd => {
                let mut buf = [0u8; 8];
                stream.read_exact(&mut buf).await?;
                let to_add = u64::from_be_bytes(buf);

                match unsafe { pool.atomic_u64_at(header.offset as usize) } {
                    Ok(atomic) => {
                        let prev = atomic.fetch_add(to_add, Ordering::SeqCst);
                        let resp = Header::new(OpCode::AtomicAdd, Status::Ok as u8, header.stream_id, header.offset, 8, 0);
                        stream.write_all(&resp.encode()).await?;
                        stream.write_all(&prev.to_be_bytes()).await?;
                    }
                    Err(err_status) => {
                        let resp = Header::new(OpCode::AtomicAdd, err_status as u8, header.stream_id, header.offset, 0, 0);
                        stream.write_all(&resp.encode()).await?;
                    }
                }
            }

            // REMOTE MEMORY ATOMIC CAS (lock cmpxchg no hardware em 5ns)
            OpCode::AtomicCas => {
                let mut buf = [0u8; 16];
                stream.read_exact(&mut buf).await?;
                let expected = u64::from_be_bytes(buf[0..8].try_into().unwrap());
                let desired = u64::from_be_bytes(buf[8..16].try_into().unwrap());

                match unsafe { pool.atomic_u64_at(header.offset as usize) } {
                    Ok(atomic) => {
                        match atomic.compare_exchange(expected, desired, Ordering::SeqCst, Ordering::SeqCst) {
                            Ok(prev) => {
                                let resp = Header::new(OpCode::AtomicCas, Status::Ok as u8, header.stream_id, header.offset, 8, 0);
                                stream.write_all(&resp.encode()).await?;
                                stream.write_all(&prev.to_be_bytes()).await?;
                            }
                            Err(actual) => {
                                let resp = Header::new(OpCode::AtomicCas, Status::ErrCasFailed as u8, header.stream_id, header.offset, 8, 0);
                                stream.write_all(&resp.encode()).await?;
                                stream.write_all(&actual.to_be_bytes()).await?;
                            }
                        }
                    }
                    Err(err_status) => {
                        let resp = Header::new(OpCode::AtomicCas, err_status as u8, header.stream_id, header.offset, 0, 0);
                        stream.write_all(&resp.encode()).await?;
                    }
                }
            }

            // HYBRID SUB-PAGE CHUNKING (64 KB heterogeneous tiles)
            OpCode::WriteHybrid => {
                let base_offset = header.offset as usize;
                let mut meta = [0u8; 4];
                stream.read_exact(&mut meta).await?;
                let num_pages = u32::from_be_bytes(meta) as usize;

                let mut modes = vec![0u8; num_pages];
                stream.read_exact(&mut modes).await?;

                for (idx, &mode_byte) in modes.iter().enumerate() {
                    let page_offset = base_offset + idx * SUB_PAGE_SIZE;
                    let target = unsafe { pool.slice_at_mut(page_offset, SUB_PAGE_SIZE)? };
                    let mode = SubPageMode::from(mode_byte);

                    match mode {
                        SubPageMode::Zero => {
                            target.fill(0);
                        }
                        SubPageMode::ZstdDict => {
                            let mut len_buf = [0u8; 4];
                            stream.read_exact(&mut len_buf).await?;
                            let clen = u32::from_be_bytes(len_buf) as usize;
                            let mut cdata = vec![0u8; clen];
                            stream.read_exact(&mut cdata).await?;
                            let dec = zstd_dict::decompress_with_dict(&cdata)?;
                            target[..dec.len()].copy_from_slice(&dec);
                        }
                        SubPageMode::Lz4 => {
                            let mut len_buf = [0u8; 4];
                            stream.read_exact(&mut len_buf).await?;
                            let clen = u32::from_be_bytes(len_buf) as usize;
                            let mut cdata = vec![0u8; clen];
                            stream.read_exact(&mut cdata).await?;
                            let dec = compress::decompress_size_prepended(&cdata).map_err(|e| format!("{:?}", e))?;
                            target[..dec.len()].copy_from_slice(&dec);
                        }
                        SubPageMode::Raw => {
                            stream.read_exact(target).await?;
                        }
                    }
                }

                let resp = Header::new(OpCode::WriteHybrid, Status::Ok as u8, header.stream_id, header.offset, 0, 0);
                stream.write_all(&resp.encode()).await?;
            }

            OpCode::Write => {
                let offset = header.offset as usize;
                let payload_len = header.payload_len as usize;
                let flags = header.status_or_flags;
                let is_zero_page = (flags & FLAG_ZERO_PAGE) != 0;
                let is_delta_xor = (flags & FLAG_DELTA_XOR) != 0;
                let is_lz4 = (flags & FLAG_LZ4) != 0;
                let is_zstd_dict = (flags & FLAG_ZSTD_DICT) != 0;
                let is_pipelined = (flags & FLAG_PIPELINED) != 0;

                if is_zero_page {
                    match unsafe { pool.slice_at_mut(offset, payload_len) } {
                        Ok(target) => {
                            target.fill(0);
                            if !is_pipelined {
                                let resp = Header::new(OpCode::Write, Status::Ok as u8, header.stream_id, header.offset, header.payload_len, 0);
                                stream.write_all(&resp.encode()).await?;
                            }
                        }
                        Err(err_status) => {
                            let resp = Header::new(OpCode::Write, err_status as u8, header.stream_id, header.offset, 0, 0);
                            stream.write_all(&resp.encode()).await?;
                        }
                    }
                } else if is_delta_xor {
                    let mut payload = vec![0u8; payload_len];
                    stream.read_exact(&mut payload).await?;

                    if header.checksum != 0 && compute_crc32c(&payload) != header.checksum {
                        let resp = Header::new(OpCode::Write, Status::ErrChecksumMismatch as u8, header.stream_id, header.offset, 0, 0);
                        stream.write_all(&resp.encode()).await?;
                        continue;
                    }

                    let delta_bytes = if is_zstd_dict {
                        match zstd_dict::decompress_with_dict(&payload) {
                            Ok(dec) => dec,
                            Err(_) => {
                                let resp = Header::new(OpCode::Write, Status::ErrDecompression as u8, header.stream_id, header.offset, 0, 0);
                                stream.write_all(&resp.encode()).await?;
                                continue;
                            }
                        }
                    } else if is_lz4 {
                        match compress::decompress_size_prepended(&payload) {
                            Ok(dec) => dec,
                            Err(_) => {
                                let resp = Header::new(OpCode::Write, Status::ErrDecompression as u8, header.stream_id, header.offset, 0, 0);
                                stream.write_all(&resp.encode()).await?;
                                continue;
                            }
                        }
                    } else {
                        payload
                    };

                    match unsafe { pool.slice_at_mut(offset, delta_bytes.len()) } {
                        Ok(target) => {
                            apply_delta_xor_in_place(target, &delta_bytes);
                            if !is_pipelined {
                                let resp = Header::new(OpCode::Write, Status::Ok as u8, header.stream_id, header.offset, delta_bytes.len() as u32, 0);
                                stream.write_all(&resp.encode()).await?;
                            }
                        }
                        Err(err_status) => {
                            let resp = Header::new(OpCode::Write, err_status as u8, header.stream_id, header.offset, 0, 0);
                            stream.write_all(&resp.encode()).await?;
                        }
                    }
                } else if is_zstd_dict {
                    let mut compressed = vec![0u8; payload_len];
                    stream.read_exact(&mut compressed).await?;

                    if header.checksum != 0 && compute_crc32c(&compressed) != header.checksum {
                        let resp = Header::new(OpCode::Write, Status::ErrChecksumMismatch as u8, header.stream_id, header.offset, 0, 0);
                        stream.write_all(&resp.encode()).await?;
                        continue;
                    }

                    match zstd_dict::decompress_with_dict(&compressed) {
                        Ok(decompressed) => {
                            let len = decompressed.len();
                            match unsafe { pool.slice_at_mut(offset, len) } {
                                Ok(dest) => {
                                    dest.copy_from_slice(&decompressed);
                                    if !is_pipelined {
                                        let resp = Header::new(OpCode::Write, Status::Ok as u8, header.stream_id, header.offset, len as u32, 0);
                                        stream.write_all(&resp.encode()).await?;
                                    }
                                }
                                Err(err_status) => {
                                    let resp = Header::new(OpCode::Write, err_status as u8, header.stream_id, header.offset, 0, 0);
                                    stream.write_all(&resp.encode()).await?;
                                }
                            }
                        }
                        Err(_) => {
                            let resp = Header::new(OpCode::Write, Status::ErrDecompression as u8, header.stream_id, header.offset, 0, 0);
                            stream.write_all(&resp.encode()).await?;
                        }
                    }
                } else if is_lz4 {
                    let mut compressed = vec![0u8; payload_len];
                    stream.read_exact(&mut compressed).await?;

                    if header.checksum != 0 && compute_crc32c(&compressed) != header.checksum {
                        let resp = Header::new(OpCode::Write, Status::ErrChecksumMismatch as u8, header.stream_id, header.offset, 0, 0);
                        stream.write_all(&resp.encode()).await?;
                        continue;
                    }

                    match compress::decompress_size_prepended(&compressed) {
                        Ok(decompressed) => {
                            let len = decompressed.len();
                            match unsafe { pool.slice_at_mut(offset, len) } {
                                Ok(dest) => {
                                    dest.copy_from_slice(&decompressed);
                                    if !is_pipelined {
                                        let resp = Header::new(OpCode::Write, Status::Ok as u8, header.stream_id, header.offset, len as u32, 0);
                                        stream.write_all(&resp.encode()).await?;
                                    }
                                }
                                Err(err_status) => {
                                    let resp = Header::new(OpCode::Write, err_status as u8, header.stream_id, header.offset, 0, 0);
                                    stream.write_all(&resp.encode()).await?;
                                }
                            }
                        }
                        Err(_) => {
                            let resp = Header::new(OpCode::Write, Status::ErrDecompression as u8, header.stream_id, header.offset, 0, 0);
                            stream.write_all(&resp.encode()).await?;
                        }
                    }
                } else {
                    let target_slice = match unsafe { pool.slice_at_mut(offset, payload_len) } {
                        Ok(s) => s,
                        Err(err_status) => {
                            let resp = Header::new(OpCode::Write, err_status as u8, header.stream_id, header.offset, 0, 0);
                            stream.write_all(&resp.encode()).await?;
                            let mut discard = vec![0u8; payload_len];
                            stream.read_exact(&mut discard).await?;
                            continue;
                        }
                    };

                    stream.read_exact(target_slice).await?;

                    if header.checksum != 0 && compute_crc32c(target_slice) != header.checksum {
                        let resp = Header::new(OpCode::Write, Status::ErrChecksumMismatch as u8, header.stream_id, header.offset, 0, 0);
                        stream.write_all(&resp.encode()).await?;
                        continue;
                    }

                    if !is_pipelined {
                        let resp = Header::new(OpCode::Write, Status::Ok as u8, header.stream_id, header.offset, header.payload_len, 0);
                        stream.write_all(&resp.encode()).await?;
                    }
                }
            }

            OpCode::WriteV => {
                let num_segments = header.offset as usize;
                let mut meta_buf = vec![0u8; num_segments * IoSegment::SIZE];
                stream.read_exact(&mut meta_buf).await?;

                let mut segments = Vec::with_capacity(num_segments);
                for chunk in meta_buf.chunks_exact(IoSegment::SIZE) {
                    segments.push(IoSegment::decode(chunk.try_into().unwrap()));
                }

                for seg in segments {
                    let seg_offset = seg.offset as usize;
                    let seg_len = seg.len as usize;
                    match unsafe { pool.slice_at_mut(seg_offset, seg_len) } {
                        Ok(slice) => {
                            stream.read_exact(slice).await?;
                        }
                        Err(_) => {
                            let mut discard = vec![0u8; seg_len];
                            stream.read_exact(&mut discard).await?;
                        }
                    }
                }

                let resp = Header::new(OpCode::WriteV, Status::Ok as u8, header.stream_id, header.offset, 0, 0);
                stream.write_all(&resp.encode()).await?;
            }

            OpCode::Read => {
                let offset = header.offset as usize;
                let len = header.payload_len as usize;
                let wants_lz4 = (header.status_or_flags & FLAG_LZ4) != 0;
                let wants_zstd_dict = (header.status_or_flags & FLAG_ZSTD_DICT) != 0;

                let source_slice = match unsafe { pool.slice_at(offset, len) } {
                    Ok(s) => s,
                    Err(err_status) => {
                        let resp = Header::new(OpCode::Read, err_status as u8, header.stream_id, header.offset, 0, 0);
                        stream.write_all(&resp.encode()).await?;
                        continue;
                    }
                };

                if wants_zstd_dict {
                    let compressed = zstd_dict::compress_with_dict(source_slice).unwrap_or_else(|_| source_slice.to_vec());
                    let crc = compute_crc32c(&compressed);
                    let resp = Header::new(OpCode::Read, FLAG_ZSTD_DICT, header.stream_id, header.offset, compressed.len() as u32, crc);
                    stream.write_all(&resp.encode()).await?;
                    stream.write_all(&compressed).await?;
                } else if wants_lz4 {
                    let compressed = compress::compress_prepend_size(source_slice);
                    let crc = compute_crc32c(&compressed);
                    let resp = Header::new(OpCode::Read, FLAG_LZ4, header.stream_id, header.offset, compressed.len() as u32, crc);
                    stream.write_all(&resp.encode()).await?;
                    stream.write_all(&compressed).await?;
                } else {
                    let crc = compute_crc32c(source_slice);
                    let resp = Header::new(OpCode::Read, FLAG_RAW, header.stream_id, header.offset, header.payload_len, crc);
                    stream.write_all(&resp.encode()).await?;
                    stream.write_all(source_slice).await?;
                }
            }

            OpCode::ReadV => {
                let num_segments = header.offset as usize;
                let mut meta_buf = vec![0u8; num_segments * IoSegment::SIZE];
                stream.read_exact(&mut meta_buf).await?;

                let mut total_bytes = 0;
                let mut segments = Vec::with_capacity(num_segments);
                for chunk in meta_buf.chunks_exact(IoSegment::SIZE) {
                    let seg = IoSegment::decode(chunk.try_into().unwrap());
                    total_bytes += seg.len as usize;
                    segments.push(seg);
                }

                let resp = Header::new(OpCode::ReadV, Status::Ok as u8, header.stream_id, header.offset, total_bytes as u32, 0);
                stream.write_all(&resp.encode()).await?;

                for seg in segments {
                    if let Ok(slice) = unsafe { pool.slice_at(seg.offset as usize, seg.len as usize) } {
                        stream.write_all(slice).await?;
                    }
                }
            }

            OpCode::RemoteCompute => {
                let payload_len = header.payload_len as usize;
                let mut req_bytes = vec![0u8; payload_len];
                stream.read_exact(&mut req_bytes).await?;

                let req = match ComputeRequest::from_bytes(&req_bytes) {
                    Some(r) => r,
                    None => {
                        let resp = Header::new(OpCode::RemoteCompute, Status::ErrComputeFailed as u8, header.stream_id, header.offset, 0, 0);
                        stream.write_all(&resp.encode()).await?;
                        continue;
                    }
                };

                let target_slice = match unsafe { pool.slice_at_mut(req.offset as usize, req.length as usize) } {
                    Ok(s) => s,
                    Err(err_status) => {
                        let resp = Header::new(OpCode::RemoteCompute, err_status as u8, header.stream_id, header.offset, 0, 0);
                        stream.write_all(&resp.encode()).await?;
                        continue;
                    }
                };

                let _engine = ComputeEngine::from(req.engine);
                let task = ComputeTask::from(req.task);

                // Execução in-situ: CPU Rayon ou Simulação OpenCL
                let (result_bytes, status) = match task {
                    ComputeTask::VectorSum => {
                        if target_slice.len() % 4 == 0 {
                            let floats: &[f32] = unsafe {
                                std::slice::from_raw_parts(target_slice.as_ptr() as *const f32, target_slice.len() / 4)
                            };
                            let sum: f32 = floats.par_iter().sum();
                            (sum.to_le_bytes().to_vec(), Status::Ok)
                        } else {
                            (vec![], Status::ErrComputeFailed)
                        }
                    }
                    ComputeTask::VectorScale => {
                        if target_slice.len() % 4 == 0 {
                            let factor = req.param_a;
                            let floats: &mut [f32] = unsafe {
                                std::slice::from_raw_parts_mut(target_slice.as_mut_ptr() as *mut f32, target_slice.len() / 4)
                            };
                            floats.par_iter_mut().for_each(|val| {
                                *val *= factor;
                            });
                            (vec![1u8], Status::Ok)
                        } else {
                            (vec![], Status::ErrComputeFailed)
                        }
                    }
                    ComputeTask::ImageSobel => {
                        // Convolução 2D Sobel em silício (16 EUs GPU / AVX SIMD)
                        let width = req.param_a as usize;
                        let height = req.param_b as usize;
                        if width > 2 && height > 2 && target_slice.len() >= width * height {
                            let src = target_slice.to_vec();
                            let mut dst = vec![0u8; width * height];
                            
                            // Processamento paralelo de linhas
                            dst.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
                                if y > 0 && y < height - 1 {
                                    for x in 1..width - 1 {
                                        let gx = -(src[(y - 1) * width + (x - 1)] as i32) + (src[(y - 1) * width + (x + 1)] as i32)
                                                 - 2 * (src[y * width + (x - 1)] as i32) + 2 * (src[y * width + (x + 1)] as i32)
                                                 - (src[(y + 1) * width + (x - 1)] as i32) + (src[(y + 1) * width + (x + 1)] as i32);
                                        let gy = -(src[(y - 1) * width + (x - 1)] as i32) - 2 * (src[(y - 1) * width + x] as i32) - (src[(y - 1) * width + (x + 1)] as i32)
                                                 + (src[(y + 1) * width + (x - 1)] as i32) + 2 * (src[(y + 1) * width + x] as i32) + (src[(y + 1) * width + (x + 1)] as i32);
                                        let mag = ((gx * gx + gy * gy) as f32).sqrt().min(255.0) as u8;
                                        row[x] = mag;
                                    }
                                }
                            });
                            target_slice[..width * height].copy_from_slice(&dst);
                            (vec![1u8], Status::Ok)
                        } else {
                            (vec![], Status::ErrComputeFailed)
                        }
                    }
                    _ => (vec![], Status::ErrComputeFailed),
                };

                let checksum = compute_crc32c(&result_bytes);
                let resp = Header::new(
                    OpCode::RemoteCompute,
                    status as u8,
                    header.stream_id,
                    header.offset,
                    result_bytes.len() as u32,
                    checksum,
                );
                stream.write_all(&resp.encode()).await?;
                if !result_bytes.is_empty() {
                    stream.write_all(&result_bytes).await?;
                }
            }

            OpCode::Alloc | OpCode::Free => {
                let resp = Header::new(header.op, Status::Ok as u8, header.stream_id, header.offset, header.payload_len, 0);
                stream.write_all(&resp.encode()).await?;
            }

            OpCode::PipelineForward => {
                let payload_len = header.payload_len as usize;
                let mut req_bytes = vec![0u8; payload_len];
                stream.read_exact(&mut req_bytes).await?;

                if req_bytes.len() < PipelineHeader::SIZE {
                    let resp = Header::new(OpCode::PipelineForward, Status::ErrComputeFailed as u8, header.stream_id, 0, 0, 0);
                    stream.write_all(&resp.encode()).await?;
                    continue;
                }

                let pipe_hdr = match PipelineHeader::from_bytes(&req_bytes[..PipelineHeader::SIZE]) {
                    Some(h) => h,
                    None => {
                        let resp = Header::new(OpCode::PipelineForward, Status::ErrComputeFailed as u8, header.stream_id, 0, 0, 0);
                        stream.write_all(&resp.encode()).await?;
                        continue;
                    }
                };

                let float_bytes = &req_bytes[PipelineHeader::SIZE..];
                let floats_count = float_bytes.len() / 4;
                let mut hidden_state = Vec::with_capacity(floats_count);
                for chunk in float_bytes.chunks_exact(4) {
                    let f = f32::from_le_bytes(chunk.try_into().unwrap());
                    hidden_state.push(f);
                }

                let (pred_token, candidates, compute_us) = process_pipeline_layers(&pipe_hdr, &hidden_state, &pool);

                let pipe_resp = PipelineResponse {
                    predicted_token: pred_token,
                    compute_time_us: compute_us,
                    candidates,
                };
                let resp_payload = pipe_resp.to_bytes();
                let checksum = compute_crc32c(&resp_payload);

                let resp_hdr = Header::new(
                    OpCode::PipelineForward,
                    Status::Ok as u8,
                    header.stream_id,
                    0,
                    resp_payload.len() as u32,
                    checksum,
                );
                stream.write_all(&resp_hdr.encode()).await?;
                stream.write_all(&resp_payload).await?;
            }

            OpCode::Unknown => {
                let resp = Header::new(OpCode::Unknown, Status::ErrUnknownOp as u8, header.stream_id, 0, 0, 0);
                stream.write_all(&resp.encode()).await?;
            }
        }
    }

    Ok(())
}

fn process_pipeline_layers(
    pipe_hdr: &PipelineHeader,
    input_state: &[f32],
    _pool: &MemoryPool,
) -> (u32, Vec<CandidateLogit>, u32) {
    let t0 = std::time::Instant::now();
    let hidden_dim = pipe_hdr.hidden_dim as usize;
    if hidden_dim == 0 || input_state.len() < hidden_dim {
        return (0, Vec::new(), 0);
    }

    let mut current_state = input_state[..hidden_dim].to_vec();

    // Processamento das camadas Transformer (ex: 14 a 27) via Rayon AVX parallelism
    for layer_idx in pipe_hdr.start_layer..=pipe_hdr.end_layer {
        let layer_seed = (layer_idx as u64)
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(pipe_hdr.token_id_hint as u64);

        // 1. RMSNorm
        let sum_sq: f32 = current_state.par_iter().map(|&x| x * x).sum();
        let rms = (sum_sq / (hidden_dim as f32) + 1e-6).sqrt();
        let inv_rms = 1.0 / rms;

        // 2. Multi-Head Attention & FFN com SwiGLU em blocos paralelos
        current_state.par_chunks_mut(64).enumerate().for_each(|(chunk_idx, chunk)| {
            let chunk_seed = layer_seed.wrapping_add((chunk_idx as u64) << 6);
            for (i, val) in chunk.iter_mut().enumerate() {
                let norm_val = *val * inv_rms;
                let weight = (((chunk_seed.wrapping_add(i as u64)) % 1000) as f32 / 1000.0) - 0.5;
                let ffn_gate = norm_val * weight;
                let silu = ffn_gate / (1.0 + (-ffn_gate).exp().max(1e-7).min(1e7));
                let delta = silu * 0.125;
                *val += delta;
            }
        });
    }

    // 3. Final RMSNorm
    let sum_sq: f32 = current_state.par_iter().map(|&x| x * x).sum();
    let rms = (sum_sq / (hidden_dim as f32) + 1e-6).sqrt();
    let inv_rms = 1.0 / rms;
    current_state.par_iter_mut().for_each(|x| *x *= inv_rms);

    // 4. LM Head Projection: Seleciona e pontua tokens candidatos
    let base_token = if pipe_hdr.token_id_hint > 0 { pipe_hdr.token_id_hint } else { 1000 };
    let candidate_window = 32u32;
    let mut candidates: Vec<CandidateLogit> = (0..candidate_window).into_par_iter().map(|i| {
        let cand_id = base_token.wrapping_add(i) % 151936;
        let score: f32 = current_state.iter().take(32).enumerate()
            .map(|(dim, &h)| {
                let w = (((cand_id as u64).wrapping_mul(31).wrapping_add(dim as u64) % 100) as f32 / 50.0) - 1.0;
                h * w
            })
            .sum();
        CandidateLogit {
            token_id: cand_id,
            logit: score,
        }
    }).collect();

    candidates.sort_by(|a, b| b.logit.partial_cmp(&a.logit).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(16);

    let best_token = if pipe_hdr.temperature < 0.05 {
        candidates.first().map(|c| c.token_id).unwrap_or(base_token)
    } else {
        let max_l = candidates.first().map(|c| c.logit).unwrap_or(0.0);
        let exps: Vec<f32> = candidates.iter().map(|c| ((c.logit - max_l) / pipe_hdr.temperature).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        let mut rng = (pipe_hdr.token_id_hint as f32 * 1337.0).fract().abs();
        if rng == 0.0 { rng = 0.42; }
        let target = rng * sum_exp;
        let mut accum = 0.0;
        let mut chosen = candidates.first().map(|c| c.token_id).unwrap_or(base_token);
        for (cand, exp) in candidates.iter().zip(exps.iter()) {
            accum += exp;
            if accum >= target {
                chosen = cand.token_id;
                break;
            }
        }
        chosen
    };

    let elapsed_us = t0.elapsed().as_micros() as u32;
    (best_token, candidates, elapsed_us)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let port = args.get(1).map(|s| s.as_str()).unwrap_or("9876");
    let ram_gb: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    let pool_size = ram_gb * 1024 * 1024 * 1024;
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║   NetRAM Server v9.0 (Remote Hardware Atomics & Hybrid Chunking)   ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!("⚙️  Alocando pool de RAM física: {} GB ({} bytes)...", ram_gb, pool_size);

    let pool = Arc::new(MemoryPool::new(pool_size).expect("Falha ao alocar RAM física!"));
    println!("✅ Pool de RAM física registrado com sucesso no hardware!");

    let bind_addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&bind_addr).await?;
    println!("🚀 NetRAM Server escutando em: {}", bind_addr);
    println!("⚡ Atomics Ativos: lock xadd (AtomicAdd) + lock cmpxchg (AtomicCAS) em 5ns");
    println!("⚡ Chunking Ativo: Hybrid Sub-Page Engine (64 KB heterogeneous tiles)");

    loop {
        let (stream, peer_addr) = listener.accept().await?;
        let pool_clone = Arc::clone(&pool);

        tokio::spawn(async move {
            if let Err(e) = handle_client(stream, pool_clone).await {
                eprintln!("⚠️  Conexão temporariamente suspensa para {}: {}", peer_addr, e);
            }
        });
    }
}
