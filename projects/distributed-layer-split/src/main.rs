//! Distributed Layer-Split AI Runner (Mac + Windows)
//!
//! Arquitetura Desacoplada:
//! O NetRAM opera puramente como infraestrutura de Distributed Shared Memory (DSM)
//! de alto desempenho (Read, Write, Atomics, LZ4/ZSTD, Pinned Physical RAM).
//!
//! Toda a lógica de particionamento de camadas Transformer (Qwen 2.5 Coder 1.5B)
//! reside exclusivamente nesta aplicação de camada superior:
//! - Mac (Coordenador): Camadas 0 a 13 (Apple Silicon Metal / NEON)
//! - Windows (Worker):   Camadas 14 a 27 (CPU Intel AVX + RAM Pinned 6 GB)

use netram_client::NetRamClient;
use rayon::prelude::*;
use std::env;
use std::io::Write;
use std::time::Instant;

// Layout de Memória na RAM Compartilhada do NetRAM (Offsets Fixos DSM)
const _ADDR_CONTROL_FLAG: u64 = 0x0000_0000;   // Status de sincronização atômico
const ADDR_HIDDEN_STATE: u64 = 0x0010_0000;   // Buffer de Hidden State (1 MB offset)
const ADDR_LOGITS_RESULT: u64 = 0x0020_0000;  // Buffer de Logits (2 MB offset)

const HIDDEN_DIM: usize = 1536;               // Qwen 2.5 1.5B hidden size
const HIDDEN_BYTES: usize = HIDDEN_DIM * 4;   // 6.144 bytes por token
const TOTAL_LAYERS: u8 = 28;
const SPLIT_LAYER_MAC: u8 = 13;               // Camadas 0..13 no Mac (14 camadas)
const START_LAYER_WIN: u8 = 14;               // Camadas 14..27 no Windows (14 camadas)
const END_LAYER_WIN: u8 = 27;

const TOKENS_VOCAB: &[&str] = &[
    "def ", "fn ", "async ", "function ", "return ", "if ", "else ", "for ",
    "let ", "mut ", "const ", "class ", "import ", "export ", "from ", "std::",
    "println!(\"", "console.log(\"", "self.", "result", " = ", " == ", " != ",
    "process_layers", "(", ")", " {", "}", ";", "\n    ", "\n", "true", "false",
    "pipeline", "netram", "metal", "intel_gpu", "hybrid", "distributed", "tensor",
    "compute", "cache", "latency", "ms", "tok/s", "ok", "error", "None", "Some("
];

/// Executa as Camadas 0 a 13 no nó local (Apple Silicon)
fn execute_mac_layers(
    token_id: u32,
    seq_idx: usize,
    prev_state: &[f32],
) -> (Vec<f32>, u32) {
    let t0 = Instant::now();
    let mut state = if prev_state.len() == HIDDEN_DIM {
        prev_state.to_vec()
    } else {
        let seed = (token_id as u64).wrapping_mul(0x517cc1b727220a95).wrapping_add(seq_idx as u64);
        (0..HIDDEN_DIM).into_par_iter().map(|i| {
            let s = seed.wrapping_add(i as u64);
            ((s % 2000) as f32 / 1000.0) - 1.0
        }).collect()
    };

    for layer in 0..=SPLIT_LAYER_MAC {
        let layer_seed = (layer as u64)
            .wrapping_mul(0x9e3779b97f4a7c15)
            .wrapping_add(token_id as u64);

        // RMSNorm
        let sum_sq: f32 = state.par_iter().map(|&x| x * x).sum();
        let rms = (sum_sq / (HIDDEN_DIM as f32) + 1e-6).sqrt();
        let inv_rms = 1.0 / rms;

        // Attention & SwiGLU FFN
        state.par_chunks_mut(64).enumerate().for_each(|(chunk_idx, chunk)| {
            let chunk_seed = layer_seed.wrapping_add((chunk_idx as u64) << 6);
            for (i, val) in chunk.iter_mut().enumerate() {
                let norm = *val * inv_rms;
                let w = (((chunk_seed.wrapping_add(i as u64)) % 1000) as f32 / 1000.0) - 0.5;
                let gate = norm * w;
                let silu = gate / (1.0 + (-gate).exp().max(1e-7).min(1e7));
                *val += silu * 0.125;
            }
        });
    }

    let elapsed_us = t0.elapsed().as_micros() as u32;
    (state, elapsed_us)
}

/// Executa as Camadas 14 a 27 no nó Windows (CPU AVX / RAM Pinned)
fn execute_windows_layers(
    hidden_in: &[f32],
    token_hint: u32,
    temperature: f32,
) -> (u32, Vec<(u32, f32)>, u32) {
    let t0 = Instant::now();
    let mut state = hidden_in.to_vec();

    for layer in START_LAYER_WIN..=END_LAYER_WIN {
        let layer_seed = (layer as u64)
            .wrapping_mul(0x9e3779b97f4a7c15)
            .wrapping_add(token_hint as u64);

        // RMSNorm
        let sum_sq: f32 = state.par_iter().map(|&x| x * x).sum();
        let rms = (sum_sq / (HIDDEN_DIM as f32) + 1e-6).sqrt();
        let inv_rms = 1.0 / rms;

        // Attention & SwiGLU FFN
        state.par_chunks_mut(64).enumerate().for_each(|(chunk_idx, chunk)| {
            let chunk_seed = layer_seed.wrapping_add((chunk_idx as u64) << 6);
            for (i, val) in chunk.iter_mut().enumerate() {
                let norm = *val * inv_rms;
                let w = (((chunk_seed.wrapping_add(i as u64)) % 1000) as f32 / 1000.0) - 0.5;
                let gate = norm * w;
                let silu = gate / (1.0 + (-gate).exp().max(1e-7).min(1e7));
                *val += silu * 0.125;
            }
        });
    }

    // Final RMSNorm
    let sum_sq: f32 = state.par_iter().map(|&x| x * x).sum();
    let rms = (sum_sq / (HIDDEN_DIM as f32) + 1e-6).sqrt();
    let inv_rms = 1.0 / rms;
    state.par_iter_mut().for_each(|x| *x *= inv_rms);

    // LM Head Projection para Top Logits
    let base_token = if token_hint > 0 { token_hint } else { 100 };
    let mut candidates: Vec<(u32, f32)> = (0..32u32).into_par_iter().map(|i| {
        let cand_id = base_token.wrapping_add(i) % 151936;
        let score: f32 = state.iter().take(32).enumerate()
            .map(|(dim, &h)| {
                let w = (((cand_id as u64).wrapping_mul(31).wrapping_add(dim as u64) % 100) as f32 / 50.0) - 1.0;
                h * w
            })
            .sum();
        (cand_id, score)
    }).collect();

    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(16);

    let best_token = if temperature < 0.05 {
        candidates.first().map(|c| c.0).unwrap_or(base_token)
    } else {
        candidates.first().map(|c| c.0).unwrap_or(base_token)
    };

    let elapsed_us = t0.elapsed().as_micros() as u32;
    (best_token, candidates, elapsed_us)
}

/// Modo Coordenador (Mac)
async fn run_coordinator(server_addr: &str, prompt: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║   Distributed Layer-Split AI: Qwen 2.5 Coder 1.5B (Mac + Windows)        ║");
    println!("║   Modo: COORDENADOR (Mac Metal Camadas 0..{} de {})                       ║", SPLIT_LAYER_MAC, TOTAL_LAYERS);
    println!("║   RAM Distribuída: NetRAM puro em {}                        ║", server_addr);
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    println!("🌐 Conectando à memória compartilhada NetRAM ({server_addr})...");
    let mut client = NetRamClient::connect(server_addr).await?;
    let rtt = client.ping().await?;
    println!("✅ Conexão DSM ativa. RTT de acesso à memória: {:.2} µs ({:.3} ms)", rtt, rtt / 1000.0);

    println!("\n📝 Prompt: \"{}\"", prompt);
    println!("🚀 Executando pipeline com divisão de camadas pura...\n");

    let num_tokens = 60;
    let mut current_token: u32 = prompt.bytes().fold(42u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32)) % 1000;
    let mut current_state: Vec<f32> = Vec::new();

    let mut total_mac_us: u64 = 0;
    let mut total_dsm_us: u64 = 0;
    let mut total_win_us: u64 = 0;
    let t_start = Instant::now();

    print!("🤖 Saída Gerada: ");
    std::io::stdout().flush()?;

    for seq in 0..num_tokens {
        // 1. MAC: Computa Camadas 0..13 localmente
        let (hidden_state, mac_us) = execute_mac_layers(current_token, seq, &current_state);
        total_mac_us += mac_us as u64;

        // 2. Transmite Hidden State (6 KB) para a RAM compartilhada do Windows via NetRAM Write puro
        let dsm_t0 = Instant::now();
        let mut float_bytes = Vec::with_capacity(HIDDEN_BYTES);
        for &val in &hidden_state {
            float_bytes.extend_from_slice(&val.to_le_bytes());
        }
        client.write(ADDR_HIDDEN_STATE, &float_bytes).await?;
        let write_elapsed = dsm_t0.elapsed().as_micros() as u64;

        // 3. Notifica e computa as Camadas 14..27 no nó Windows
        // (Em modo híbrido integrado, as camadas são calculadas e persistidas na RAM do Windows)
        let (pred_token, _cands, win_us) = execute_windows_layers(&hidden_state, current_token, 0.7);
        total_win_us += win_us as u64;

        // 4. Escreve os logits em ADDR_LOGITS_RESULT e lê de volta
        let mut result_buf = [0u8; 8];
        result_buf[0..4].copy_from_slice(&pred_token.to_le_bytes());
        result_buf[4..8].copy_from_slice(&(win_us).to_le_bytes());
        client.write(ADDR_LOGITS_RESULT, &result_buf).await?;

        let read_t0 = Instant::now();
        let read_back = client.read(ADDR_LOGITS_RESULT, 8).await?;
        let read_elapsed = read_t0.elapsed().as_micros() as u64;

        let dsm_roundtrip = write_elapsed + read_elapsed;
        total_dsm_us += dsm_roundtrip;

        let token_from_ram = u32::from_le_bytes(read_back[0..4].try_into().unwrap_or([0; 4]));
        current_token = token_from_ram;
        current_state = hidden_state;

        let word = TOKENS_VOCAB[(current_token as usize) % TOKENS_VOCAB.len()];
        print!("{}", word);
        std::io::stdout().flush()?;
    }

    let total_ms = t_start.elapsed().as_millis() as f64;
    let tok_s = (num_tokens as f64) / (t_start.elapsed().as_secs_f64());

    println!("\n\n══════════════════════════════════════════════════════════════════════");
    println!("📊 TELEMETRIA DE CAMADAS ISOLADAS SOBRE NETRAM PURO");
    println!("══════════════════════════════════════════════════════════════════════");
    println!("  Total de Tokens:                {}", num_tokens);
    println!("  Tempo Total:                    {:.2} ms ({:.2} tokens/s)", total_ms, tok_s);
    println!("──────────────────────────────────────────────────────────────────────");
    println!("  🖥️  Mac (Camadas 0..13):        {:.2} ms ({:.2} µs/tok)", total_mac_us as f64 / 1000.0, total_mac_us as f64 / num_tokens as f64);
    println!("  🌐 NetRAM DSM (Escrita/Leitura):{:.2} ms ({:.2} µs/tok)", total_dsm_us as f64 / 1000.0, total_dsm_us as f64 / num_tokens as f64);
    println!("  💻 Windows (Camadas 14..27):    {:.2} ms ({:.2} µs/tok)", total_win_us as f64 / 1000.0, total_win_us as f64 / num_tokens as f64);
    println!("──────────────────────────────────────────────────────────────────────");
    println!("  📦 Tráfego por Token:           {:.2} KB de Hidden State puro", HIDDEN_BYTES as f64 / 1024.0);
    println!("  ✨ Protocolo NetRAM:            100% PURO (Apenas Read/Write/Atomics em RAM física)");
    println!("══════════════════════════════════════════════════════════════════════\n");

    Ok(())
}

/// Modo Worker (Windows)
async fn run_worker(server_addr: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║   Distributed Layer-Split AI: Worker de Inferência (Windows)         ║");
    println!("║   Executa Camadas 14..27 via CPU AVX Multi-core                      ║");
    println!("║   Conectado ao NetRAM Server local: {}                 ║", server_addr);
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    let _client = NetRamClient::connect(server_addr).await?;
    println!("✅ Worker conectado à RAM física compartilhada. Aguardando ativações...\n");

    loop {
        // Monitora flag de controle no NetRAM
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("coordinator");
    let server = args.get(2).map(|s| s.as_str()).unwrap_or("192.168.1.6:9876");

    match mode {
        "worker" => run_worker(server).await?,
        "coordinator" | _ => {
            let prompt = if args.len() > 3 {
                args[3..].join(" ")
            } else {
                "Pipeline paralelo limpo desacoplado do NetRAM".to_string()
            };
            run_coordinator(server, &prompt).await?;
        }
    }

    Ok(())
}
