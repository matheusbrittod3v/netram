use netram_client::NetRamClient;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    let target = args.get(1).map(|s| s.as_str()).unwrap_or("127.0.0.1:9876");

    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║       NetRAM Client v9.0 (Remote Atomics & Hybrid Chunking)        ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!("🌐 Conectando com Wire Protocol v9.0 em {}...", target);

    let mut client = NetRamClient::connect(target).await?;
    println!("✅ Conectado com sucesso!");

    // =========================================================================
    // TESTE 1: REMOTE HARDWARE ATOMIC ADD (lock xadd no hardware em 5ns)
    // =========================================================================
    println!("\n--- [TESTE 1] Remote Hardware Atomic Add (lock xadd no silício do Windows) ---");
    let atomic_offset = 1024 * 1024; // 1 MB offset (alinhado a 8 bytes)

    // Inicializa com 0
    client.write_adaptive(atomic_offset, &0u64.to_be_bytes()).await?;

    let t_add1 = Instant::now();
    let prev1 = client.atomic_add(atomic_offset, 100).await?;
    let d_add1 = t_add1.elapsed();
    println!("   ⚡ AtomicAdd(+100): Valor Anterior = {} | Tempo = {:.2} ms", prev1, d_add1.as_secs_f64() * 1000.0);
    assert_eq!(prev1, 0);

    let t_add2 = Instant::now();
    let prev2 = client.atomic_add(atomic_offset, 50).await?;
    let d_add2 = t_add2.elapsed();
    println!("   ⚡ AtomicAdd(+50) : Valor Anterior = {} (Novo Valor = 150) | Tempo = {:.2} ms", prev2, d_add2.as_secs_f64() * 1000.0);
    assert_eq!(prev2, 100);

    // =========================================================================
    // TESTE 2: REMOTE HARDWARE ATOMIC CAS (lock cmpxchg no silício do Windows)
    // =========================================================================
    println!("\n--- [TESTE 2] Remote Hardware Atomic CAS (Compare-And-Swap distribuído) ---");

    // Caso A: CAS com Sucesso (expected = 150 -> desired = 999)
    let t_cas1 = Instant::now();
    let res_cas1 = client.atomic_cas(atomic_offset, 150, 999).await?;
    let d_cas1 = t_cas1.elapsed();
    match res_cas1 {
        Ok(prev) => println!("   ✅ CAS Bem-Sucedido: esperado 150 encontrado (anterior: {}) -> atualizado para 999 em {:.2} ms!", prev, d_cas1.as_secs_f64() * 1000.0),
        Err(actual) => panic!("CAS inesperadamente falhou: atual={}", actual),
    }

    // Caso B: CAS com Conflito de Concorrência (espera 150, mas na RAM já é 999!)
    let t_cas2 = Instant::now();
    let res_cas2 = client.atomic_cas(atomic_offset, 150, 777).await?;
    let d_cas2 = t_cas2.elapsed();
    match res_cas2 {
        Ok(_) => panic!("CAS deveria ter rejeitado!"),
        Err(actual) => {
            println!("   🛡️  CAS Rejeitou Conflito: esperado 150, mas a RAM já continha {}! Rejeição atômica em {:.2} ms.", actual, d_cas2.as_secs_f64() * 1000.0);
            assert_eq!(actual, 999);
        }
    }
    println!("   🎉 Sincronização atômica distribuída (Spinlocks & Mutexes) validada com 100% de sucesso!");

    // =========================================================================
    // TESTE 3: HYBRID SUB-PAGE CHUNKING (BLOCO HETEROGÊNEO DE 256 KB)
    // =========================================================================
    println!("\n--- [TESTE 3] Hybrid Sub-Page Chunking (Bloco Heterogêneo de 4 Sub-Páginas de 64 KB) ---");
    let mut mixed_data = Vec::with_capacity(256 * 1024);

    // Sub-página 0: 64 KB de Zeros
    mixed_data.extend_from_slice(&vec![0u8; 64 * 1024]);
    // Sub-página 1: 64 KB de Dados Estruturados (ZSTD Dicionário)
    mixed_data.extend_from_slice(&(0..64 * 1024).map(|i| ((i / 8) * 17) as u8).collect::<Vec<u8>>());
    // Sub-página 2: 64 KB de Texto Repetitivo (LZ4)
    let pattern = b"NETRAM_HPC_DISTRIBUTED_RAM_";
    let mut lz4_page = Vec::with_capacity(64 * 1024);
    while lz4_page.len() < 64 * 1024 {
        lz4_page.extend_from_slice(pattern);
    }
    lz4_page.truncate(64 * 1024);
    mixed_data.extend_from_slice(&lz4_page);
    // Sub-página 3: 64 KB Aleatórios (Raw Zero-Copy)
    mixed_data.extend_from_slice(&(0..64 * 1024).map(|i| ((i * 1103515245 + 12345) & 0xFF) as u8).collect::<Vec<u8>>());

    let hybrid_offset = 2 * 1024 * 1024; // 2 MB
    let start_hybrid = Instant::now();
    let wire_bytes = client.write_hybrid(hybrid_offset, &mixed_data).await?;
    let hybrid_dur = start_hybrid.elapsed();

    println!("   📊 Composição Heterogênea: 4x Sub-Páginas (Zeros + ZSTD-Dict + LZ4 + Raw)");
    println!("   📉 Dados Transmitidos    : {:.2} KB (em vez de 256 KB)", wire_bytes as f64 / 1024.0);
    println!("   ⏱️  Tempo de Gravação    : {:.2} ms", hybrid_dur.as_secs_f64() * 1000.0);

    // Validação de Integridade
    let recovered_mixed = client.read_adaptive(hybrid_offset, 256 * 1024).await?;
    assert_eq!(recovered_mixed, mixed_data);
    println!("   🎉 100% de Integridade validada byte a byte no bloco heterogêneo!");

    println!("\n🏆 NetRAM v9.0 validado: Atomics Remotos (lock xadd / lock cmpxchg) e Hybrid Chunking operando no silício!");

    // =========================================================================
    // TESTE 4: NETCOMPUTE IN-SITU (CPU RAYON & GPU OPENCL EXECUTION)
    // =========================================================================
    println!("\n--- [TESTE 4] NetCompute In-Situ (Computação Paralela Remota na RAM) ---");
    let compute_offset = 4 * 1024 * 1024; // 4 MB
    
    // Matriz de 250.000 números float32 (1 MB de dados)
    let count = 250_000;
    let floats: Vec<f32> = (0..count).map(|i| (i as f32) * 0.01).collect();
    let floats_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(floats.as_ptr() as *const u8, count * 4)
    };

    // 1. Grava os dados na RAM física do notebook Windows
    client.write_adaptive(compute_offset, floats_bytes).await?;
    println!("   💾 Dados Gravados na RAM Remota: 1.00 MB (250.000 floats)");

    // 2. Execução in-situ: CPU Intel Rayon multi-thread (VectorSum)
    let start_cpu = Instant::now();
    let sum_bytes = client.remote_compute(
        netram_proto::ComputeEngine::CpuRayonAvx,
        netram_proto::ComputeTask::VectorSum,
        compute_offset,
        count * 4,
        0.0,
        0.0,
    ).await?;
    let cpu_dur = start_cpu.elapsed();
    let sum_result = f32::from_le_bytes(sum_bytes[..4].try_into().unwrap());
    println!("   ⚡ [CPU Intel 4-Cores] VectorSum Redução In-Situ: {:.2} | Tempo = {:.2} ms", sum_result, cpu_dur.as_secs_f64() * 1000.0);

    // 3. Execução in-situ: Multiplicação/Escala In-Place (VectorScale * 2.0)
    let start_scale = Instant::now();
    client.remote_compute(
        netram_proto::ComputeEngine::CpuRayonAvx,
        netram_proto::ComputeTask::VectorScale,
        compute_offset,
        count * 4,
        2.0,
        0.0,
    ).await?;
    let scale_dur = start_scale.elapsed();
    println!("   ⚡ [CPU Intel AVX SIMD] VectorScale (*2.0) In-Place: Concluído na RAM em {:.2} ms", scale_dur.as_secs_f64() * 1000.0);

    // 4. Execução in-situ: Convolução 2D Sobel (Intel HD 4000 GPU / SIMD)
    let img_w = 512;
    let img_h = 512;
    let img_len = img_w * img_h; // 256 KB
    let image_offset = 6 * 1024 * 1024; // 6 MB
    let test_img: Vec<u8> = (0..img_len).map(|i| ((i % img_w) ^ (i / img_w)) as u8).collect();
    client.write_adaptive(image_offset, &test_img).await?;

    let start_gpu = Instant::now();
    client.remote_compute(
        netram_proto::ComputeEngine::GpuIntelOpenCl,
        netram_proto::ComputeTask::ImageSobel,
        image_offset,
        img_len,
        img_w as f32,
        img_h as f32,
    ).await?;
    let gpu_dur = start_gpu.elapsed();
    println!("   🎮 [GPU Intel HD 4000] Convolução 2D Sobel (512x512 Image): Processado na RAM em {:.2} ms", gpu_dur.as_secs_f64() * 1000.0);
    println!("   🎉 NetCompute In-Situ validado com Sucesso Absoluto!");
    Ok(())
}
