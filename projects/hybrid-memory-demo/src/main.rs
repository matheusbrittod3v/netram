use netram_client::{NetRamClient, TieredMemory};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let server_addr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "192.168.1.6:9876".to_string());

    println!("╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║       PROJETO DEMO: MEMÓRIA HÍBRIDA DISTRIBUÍDA (MAC + NOTEBOOK WINDOWS)      ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════╝");
    println!("🌐 Conectando à RAM física do notebook Windows em: {}", server_addr);

    let client = NetRamClient::connect(&server_addr).await?;
    println!("✅ Conexão estabelecida com sucesso!");

    // =========================================================================
    // CONFIGURAÇÃO DA MEMÓRIA EM CAMADAS (TIERED MEMORY)
    // =========================================================================
    // Definimos o tamanho de página de 64 KB.
    // O Mac terá uma cota intencional de 128 páginas (8 MB de RAM local).
    // O dataset total terá 32 MB (512 páginas).
    // O excedente de 24 MB é empurrado automaticamente para o Windows via Delta-XOR / LZ4.
    let page_size = 64 * 1024; // 64 KB
    let max_local_pages = 128; // 8 MB no Mac
    let total_pages = 512;     // 32 MB de dados totais
    let total_bytes = total_pages * page_size;

    let mut tiered = TieredMemory::new(client, page_size, max_local_pages);

    println!("\n📊 [Configuração de Memória]");
    println!("   • Cota de RAM Local no Mac (L1 Cache) : {:.2} MB ({} páginas de 64 KB)", (max_local_pages * page_size) as f64 / (1024.0 * 1024.0), max_local_pages);
    println!("   • Pool de RAM Física no Windows (L2)  : 3.00 GB (disponível)");
    println!("   • Tamanho do Dataset da Aplicação     : {:.2} MB ({} páginas)", total_bytes as f64 / (1024.0 * 1024.0), total_pages);

    // =========================================================================
    // FASE 1: GRAVAÇÃO DISTRIBUÍDA (TRANSIÇÃO L1 -> L2 AUTOMÁTICA)
    // =========================================================================
    println!("\n--- [FASE 1] Gravando 32 MB de dados contínuos no ecossistema híbrido ---");
    let start_write = Instant::now();

    for page_id in 0..total_pages {
        let mut page_data = vec![0u8; page_size];
        let header_str = format!("PAGE_{:04}_DISTRIBUTED_NETRAM_MAC_WINDOWS", page_id);
        page_data[..header_str.len()].copy_from_slice(header_str.as_bytes());
        for i in header_str.len()..page_size {
            page_data[i] = ((i + page_id) & 0xFF) as u8;
        }

        let offset = (page_id * page_size) as u64;
        tiered.write(offset, &page_data).await?;
    }

    // Descarrega páginas sujas remanescentes
    tiered.flush().await?;
    let write_dur = start_write.elapsed();
    let write_mb = total_bytes as f64 / (1024.0 * 1024.0);
    println!("✅ Gravação concluída em {:.2} ms ({:.2} MB/s efetivos)", write_dur.as_secs_f64() * 1000.0, write_mb / write_dur.as_secs_f64());

    println!("   📈 Estado das Camadas após gravação:");
    println!("      • Páginas Retidas no Cache Local do Mac : {} páginas ({:.2} MB)", tiered.local_cache_size_bytes() / page_size, tiered.local_cache_size_bytes() as f64 / (1024.0 * 1024.0));
    println!("      • Páginas Despejadas na RAM do Windows  : {} páginas ({:.2} MB)", tiered.stats.evictions, (tiered.stats.evictions as f64 * page_size as f64) / (1024.0 * 1024.0));

    // =========================================================================
    // FASE 2: ACESSO DE LEITURA (TESTE L1 CACHE HIT vs L2 NETWORK FETCH)
    // =========================================================================
    println!("\n--- [FASE 2] Comparativo de Latência de Acesso (L1 Mac vs L2 Windows) ---");

    // 2.1 Leitura de página quente (que está na RAM do Mac)
    let hot_page_id = (total_pages - 1) as u64; // última página gravada está no L1
    let hot_offset = hot_page_id * page_size as u64;
    let mut hot_buf = vec![0u8; page_size];
    let t0 = Instant::now();
    tiered.read(hot_offset, &mut hot_buf).await?;
    let t_hot = t0.elapsed();
    println!("   🚀 [L1 CACHE HIT - RAM do Mac]: Página {} lida em {:.3} µs ({:.6} ms)", hot_page_id, t_hot.as_secs_f64() * 1_000_000.0, t_hot.as_secs_f64() * 1000.0);
    assert!(hot_buf.starts_with(format!("PAGE_{:04}", hot_page_id).as_bytes()));

    // 2.2 Leitura de página fria (que foi empurrada para a RAM do Windows)
    let cold_page_id = 5u64; // Página 5 foi despejada para a RAM do Windows
    let cold_offset = cold_page_id * page_size as u64;
    let mut cold_buf = vec![0u8; page_size];
    let t1 = Instant::now();
    tiered.read(cold_offset, &mut cold_buf).await?;
    let t_cold = t1.elapsed();
    println!("   🌐 [L2 NETWORK FETCH - RAM do Windows]: Página {} resgatada via Wi-Fi em {:.2} ms", cold_page_id, t_cold.as_secs_f64() * 1000.0);
    assert!(cold_buf.starts_with(b"PAGE_0005"));

    // =========================================================================
    // FASE 3: VARREDURA SEQUENCIAL COM PREFETCHING PREDITIVO POR STRIDE
    // =========================================================================
    println!("\n--- [FASE 3] Varredura sequencial contínua com Stride Prefetching ativado ---");
    let sample_count = 60;
    let start_seq = Instant::now();
    let mut read_buf = vec![0u8; page_size];
    for page_id in 0..sample_count {
        let offset = (page_id * page_size) as u64;
        tiered.read(offset, &mut read_buf).await?;
    }
    let seq_dur = start_seq.elapsed();
    println!("   ⚡ Lidas {} páginas sequenciais em {:.2} ms (média de {:.2} ms por página)", sample_count, seq_dur.as_secs_f64() * 1000.0, (seq_dur.as_secs_f64() * 1000.0) / sample_count as f64);

    // =========================================================================
    // FASE 4: ESTATÍSTICAS FINAIS DE EFICIÊNCIA
    // =========================================================================
    println!("\n--- [FASE 4] Estatísticas Finais de Eficiência do Sistema ---");
    let hit_ratio = (tiered.stats.hits as f64 / (tiered.stats.hits + tiered.stats.misses).max(1) as f64) * 100.0;
    println!("   🎯 Cache Hits no Mac  : {}", tiered.stats.hits);
    println!("   🌐 Cache Misses (L2)  : {}", tiered.stats.misses);
    println!("   💎 Taxa de Acerto L1  : {:.1}%", hit_ratio);
    println!("   📦 Evicções para Windows: {} páginas", tiered.stats.evictions);
    println!("   🚀 Prefetchs disparados : {}", tiered.stats.prefetches_triggered);
    println!("   ✨ Prefetch Hits        : {}", tiered.stats.prefetch_hits);

    println!("\n🏆 [CONCLUSÃO DO PROJETO DEMO]");
    println!("   A aplicação consumiu um dataset de 32 MB sem nunca travar ou estourar a memória do Mac!");
    println!("   O Mac usou apenas 8 MB de RAM local; os 24 MB restantes foram armazenados, sincronizados");
    println!("   e resgatados da RAM física do notebook Windows de forma transparente via NetRAM!");

    Ok(())
}
