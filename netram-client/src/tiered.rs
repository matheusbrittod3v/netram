use crate::NetRamClient;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

pub struct CachePage {
    pub page_id: u64,
    pub data: Vec<u8>,
    pub clean_baseline: Option<Vec<u8>>,
    pub is_dirty: bool,
    pub last_accessed: Instant,
    pub is_prefetched: bool,
}

#[derive(Debug, Default, Clone)]
pub struct TieredStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub delta_evictions: u64,
    pub prefetches_triggered: u64,
    pub prefetch_hits: u64, // Acessos que seriam Cache Miss mas foram salvos pelo Prefetching!
}

/// Gerenciador de Memória em Camadas com Prefetching Preditivo por Stride
pub struct TieredMemory {
    client: Arc<Mutex<NetRamClient>>,
    page_size: usize,
    max_local_pages: usize,
    pages: HashMap<u64, CachePage>,
    pub stats: TieredStats,

    // Motor de Predição por Stride
    last_accessed_page: Option<u64>,
    detected_stride: Option<i64>,
    prefetch_depth: usize,
}

impl TieredMemory {
    pub fn new(client: NetRamClient, page_size: usize, max_local_pages: usize) -> Self {
        Self {
            client: Arc::new(Mutex::new(client)),
            page_size,
            max_local_pages,
            pages: HashMap::new(),
            stats: TieredStats::default(),
            last_accessed_page: None,
            detected_stride: None,
            prefetch_depth: 2, // Antecipa 2 páginas à frente
        }
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn local_cache_size_bytes(&self) -> usize {
        self.pages.len() * self.page_size
    }

    pub fn set_prefetch_depth(&mut self, depth: usize) {
        self.prefetch_depth = depth;
    }

    /// Expulsa a página LRU (menos recentemente acessada) para o Windows
    pub async fn evict_lru(&mut self) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        if self.pages.is_empty() {
            return Ok(0);
        }

        let lru_page_id = self
            .pages
            .iter()
            .min_by_key(|(_, page)| page.last_accessed)
            .map(|(id, _)| *id)
            .unwrap();

        let page = self.pages.remove(&lru_page_id).unwrap();

        if page.is_dirty {
            let offset = lru_page_id * self.page_size as u64;
            let mut client = self.client.lock().await;

            if let Some(ref baseline) = page.clean_baseline {
                let (bytes_sent, _) = client.write_delta_xor(offset, baseline, &page.data).await?;
                self.stats.delta_evictions += 1;
                let _ = bytes_sent;
            } else {
                client.write_adaptive(offset, &page.data).await?;
            }
        }

        self.stats.evictions += 1;
        Ok(lru_page_id)
    }

    /// Garante que uma página esteja no Cache Local do Mac
    async fn ensure_page_loaded(&mut self, page_id: u64) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.pages.contains_key(&page_id) {
            let page = self.pages.get_mut(&page_id).unwrap();
            page.last_accessed = Instant::now();

            if page.is_prefetched {
                self.stats.prefetch_hits += 1;
                page.is_prefetched = false; // Promovida a página ativa normal
            }

            self.stats.hits += 1;
            return Ok(());
        }

        self.stats.misses += 1;

        if self.pages.len() >= self.max_local_pages {
            self.evict_lru().await?;
        }

        let offset = page_id * self.page_size as u64;
        let remote_data = {
            let mut client = self.client.lock().await;
            match client.read_lz4(offset, self.page_size).await {
                Ok(data) => data,
                Err(_) => vec![0u8; self.page_size],
            }
        };

        let clean_copy = remote_data.clone();

        self.pages.insert(
            page_id,
            CachePage {
                page_id,
                data: remote_data,
                clean_baseline: Some(clean_copy),
                is_dirty: false,
                last_accessed: Instant::now(),
                is_prefetched: false,
            },
        );

        Ok(())
    }

    /// Executa o Prefetch Preditivo em Background para as próximas páginas
    async fn trigger_predictive_prefetch(&mut self, next_page_id: u64) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.pages.contains_key(&next_page_id) {
            return Ok(());
        }

        // Se o cache tiver espaço ou se couber mais uma página antecipada
        if self.pages.len() < self.max_local_pages {
            let offset = next_page_id * self.page_size as u64;
            let remote_data = {
                let mut client = self.client.lock().await;
                match client.read_lz4(offset, self.page_size).await {
                    Ok(data) => data,
                    Err(_) => vec![0u8; self.page_size],
                }
            };

            let clean_copy = remote_data.clone();
            self.pages.insert(
                next_page_id,
                CachePage {
                    page_id: next_page_id,
                    data: remote_data,
                    clean_baseline: Some(clean_copy),
                    is_dirty: false,
                    last_accessed: Instant::now(),
                    is_prefetched: true,
                },
            );

            self.stats.prefetches_triggered += 1;
        }

        Ok(())
    }

    /// Atualiza o detector de Stride e antecipa o carregamento preditivo
    async fn update_stride_and_prefetch(&mut self, current_page: u64) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(last_page) = self.last_accessed_page {
            let stride = (current_page as i64) - (last_page as i64);

            if stride != 0 {
                self.detected_stride = Some(stride);

                // Antecipa as próximas páginas preditas pelo stride
                for step in 1..=self.prefetch_depth {
                    let predicted_page = (current_page as i64) + stride * (step as i64);
                    if predicted_page >= 0 {
                        self.trigger_predictive_prefetch(predicted_page as u64).await?;
                    }
                }
            }
        }

        self.last_accessed_page = Some(current_page);
        Ok(())
    }

    /// Escrita com Cache LRU Inteligente
    pub async fn write(&mut self, offset: u64, data: &[u8]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut written = 0;

        while written < data.len() {
            let current_offset = offset + written as u64;
            let page_id = current_offset / self.page_size as u64;
            let page_offset = (current_offset % self.page_size as u64) as usize;
            let chunk_len = (self.page_size - page_offset).min(data.len() - written);

            self.ensure_page_loaded(page_id).await?;

            let page = self.pages.get_mut(&page_id).unwrap();
            page.data[page_offset..page_offset + chunk_len].copy_from_slice(&data[written..written + chunk_len]);
            page.is_dirty = true;
            page.last_accessed = Instant::now();

            self.update_stride_and_prefetch(page_id).await?;
            written += chunk_len;
        }

        Ok(())
    }

    /// Leitura com Prefetching Preditivo por Stride (Oculta os 8ms de latência de rede!)
    pub async fn read(&mut self, offset: u64, dest: &mut [u8]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut read_bytes = 0;

        while read_bytes < dest.len() {
            let current_offset = offset + read_bytes as u64;
            let page_id = current_offset / self.page_size as u64;
            let page_offset = (current_offset % self.page_size as u64) as usize;
            let chunk_len = (self.page_size - page_offset).min(dest.len() - read_bytes);

            self.ensure_page_loaded(page_id).await?;

            let page = self.pages.get_mut(&page_id).unwrap();
            dest[read_bytes..read_bytes + chunk_len].copy_from_slice(&page.data[page_offset..page_offset + chunk_len]);
            page.last_accessed = Instant::now();

            self.update_stride_and_prefetch(page_id).await?;
            read_bytes += chunk_len;
        }

        Ok(())
    }

    /// Descarrega todas as páginas dirty modificadas para o Windows
    pub async fn flush(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut client = self.client.lock().await;

        for (page_id, page) in self.pages.iter_mut() {
            if page.is_dirty {
                let offset = *page_id * self.page_size as u64;
                if let Some(ref baseline) = page.clean_baseline {
                    client.write_delta_xor(offset, baseline, &page.data).await?;
                } else {
                    client.write_adaptive(offset, &page.data).await?;
                }
                page.clean_baseline = Some(page.data.clone());
                page.is_dirty = false;
            }
        }
        Ok(())
    }
}
