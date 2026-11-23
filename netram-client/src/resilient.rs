use crate::NetRamClient;
use netram_proto::{FLAG_SESSION_RESUME, Header, OpCode, Status, HEADER_SIZE};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub struct QueuedWrite {
    pub offset: u64,
    pub data: Vec<u8>,
    pub timestamp: Instant,
}

pub struct ResilientNetRamClient {
    target_addr: String,
    client: Option<NetRamClient>,
    write_journal: Vec<QueuedWrite>,
    pub reconnect_count: u64,
    pub recovered_writes: u64,
}

impl ResilientNetRamClient {
    pub async fn connect(target_addr: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let client = NetRamClient::connect(target_addr).await?;
        Ok(Self {
            target_addr: target_addr.to_string(),
            client: Some(client),
            write_journal: Vec::new(),
            reconnect_count: 0,
            recovered_writes: 0,
        })
    }

    pub async fn ensure_connected(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.client.is_some() {
            return Ok(());
        }

        let mut delay = Duration::from_millis(100);
        let max_delay = Duration::from_secs(2);
        let start = Instant::now();
        let timeout = Duration::from_secs(30);

        println!("⚠️  [RESILIÊNCIA] Conexão com Windows perdida. Tentando reconectar em segundo plano...");

        while start.elapsed() < timeout {
            tokio::time::sleep(delay).await;

            match NetRamClient::connect(&self.target_addr).await {
                Ok(mut new_client) => {
                    let resume_req = Header::new(OpCode::Ping, FLAG_SESSION_RESUME, 0, 0, 0, 0);
                    if let Ok(_) = new_client.stream.write_all(&resume_req.encode()).await {
                        let mut resp_buf = [0u8; HEADER_SIZE];
                        let _ = new_client.stream.read_exact(&mut resp_buf).await;
                    }

                    println!("✅ [RESILIÊNCIA] Notebook Windows reconectado após {:.2}s!", start.elapsed().as_secs_f64());
                    self.reconnect_count += 1;
                    self.client = Some(new_client);

                    self.drain_journal().await?;
                    return Ok(());
                }
                Err(_) => {
                    delay = (delay * 2).min(max_delay);
                }
            }
        }

        Err("Timeout de reconexão de 30s excedido".into())
    }

    async fn drain_journal(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.write_journal.is_empty() {
            return Ok(());
        }

        println!("🔄 [JOURNAL] Descarregando {} escritas pendentes acumuladas...", self.write_journal.len());

        let pending = std::mem::take(&mut self.write_journal);
        let client = self.client.as_mut().unwrap();

        for item in pending {
            client.write_adaptive(item.offset, &item.data).await?;
            self.recovered_writes += 1;
        }

        println!("🎉 [JOURNAL] Todas as escritas pendentes foram sincronizadas com sucesso!");
        Ok(())
    }

    pub async fn heartbeat(&mut self) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        self.ensure_connected().await?;

        let req = Header::new(OpCode::Heartbeat, 0, 0, 0, 0, 0);
        let start = Instant::now();
        let client = self.client.as_mut().unwrap();

        client.stream.write_all(&req.encode()).await?;
        let mut resp_buf = [0u8; HEADER_SIZE];
        client.stream.read_exact(&mut resp_buf).await?;

        let elapsed_us = start.elapsed().as_nanos() as f64 / 1_000.0;
        let resp = Header::decode(&resp_buf)?;

        if resp.status_or_flags != Status::Ok as u8 {
            return Err("Heartbeat falhou".into());
        }

        Ok(elapsed_us)
    }

    pub async fn write_resilient(&mut self, offset: u64, data: &[u8]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(client) = self.client.as_mut() {
            if let Ok(_) = client.write_adaptive(offset, data).await {
                return Ok(());
            }
        }

        println!("⚠️  [JOURNAL] Queda detectada no offset {}. Salvando no Journal Local...", offset);
        self.write_journal.push(QueuedWrite {
            offset,
            data: data.to_vec(),
            timestamp: Instant::now(),
        });
        self.client = None;

        self.ensure_connected().await?;
        Ok(())
    }

    pub async fn read_resilient(&mut self, offset: u64, dest_len: usize) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        self.ensure_connected().await?;
        let client = self.client.as_mut().unwrap();

        match client.read_adaptive(offset, dest_len).await {
            Ok(data) => Ok(data),
            Err(_) => {
                self.client = None;
                self.ensure_connected().await?;
                let client = self.client.as_mut().unwrap();
                client.read_adaptive(offset, dest_len).await
            }
        }
    }
}
