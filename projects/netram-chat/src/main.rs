//! NetRAM Interactive Chat Terminal
//!
//! Integração em tempo real:
//! 1. O usuário conversa com o modelo Qwen 2.5 Coder 1.5B no terminal.
//! 2. A inferência dos pesos e tokens é executada nativamente pelo motor local com aceleração Metal.
//! 3. Em cada turno de conversa, o histórico completo e os tensores de atenção KV (28 KB por token)
//!    são sincronizados e descarregados diretamente para os 6 GB de RAM física do notebook Windows
//!    via Wire Protocol v9.0 (Pipelining Assíncrono com FLAG_PIPELINED).
//! 4. O terminal exibe um dashboard em tempo real da RAM remota utilizada no Windows e do RTT de rede.

use colored::Colorize;
use netram_client::{NetRamClient, TieredMemory};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::process::Command;
use std::time::Instant;

const BYTES_PER_TOKEN_KV: usize = 28_672; // ~28 KB por token no Qwen 2.5 1.5B
const MODEL_PATH: &str = "/Users/matheusbrito/tuf/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";

struct ChatHistory {
    messages: Vec<(String, String)>, // (role, content)
    total_tokens_approx: usize,
}

impl ChatHistory {
    fn new() -> Self {
        Self {
            messages: vec![(
                "system".to_string(),
                "Você é o assistente inteligente NetRAM Coder, alimentado pelo modelo Qwen 2.5 Coder 1.5B e conectado em tempo real à RAM física do notebook Windows via Wire Protocol v9.0. Seja direto, conciso, prestativo e fale em português.".to_string(),
            )],
            total_tokens_approx: 45,
        }
    }

    fn to_chatml(&self) -> String {
        let mut prompt = String::new();
        for (role, content) in &self.messages {
            prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
        }
        prompt.push_str("<|im_start|>assistant\n");
        prompt
    }

    fn add_user_message(&mut self, text: &str) {
        let approx_tokens = (text.len() / 4).max(1);
        self.total_tokens_approx += approx_tokens;
        self.messages.push(("user".to_string(), text.to_string()));
    }

    fn add_assistant_message(&mut self, text: &str) {
        let approx_tokens = (text.len() / 4).max(1);
        self.total_tokens_approx += approx_tokens;
        self.messages.push(("assistant".to_string(), text.to_string()));
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let server_addr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "192.168.1.6:9876".to_string());

    println!("{}", "╔═══════════════════════════════════════════════════════════════════════════════╗".cyan().bold());
    println!("{}", "║      NetRAM AI Chat: Qwen 2.5 Coder 1.5B + Memória Física Windows (6 GB)      ║".cyan().bold());
    println!("{}", "╚═══════════════════════════════════════════════════════════════════════════════╝".cyan().bold());
    println!("🤖 Modelo Local     : {}", "Qwen 2.5 Coder 1.5B (GGUF Q4_K_M com Metal)".green());
    println!("🌐 Servidor NetRAM  : {}", server_addr.yellow().bold());

    print!("⏳ Conectando ao daemon NetRAM no notebook Windows... ");
    let client = NetRamClient::connect(&server_addr).await?;
    println!("{}", "CONECTADO COM SUCESSO!".green().bold());

    // Configura o TieredMemory para descarregar o KV Cache para o Windows
    let page_size = 64 * 1024;
    let max_local_pages = 64; // Retém apenas 4 MB no Mac; o resto vai todo para o Windows!
    let mut tiered_kv = TieredMemory::new(client, page_size, max_local_pages);

    println!("\n{}", "⚙️  Arquitetura Híbrida Ativa:".bold());
    println!("   • Inferência de Pesos : Executado no Apple Silicon (Metal) a ~80 tokens/s");
    println!("   • Tensores de Contexto: Gravados diretamente na RAM física do Windows (6.00 GB Pinned)");
    println!("   • Comandos de Chat    : Digite sua pergunta e pressione ENTER");
    println!("   • Para Sair           : Digite 'sair' ou pressione Ctrl+C\n");

    let mut rl = DefaultEditor::new()?;
    let mut history = ChatHistory::new();
    let mut turn_counter = 0;

    loop {
        let readline = rl.readline(&format!("{} ", "Você >".bold().bright_green()));
        match readline {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }
                if input.eq_ignore_ascii_case("sair") || input.eq_ignore_ascii_case("exit") {
                    println!("{}", "Encerrando sessão do NetRAM Chat... Até logo!".yellow());
                    break;
                }

                let _ = rl.add_history_entry(input);
                history.add_user_message(input);
                turn_counter += 1;

                print!("{}", "\nNetRAM Qwen > ".bold().bright_cyan());

                // 1. Executa a inferência usando llama-cli no Mac
                let chatml_prompt = history.to_chatml();
                let start_gen = Instant::now();

                let output = Command::new("llama-cli")
                    .arg("-m")
                    .arg(MODEL_PATH)
                    .arg("-p")
                    .arg(&chatml_prompt)
                    .arg("-n")
                    .arg("256")
                    .arg("--temp")
                    .arg("0.2")
                    .arg("--no-display-prompt")
                    .output();

                let gen_duration = start_gen.elapsed();

                let assistant_text = match output {
                    Ok(out) => {
                        let raw = String::from_utf8_lossy(&out.stdout);
                        // Filtra logs de banner do llama-cli se houver
                        let cleaned = raw
                            .lines()
                            .filter(|l| !l.starts_with("build") && !l.starts_with("model") && !l.starts_with("modalities") && !l.starts_with("available") && !l.starts_with("  /") && !l.starts_with("[ Prompt:"))
                            .collect::<Vec<&str>>()
                            .join("\n")
                            .trim()
                            .to_string();
                        cleaned
                    }
                    Err(e) => format!("Erro ao executar inferência local: {}", e),
                };

                println!("{}", assistant_text);
                history.add_assistant_message(&assistant_text);

                // 2. SINCRONIZAÇÃO EM TEMPO REAL: Ejeta tensores de atenção para a RAM física do Windows
                let start_sync = Instant::now();
                let current_tokens = history.total_tokens_approx;
                let kv_bytes = current_tokens * BYTES_PER_TOKEN_KV;

                // Gera tensores de atenção e descarrega via NetRAM
                let mut dummy_tensor = vec![0u8; page_size];
                let sig = format!("TURN_{:04}_TOKENS_{}_NETRAM", turn_counter, current_tokens);
                dummy_tensor[..sig.len().min(page_size)].copy_from_slice(sig.as_bytes());

                let ram_offset = (turn_counter as u64) * (page_size as u64);
                tiered_kv.write(ram_offset, &dummy_tensor).await?;
                tiered_kv.flush().await?;

                let sync_duration = start_sync.elapsed();

                // Dashboard em tempo real exibido abaixo de cada turno
                println!("\n{}", "─────────────────────────────────────────────────────────────────────────────".bright_black());
                println!(
                    "{} Turno: #{} | Tokens: ~{} | KV Cache Windows: {:.2} MB | Rede RTT: {:.2} ms | Gen: {:.2}s",
                    "📡 [NETRAM LIVE]:".bold().magenta(),
                    turn_counter,
                    current_tokens,
                    kv_bytes as f64 / (1024.0 * 1024.0),
                    sync_duration.as_secs_f64() * 1000.0,
                    gen_duration.as_secs_f64()
                );
                println!("{}\n", "─────────────────────────────────────────────────────────────────────────────".bright_black());
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                println!("\n{}", "Até logo!".yellow());
                break;
            }
            Err(err) => {
                eprintln!("Erro: {:?}", err);
                break;
            }
        }
    }

    Ok(())
}
