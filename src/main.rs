use anyhow::Result;
use mistralrs::{
    Function, IsqType, RequestBuilder, TextMessageRole, TextMessages, TextModelBuilder, Tool,
    ToolChoice, ToolType, UqffTextModelBuilder,
};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

struct ToolEntry {
    spec: Tool,
    deferred: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let t0 = Instant::now();
    let model_id = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "LiquidAI/LFM3-1.7B".to_string());

    // Store pre-quantized UQFF cache in ~/.cache/rust-mlx-cli/<model>-q4k/
    let cache_dir = dirs_cache_path();
    let safe_name = model_id.replace('/', "--");
    let model_cache_dir = cache_dir.join(format!("{}-q4k", safe_name));
    let uqff_target = model_cache_dir.join("q4k.uqff");

    let model = if is_valid_cache(&model_cache_dir) {
        // FAST PATH: Load pre-quantized UQFF directly from verified local cache directory
        eprintln!(
            "Loading cached Q4K UQFF model from {}...",
            model_cache_dir.display()
        );
        UqffTextModelBuilder::new(
            model_cache_dir.display().to_string(),
            vec![PathBuf::from("q4k-0.uqff")],
        )
        .into_inner()
        .build()
        .await?
    } else {
        // FIRST RUN OR CORRUPTED CACHE: Clean up partial state and quantize to Q4K
        if model_cache_dir.exists() {
            eprintln!(
                "Incomplete or invalid cache detected in {}. Cleaning up and re-quantizing...",
                model_cache_dir.display()
            );
            let _ = std::fs::remove_dir_all(&model_cache_dir);
        }
        std::fs::create_dir_all(&model_cache_dir)?;
        eprintln!(
            "First run: quantizing {} to Q4K and saving UQFF cache to {}...",
            model_id,
            model_cache_dir.display()
        );
        TextModelBuilder::new(&model_id)
            .with_isq(IsqType::Q4K)
            .write_uqff(uqff_target)
            .build()
            .await?
    };

    eprintln!("Ready in {:.2?}. Type '/clear' or '/quit'.\n", t0.elapsed());

    let catalog = build_tool_catalog();
    let mut active_tool_names: HashSet<String> = catalog
        .iter()
        .filter(|(_, entry)| !entry.deferred)
        .map(|(name, _)| name.clone())
        .collect();

    let system_prompt = "You are a local terminal assistant. You start with only `search_tools` loaded. \
        Whenever a user request requires an external capability (files, shell, system stats), first call \
        `search_tools` with a short keyword query to discover and load the relevant tool schema, then invoke it.";

    let mut messages = TextMessages::new().add_message(TextMessageRole::System, system_prompt);

    let stdin = io::stdin();
    let mut input = String::new();

    loop {
        print!("> ");
        io::stdout().flush()?;

        input.clear();
        if stdin.read_line(&mut input)? == 0 {
            println!();
            break;
        }

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == "/quit" || trimmed == "/exit" {
            break;
        }
        if trimmed == "/clear" {
            messages = TextMessages::new().add_message(TextMessageRole::System, system_prompt);
            active_tool_names.retain(|name| !catalog[name].deferred);
            println!("[context and discovered tools cleared]\n");
            continue;
        }

        messages = messages.add_message(TextMessageRole::User, trimmed);

        // Maintain a RequestBuilder across tool turns so tool calls and results append cleanly
        let mut req_builder = RequestBuilder::from(messages.clone())
            .enable_thinking(true)
            .set_sampler_temperature(0.1)
            .set_sampler_topk(50)
            .set_sampler_frequency_penalty(0.1);

        const MAX_TOOL_TURNS: usize = 10;
        let mut tool_turn_count = 0;

        loop {
            let active_tools: Vec<Tool> = active_tool_names
                .iter()
                .filter_map(|name| catalog.get(name).map(|e| e.spec.clone()))
                .collect();

            req_builder = req_builder
                .set_tools(active_tools)
                .set_tool_choice(ToolChoice::Auto);

            let response = model.send_chat_request(req_builder.clone()).await?;
            let choice = &response.choices[0];

            if let Some(tool_calls) = &choice.message.tool_calls {
                if !tool_calls.is_empty() {
                    tool_turn_count += 1;
                    if tool_turn_count > MAX_TOOL_TURNS {
                        eprintln!(
                            "\x1b[31m[warning] Maximum tool iteration limit ({}) reached. Halting tool loop.\x1b[0m",
                            MAX_TOOL_TURNS
                        );
                        let limit_notice = format!(
                            "[Execution stopped: exceeded safety limit of {} tool turns]",
                            MAX_TOOL_TURNS
                        );
                        println!("{}\n", limit_notice);
                        messages = messages.add_message(TextMessageRole::Assistant, &limit_notice);
                        break;
                    }

                    let assistant_text = choice.message.content.clone().unwrap_or_default();
                    req_builder = req_builder.add_message_with_tool_call(
                        TextMessageRole::Assistant,
                        assistant_text.clone(),
                        tool_calls.clone(),
                    );
                    messages = messages.add_message(TextMessageRole::Assistant, &assistant_text);

                    for call in tool_calls {
                        let fn_name = &call.function.name;
                        let args_str = &call.function.arguments;
                        eprintln!("\x1b[33m[tool_call] {}({})\x1b[0m", fn_name, args_str);

                        let output = execute_tool(
                            fn_name,
                            args_str,
                            &catalog,
                            &mut active_tool_names,
                        )
                        .await;

                        eprintln!("\x1b[36m[tool_result] {}\x1b[0m", output);
                        req_builder = req_builder.add_tool_message(&output, &call.id);
                        messages = messages.add_message(TextMessageRole::Tool, &output);
                    }
                    continue;
                }
            }

            if let Some(reasoning) = &choice.message.reasoning_content {
                println!("\x1b[2m<think>\n{}\n</think>\x1b[0m\n", reasoning.trim());
            }
            if let Some(content) = &choice.message.content {
                println!("{}\n", content.trim());
                messages = messages.add_message(TextMessageRole::Assistant, content);
            }
            break;
        }
    }

    Ok(())
}

fn dirs_cache_path() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".cache").join("rust-mlx-cli")
    } else {
        PathBuf::from(".cache")
    }
}

fn is_valid_cache(model_cache_dir: &Path) -> bool {
    let shard = model_cache_dir.join("q4k-0.uqff");
    let report = model_cache_dir.join("uqff_report.json");
    if !shard.is_file() || !report.is_file() {
        return false;
    }
    let Ok(shard_meta) = std::fs::metadata(&shard) else {
        return false;
    };
    if shard_meta.len() == 0 {
        return false;
    }
    let Ok(report_content) = std::fs::read(&report) else {
        return false;
    };
    serde_json::from_slice::<Value>(&report_content).is_ok()
}

fn validate_shell_command(cmd: &str) -> std::result::Result<(), String> {
    let trimmed = cmd.trim();
    if trimmed.is_empty() {
        return Err("Error: Empty shell command".to_string());
    }

    let lower = trimmed.to_lowercase();
    let blocked_patterns = [
        "rm ", "rm\t", "rmdir", "mkfs", "dd ", "shred", "srm",
        ":(){ :|:& };:", ">/dev/sd", ">/dev/nvme", ">/dev/disk", ">/dev/rdisk",
        "> /dev/sd", "> /dev/nvme", "> /dev/disk", "> /dev/rdisk",
        "sudo ", "sudo\t", "su ", "su\t", "chmod 777", "chmod -r", "chown ",
        "shutdown", "reboot", "init 0", "halt", "poweroff",
    ];

    for pattern in blocked_patterns {
        if lower.contains(pattern) || lower == pattern.trim() {
            return Err(format!(
                "Execution blocked for safety: command contains disallowed pattern '{}'. Only safe inspection commands are permitted.",
                pattern.trim()
            ));
        }
    }

    if (lower.contains("curl") || lower.contains("wget"))
        && (lower.contains("| sh") || lower.contains("| bash") || lower.contains("| zsh"))
    {
        return Err(
            "Execution blocked for safety: piping downloaded scripts into a shell is disallowed."
                .to_string(),
        );
    }

    Ok(())
}

fn build_tool_catalog() -> HashMap<String, ToolEntry> {
    let mut map = HashMap::new();

    map.insert(
        "search_tools".into(),
        ToolEntry {
            deferred: false,
            spec: make_tool(
                "search_tools",
                "Search the deferred tool catalog by keyword to unlock tools (e.g., 'file', 'shell', 'system').",
                json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Keyword describing the capability needed"
                        }
                    },
                    "required": ["query"]
                }),
            ),
        },
    );

    map.insert(
        "read_file".into(),
        ToolEntry {
            deferred: true,
            spec: make_tool(
                "read_file",
                "Read the UTF-8 text contents of a local file path.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path to the file" }
                    },
                    "required": ["path"]
                }),
            ),
        },
    );

    map.insert(
        "get_system_info".into(),
        ToolEntry {
            deferred: true,
            spec: make_tool(
                "get_system_info",
                "Get macOS version, CPU architecture, and uptime.",
                json!({
                    "type": "object",
                    "properties": {}
                }),
            ),
        },
    );

    map.insert(
        "run_shell".into(),
        ToolEntry {
            deferred: true,
            spec: make_tool(
                "run_shell",
                "Execute a read-only shell command (such as ls, git status, or df -h).",
                json!({
                    "type": "object",
                    "properties": {
                        "command": { "type": "string", "description": "Shell command to execute" }
                    },
                    "required": ["command"]
                }),
            ),
        },
    );

    map
}

fn make_tool(name: &str, description: &str, parameters: Value) -> Tool {
    let params_map: HashMap<String, Value> = serde_json::from_value(parameters).unwrap_or_default();
    Tool {
        tp: ToolType::Function,
        function: Function {
            name: name.to_string(),
            description: Some(description.to_string()),
            parameters: Some(params_map),
            strict: None,
        },
    }
}

async fn execute_tool(
    name: &str,
    raw_args: &str,
    catalog: &HashMap<String, ToolEntry>,
    active_tool_names: &mut HashSet<String>,
) -> String {
    let args: Value = match serde_json::from_str(raw_args) {
        Ok(v) => v,
        Err(e) => return format!("Error parsing tool arguments as JSON: {}", e),
    };

    match name {
        "search_tools" => {
            let query = args["query"].as_str().unwrap_or("").to_lowercase();
            let mut matched_schemas = Vec::new();

            for (tool_name, entry) in catalog {
                if !entry.deferred {
                    continue;
                }
                let desc = entry
                    .spec
                    .function
                    .description
                    .as_deref()
                    .unwrap_or("")
                    .to_lowercase();

                if tool_name.to_lowercase().contains(&query)
                    || desc.contains(&query)
                    || query.is_empty()
                {
                    active_tool_names.insert(tool_name.clone());
                    matched_schemas.push(json!({
                        "type": "tool_reference",
                        "name": tool_name,
                        "description": entry.spec.function.description,
                        "parameters": entry.spec.function.parameters
                    }));
                }
            }

            if matched_schemas.is_empty() {
                json!({ "status": "no_matches", "available_hints": ["file", "system", "shell"] })
                    .to_string()
            } else {
                json!({
                    "unlocked_tools": matched_schemas,
                    "instruction": "These tools are now active. Call the appropriate tool to answer the user."
                })
                .to_string()
            }
        }
        "read_file" => {
            let path_str = args["path"].as_str().unwrap_or("").trim();
            if path_str.is_empty() {
                return "Error: Missing required argument 'path'".to_string();
            }
            let path = Path::new(path_str);
            let metadata = match std::fs::metadata(path) {
                Ok(m) => m,
                Err(e) => return format!("Error accessing '{}': {}", path_str, e),
            };
            if !metadata.is_file() {
                return format!(
                    "Error: '{}' is a directory or special path, not a regular file.",
                    path_str
                );
            }
            let file = match std::fs::File::open(path) {
                Ok(f) => f,
                Err(e) => return format!("Error opening '{}': {}", path_str, e),
            };

            let mut buf = Vec::new();
            if let Err(e) = file.take(4096).read_to_end(&mut buf) {
                return format!("Error reading '{}': {}", path_str, e);
            }

            if buf.contains(&0) {
                return format!("Error: '{}' appears to be a binary file.", path_str);
            }

            let content = String::from_utf8_lossy(&buf);
            if metadata.len() > 4096 {
                format!(
                    "{}\n... [truncated: showing first 4096 of {} bytes]",
                    content,
                    metadata.len()
                )
            } else {
                content.to_string()
            }
        }
        "get_system_info" => {
            let timeout_duration = Duration::from_secs(5);
            let child = tokio::process::Command::new("uname")
                .arg("-a")
                .kill_on_drop(true)
                .output();

            match tokio::time::timeout(timeout_duration, child).await {
                Ok(Ok(o)) => String::from_utf8_lossy(&o.stdout).trim().to_string(),
                Ok(Err(e)) => format!("Error: {}", e),
                Err(_) => "Error: System info probe timed out after 5 seconds.".to_string(),
            }
        }
        "run_shell" => {
            let cmd = args["command"].as_str().unwrap_or("").trim();
            if let Err(err) = validate_shell_command(cmd) {
                return err;
            }

            let timeout_duration = Duration::from_secs(15);
            let child = tokio::process::Command::new("sh")
                .arg("-c")
                .arg(cmd)
                .kill_on_drop(true)
                .output();

            match tokio::time::timeout(timeout_duration, child).await {
                Ok(Ok(output)) => {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    let combined = format!("{}{}", stdout, stderr);
                    if combined.is_empty() {
                        "[command completed with no output]".to_string()
                    } else if combined.len() > 4000 {
                        let truncated: String = combined.chars().take(4000).collect();
                        format!("{}\n... [output truncated at 4000 characters]", truncated)
                    } else {
                        combined
                    }
                }
                Ok(Err(e)) => format!("Error executing command: {}", e),
                Err(_) => format!(
                    "Error: Command timed out after {} seconds.",
                    timeout_duration.as_secs()
                ),
            }
        }
        _ => format!("Unknown tool: {}", name),
    }
}
