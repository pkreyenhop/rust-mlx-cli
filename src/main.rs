use anyhow::Result;
use mistralrs::{
    Function, IsqType, RequestBuilder, TextMessageRole, TextMessages, TextModelBuilder, Tool,
    ToolChoice, ToolType, UqffTextModelBuilder,
};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

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
    let uqff_shard = model_cache_dir.join("q4k-0.uqff");
    let uqff_target = model_cache_dir.join("q4k.uqff");

    let model = if uqff_shard.exists() {
        // FAST PATH: Load pre-quantized UQFF directly from local cache directory
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
        // FIRST RUN ONLY: Quantize once to Q4K and persist as UQFF for future launches
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
                        );

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

fn execute_tool(
    name: &str,
    raw_args: &str,
    catalog: &HashMap<String, ToolEntry>,
    active_tool_names: &mut HashSet<String>,
) -> String {
    let args: Value = serde_json::from_str(raw_args).unwrap_or_else(|_| json!({}));

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

                if tool_name.to_lowercase().contains(&query) || desc.contains(&query) || query.is_empty() {
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
                json!({ "status": "no_matches", "available_hints": ["file", "system", "shell"] }).to_string()
            } else {
                json!({
                    "unlocked_tools": matched_schemas,
                    "instruction": "These tools are now active. Call the appropriate tool to answer the user."
                })
                .to_string()
            }
        }
        "read_file" => {
            let path = args["path"].as_str().unwrap_or("");
            match std::fs::read_to_string(path) {
                Ok(content) => content.chars().take(4000).collect(),
                Err(e) => format!("Error reading '{}': {}", path, e),
            }
        }
        "get_system_info" => {
            let out = Command::new("uname").arg("-a").output();
            match out {
                Ok(o) => String::from_utf8_lossy(&o.stdout).trim().to_string(),
                Err(e) => format!("Error: {}", e),
            }
        }
        "run_shell" => {
            let cmd = args["command"].as_str().unwrap_or("");
            let out = Command::new("sh").arg("-c").arg(cmd).output();
            match out {
                Ok(o) => {
                    let stdout = String::from_utf8_lossy(&o.stdout);
                    let stderr = String::from_utf8_lossy(&o.stderr);
                    format!("{}{}", stdout, stderr).chars().take(4000).collect()
                }
                Err(e) => format!("Error: {}", e),
            }
        }
        _ => format!("Unknown tool: {}", name),
    }
}
