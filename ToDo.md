# Project Review & ToDo Roadmap

A comprehensive code review of `rust-mlx-cli` was performed on `main.rs`, `Cargo.toml`, build configuration, and UX. Below is the categorized backlog of identified issues, improvements, and feature tasks.

---

## 1. Bug Fixes & Safety Hardening (P0 / Urgent)

- [ ] **Sanitize and Sandbox `run_shell` Tool**:
  - *Current behavior*: Executes arbitrary shell commands via `sh -c <command>` without restrictions or validation.
  - *Risk*: Model can run destructive commands (`rm`, overwriting system files, executing malware).
  - *Fix*: Implement a command whitelist/blacklist, restrict write permissions, or prompt for user confirmation before executing write/mutating shell commands.
- [ ] **Add Command Timeout to `run_shell`**:
  - *Current behavior*: `Command::new("sh").arg("-c").output()` blocks synchronously with no timeout.
  - *Risk*: Commands like `cat`, `sleep 1000`, or interactive utilities freeze the entire CLI process indefinitely.
  - *Fix*: Execute asynchronously with `tokio::process::Command` wrapped in `tokio::time::timeout(Duration::from_secs(15), ...)`.
- [ ] **Prevent Out-Of-Memory in `read_file`**:
  - *Current behavior*: `std::fs::read_to_string(path)` reads the entire file into memory before `.take(4000)`.
  - *Fix*: Check file metadata first (`entry.metadata()?.len()`); read with `std::io::Read::take(4096)` to protect against multi-gigabyte or binary files.
- [ ] **Prevent Infinite Tool Loops**:
  - *Current behavior*: The tool invocation `loop { ... continue; }` has no iteration ceiling.
  - *Fix*: Add a `max_tool_iterations` counter (e.g., maximum 10-15 steps per user turn) with an error/cutoff message to prevent recursive loops.
- [ ] **Atomic Cache Completion Verification**:
  - *Current behavior*: Cache presence only checks `q4k-0.uqff.exists()`. If interrupted during quantization, a corrupted file causes subsequent launches to crash.
  - *Fix*: Verify both `uqff_report.json` and `q4k-0.uqff` exist and are valid JSON/safetensors, or write to a staging temp directory and rename atomically upon completion.

---

## 2. CLI Ergonomics & Configuration (P1 / High)

- [ ] **Adopt `clap` for Robust CLI Arguments**:
  - *Current behavior*: Uses naive `std::env::args().nth(1)`. Running `--help` attempts to download `--help` from Hugging Face.
  - *Fix*: Integrate `clap` (derive):
    - `-m, --model <ID>`: Model identifier (default: `LiquidAI/LFM3-1.7B`).
    - `-q, --quant <TYPE>`: Quantization format (`q4k`, `q8_0`, `q4_0`, etc.).
    - `-t, --temperature <FLOAT>`: Generation temperature (default: `0.1`).
    - `--top-k <INT>`: Sampler top-k.
    - `--cache-dir <PATH>`: Custom directory for storing quantized models.
    - `--system-prompt <TEXT>`: Custom system prompt override.
    - `--help` and `--version`.
- [ ] **Rich Terminal REPL (`rustyline` or `reedline`)**:
  - *Current behavior*: Uses basic `std::io::stdin().read_line()` with no line editing, history, or cursor keys.
  - *Fix*: Replace with `rustyline` to enable:
    - Persistent command/prompt history (arrow up/down).
    - Emacs/Vi cursor navigation, word jumping, and line editing.
    - History file persistence in `~/.cache/rust-mlx-cli/history.txt`.
- [ ] **Graceful Cancellation on Ctrl+C**:
  - *Current behavior*: Ctrl+C during model generation kills the entire process.
  - *Fix*: Intercept SIGINT (`tokio::signal::ctrl_c()`) so Ctrl+C cancels the active generation/tool turn and returns to the `> ` prompt.

---

## 3. Streaming & Output Experience (P1 / High)

- [ ] **Token-by-Token Streaming**:
  - *Current behavior*: Uses non-streaming `send_chat_request`, causing the terminal to hang until the entire generation completes.
  - *Fix*: Switch to `stream_chat_request` to stream tokens and reasoning content to stdout in real time.
- [ ] **Render Markdown & Code Highlighting**:
  - *Enhancement*: Format markdown tables, lists, and code fences using terminal styling (`termimad` or `syntect`/`crossterm`).

---

## 4. Code Architecture & Modularity (P2 / Medium)

- [ ] **Refactor Monolithic `src/main.rs` into Domain Modules**:
  - `src/cli.rs`: CLI argument definitions and configuration.
  - `src/model.rs`: Model initialization, quantization cache detection, and builders.
  - `src/tools/mod.rs`: `ToolCatalog`, `ToolEntry`, and dispatcher.
  - `src/tools/search.rs`: Dynamic tool search and discovery.
  - `src/tools/file.rs`: File inspection and operations.
  - `src/tools/system.rs`: System diagnostics and shell commands.
  - `src/repl.rs`: Interactive prompt loop, message history, and execution loop.
- [ ] **Context Window Management**:
  - *Issue*: `messages` grows indefinitely, eventually exceeding the model's sequence length limit.
  - *Fix*: Implement token counting and context trimming/sliding window (e.g. keeping system prompt + last N messages).
- [ ] **Preserve Structured Tool History**:
  - *Issue*: Line 123 in `src/main.rs` converts assistant tool calls into plain text when updating `messages`, which can degrade tool reasoning across subsequent turns.
  - *Fix*: Maintain full structured tool call and tool message representations in the message history.

---

## 5. Tool Expansion & Extensibility (P2 / Medium)

- [ ] **File Writing & Patching Tool**:
  - Add `write_file` and `edit_file` tools to enable agentic code editing and file creation.
- [ ] **Workspace / Project Discovery Tool**:
  - Add `grep_search` and `find_files` to allow the assistant to explore codebases without relying on shell commands.
- [ ] **Web / Fetch Capability**:
  - Add HTTP GET tool (e.g., via `reqwest`) to fetch documentation or API responses.

---

## 6. Testing & CI (P3 / Low)

- [ ] **Unit Tests for Tools**:
  - Add unit tests for `execute_tool`, `search_tools`, and input validation.
- [ ] **Cache Integration Tests**:
  - Add integration tests verifying UQFF cache detection and fallback behavior.
- [ ] **GitHub Actions Workflow**:
  - Set up CI for `cargo check`, `cargo clippy`, and `cargo fmt`.
