# Project Review & ToDo Roadmap

A comprehensive code review of `rust-mlx-cli` was performed on `main.rs`, `Cargo.toml`, build configuration, and UX. Below is the categorized backlog of identified issues, improvements, and feature tasks.

---

## 1. Bug Fixes & Safety Hardening (P0 / Urgent)

- [x] **Sanitize and Sandbox `run_shell` Tool**:
  - *Implemented*: Added `validate_shell_command` blocking destructive actions (`rm`, `rmdir`, `dd`, `mkfs`, fork bombs, privilege escalation `sudo`/`su`, permissive `chmod`, and piping remote curl/wget scripts to shell).
- [x] **Add Command Timeout to `run_shell`**:
  - *Implemented*: Runs asynchronously via `tokio::process::Command` with `kill_on_drop(true)` and a 15-second `tokio::time::timeout`.
- [x] **Prevent Out-Of-Memory in `read_file`**:
  - *Implemented*: Checks file metadata (`is_file`), streams with `std::io::Read::take(4096)`, checks for binary data (null bytes), and notes truncation.
- [x] **Prevent Infinite Tool Loops**:
  - *Implemented*: Capped tool loop execution at `MAX_TOOL_TURNS` (10 turns per user input) with graceful notification.
- [x] **Atomic Cache Completion Verification**:
  - *Implemented*: `is_valid_cache` verifies non-zero byte weight shards (`q4k-0.uqff`) and parsable `uqff_report.json`; cleans up incomplete/corrupted cache folders before re-quantization.

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
