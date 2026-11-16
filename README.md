# rust-mlx-cli

A high-performance local AI terminal assistant written in Rust, powered by [`mistral.rs`](https://github.com/EricLBuehler/mistral.rs) with Apple Silicon (Metal) acceleration, automatic Q4K UQFF quantization caching, and dynamic deferred tool calling.

---

## Features

- **Apple Silicon Metal Acceleration**: Native GPU execution on macOS via `mistralrs` Metal backend.
- **Automatic Quantization & Fast Local Caching**:
  - **First launch**: Quantizes the target model to **Q4K** and caches the generated UQFF shard and tokenizer metadata in `~/.cache/rust-mlx-cli/<model>-q4k/`.
  - **Subsequent launches**: Bypasses network and quantization, loading directly from local UQFF artifacts in sub-second time.
- **Dynamic / Deferred Tool Calling**:
  - The model starts with only `search_tools` loaded in context to minimize prompt overhead.
  - The agent dynamically discovers and loads tools on-demand via keywords before invoking them.
- **Built-in Tool Catalog**:
  - `search_tools`: Search available capabilities by keyword (e.g., `file`, `shell`, `system`).
  - `read_file`: Read UTF-8 contents of local files.
  - `get_system_info`: Inspect macOS version, kernel, and CPU architecture.
  - `run_shell`: Execute shell commands directly in the environment.
- **Thinking / Reasoning Display**: Renders model thought blocks (`<think>...</think>`) in dimmed terminal output.
- **Interactive REPL**: Conversational context preserved across multi-turn exchanges with `/clear` and `/quit` commands.

---

## Prerequisites

- **macOS** on Apple Silicon (M1/M2/M3/M4 or later)
- **Rust toolchain** (1.75+ recommended):
  ```sh
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
- **Xcode Command Line Tools**:
  ```sh
  xcode-select --install
  ```

---

## Build

Compile the release binary with Metal support enabled:

```sh
cargo build --release
```

The resulting binary will be located at `target/release/rust-mlx-cli`.

---

## Usage

### Default Model (`LiquidAI/LFM3-1.7B`)

Run without arguments to use the default model:

```sh
./target/release/rust-mlx-cli
```

### Custom Model

You can specify any Hugging Face model ID as the first argument:

```sh
./target/release/rust-mlx-cli "LiquidAI/LFM3-1.7B"
```

### REPL Commands

- `/clear`: Reset conversation context and unload dynamically discovered tools.
- `/quit` or `/exit`: Exit the assistant.

---

## Cache Management

UQFF quantized models and artifacts are stored in:

```
~/.cache/rust-mlx-cli/<model-id>-q4k/
```

To purge the cache and force re-quantization:

```sh
rm -rf ~/.cache/rust-mlx-cli/
```

---

## License

MIT or Apache-2.0.
