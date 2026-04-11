# Grand Sim Pro - Project Status & Mandates

## 1. Development Lifecycle

1. **Implementation:** Apply surgical, idiomatic changes.
2. **Verification:**
    - Run `cargo build` and ensure there are **NO** warnings or errors in any profile.
    - Run `cargo test` to ensure all unit and UI verification tests pass.
3. **Performance:**
    - Run `bash scripts/test_perf.sh` after every significant change to log performance into `PERFORMANCE_LOG.md`.
    - `PERFORMANCE_LOG.md` must hold a maximum of the **5 most recent** performance runs.
    - **Performance Regressions are NOT allowed.** Compare the latest run against the previous one. If performance has degraded, you MUST identify the bottleneck and fix it.

## 2. Codebase Architecture & File Map

| File | Purpose | Key Responsibilities |
| :--- | :--- | :--- |
| `src/lib.rs` | Library Root | Exports all modules for binary and test access. |
| `src/main.rs` | Entry Point | Window initialization, startup prompts, UI loop, and thread spawning. |
| `src/agent.rs` | Agent Logic | NN architecture, sexual reproduction, mutation, and situational probing. |
| `src/ui_logic.rs` | UI Logic | (NEW) Pure-logic backend for UI. Handles behavior math, config filtering, and layouts. |
| `src/ui.rs` | UI Components | Macroquad-based rendering for dashboards, graphs, and inspectors. |
| `src/telemetry.rs` | Data Exporter | (NEW) High-fidelity CSV export of population and environmental metrics. |
| `src/simulation.rs` | Sim Manager | High-level simulation loop, birth processing, and grid occupancy. |
| `src/environment.rs` | World Engine | Procedural 3D spherical noise generation and tile state management. |
| `src/gpu_engine.rs` | WGPU Backend | Handles GPGPU compute shaders and agent/map rendering. |
| `src/sim.wgsl` | Compute Shader | The "Hot Loop" - GPU code for agent movement, perception, and NN inference. |
| `src/config.rs` | Global Config | Simulation constants, economic parameters, and GPU-safe struct. |
| `src/shared.rs` | Shared Types | Data structures shared between the UI thread and Simulation thread. |
| `tests/performance.rs` | Benchmarking | Integration tests for tracking sim and reproduction throughput. |
| `tests/ui_verification.rs` | UI Tests | Headless verification of UI logic and behavioral inference. |
| `scripts/test_perf.sh` | Perf Logger | Automated tool for updating `PERFORMANCE_LOG.md`. |

## 3. Code Standards

- **Mandatory Documentation Sync:** As an agent, you MUST update `readme.md` and `ARCHITECTURE.md` immediately after implementing new features or making architectural changes. This ensures the documentation never drifts from the current implementation.
- **Sync Codebase Map:** If you modify a file's responsibility or add new modules (e.g., `ui_logic.rs`), you MUST update the table in Section 2 above.
- **Library First:** Move core logic to the library (`src/lib.rs`) to ensure testability.

- **High Performance:** Maintain GPGPU standards; avoid expensive operations in hot loops.
- **GPU Safety:** Use `bytemuck` for data passed to the GPU; ensure strict 16-byte alignment.
- **UI Testability:** Logic MUST be decoupled from Macroquad using the `ui_logic.rs` pattern.

## 4. Communication & Testing

- Fulfill user directives thoroughly, including adding tests for new features.
- If a change affects simulation logic, update unit tests in `src/simulation.rs` or `src/agent.rs`.
- Aim for >90% coverage on `src/ui_logic.rs`.
