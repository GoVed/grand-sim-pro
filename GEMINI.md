# Project Mandates (GEMINI.md)

As a senior engineer agent working on this project, you MUST follow these instructions for every change you make.

## 1. Development Lifecycle
- **Implementation:** Apply surgical, idiomatic changes.
- **Verification:** 
    - Run `cargo build` and ensure there are NO warnings or errors.
    - Run `cargo test` to ensure all unit tests pass.
- **Performance:**
    - Run `bash scripts/test_perf.sh` after every significant change to log performance into `PERFORMANCE_LOG.md`.
    - `PERFORMANCE_LOG.md` must hold a maximum of the 5 most recent performance runs.
    - **Performance Regressions are NOT allowed.** Compare the latest run against the previous one in `PERFORMANCE_LOG.md`. If performance has degraded (e.g., times increased significantly without justification), you MUST identify the bottleneck and fix the degradation.

## 2. Codebase Architecture & File Map
You MUST update this section if you add, move, or significantly change the responsibility of any file.

| File | Purpose | Key Responsibilities |
|------|---------|----------------------|
| `src/lib.rs` | Library Root | Exports all simulation modules for binary and test access. |
| `src/main.rs` | Entry Point | Handles window initialization, UI loop, and thread spawning. |
| `src/agent.rs` | Agent Logic | NN architecture (Sparse W1), sexual reproduction, and mutation. |
| `src/simulation.rs` | Sim Manager | High-level simulation loop, birth processing, and grid occupancy. |
| `src/environment.rs` | World Engine | Procedural 3D spherical noise generation and tile state management. |
| `src/gpu_engine.rs` | WGPU Backend | Handles GPGPU compute shaders and agent/map rendering. |
| `src/sim.wgsl` | Compute Shader | The "Hot Loop" - GPU code for agent movement, perception, and NN inference. |
| `src/config.rs` | Global Config | Simulation constants, economic parameters, and GPU-safe struct. |
| `src/shared.rs` | Shared Types | Data structures shared between the UI thread and Simulation thread. |
| `src/ui.rs` | UI Components | Macroquad-based dashboard, graphs, and agent inspectors. |
| `tests/performance.rs` | Benchmarking | Integration tests for tracking sim and reproduction throughput. |
| `scripts/test_perf.sh` | Perf Logger | Automated tool for updating `PERFORMANCE_LOG.md`. |

## 3. Code Standards
- **Sync Documentation:** If you modify a file's responsibility, you MUST update the "Codebase Architecture" table above. You MUST also update `readme.md` if your changes affect the core engine, add features, or change the user-facing controls/benchmarks.
- **Library First:** Always move core logic to the library (`src/lib.rs`) rather than the binary (`src/main.rs`) to ensure testability.
- Maintain high-performance GPGPU standards; avoid expensive operations in hot loops.
- Use `bytemuck` for data passed to the GPU; ensure strict memory alignment.

## 3. Communication
- Fulfill user directives thoroughly, including adding tests for new features.
- If a change affects the simulation logic, update the corresponding unit tests in `src/simulation.rs` or `src/agent.rs`.
