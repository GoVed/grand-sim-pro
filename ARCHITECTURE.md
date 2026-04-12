# Architecture & Implementation Detail

This document outlines the technical design decisions and internal workings of the Grand Sim Pro engine.

## 1. Hybrid CPU-GPU Compute Engine
The simulation utilizes a "Think on GPU, Synchronize on CPU" pattern to achieve massive parallelism.

### Split Memory Architecture (Optimized)
To overcome PCIe bandwidth bottlenecks and CPU-side sorting overhead, the engine uses a **Split-Vector Pattern**:
- **`AgentState` (~200 bytes):** Contains dynamic, frequently updated fields (x, y, health, wealth, intents). This vector is fetched at 60Hz and is the primary target for CPU-side spatial sorting.
- **`Genetics` (28 KB):** Contains static neural network weights. This massive buffer remains stable on the GPU.
- **`genetics_index` Indirection:** Each `AgentState` contains a fixed pointer to its entry in the `Genetics` buffer. This allows the CPU to reorder (sort) the small state structs while the heavy weights stay in place, reducing sorting memory moves by **>100x**.

### Memory Alignment & Synchronization
- **Strict Alignment:** All data structures shared with the GPU (`src/config.rs`, `src/agent.rs`) are marked with `#[repr(C)]` and use `bytemuck` for zero-copy casting. Memory is strictly 16-byte aligned to match WGSL requirements.
- **Buffer Reuse:** To avoid per-frame allocation overhead, `SimulationManager` maintains reusable `Vec` and `HashMap` buffers for genetics and birth processing.
- **Decoupled Spatial Sorting:** Before each GPU dispatch, the CPU performs a **Parallel Spatial Sort** (`rayon`) of the `AgentState` population. This ensures that agents processed in the same GPU workgroup are spatially clustered, maximizing the hit rate of the LDS cache.

## 2. Logic-View Separation (UI)
The UI is architected to be testable without an active graphics context (headless).

- **`src/ui_logic.rs`:** The "Brain" of the UI. It handles:
    - Neural Influence Matrix multiplication ($W_3 \times W_2 \times W_1$).
    - Situational behavioral probing logic.
    - Configuration filtering and search matching.
    - Responsive layout calculations (Panel X/Y/W/H).
- **`src/ui.rs`:** The "Body" of the UI. It strictly performs rendering using `macroquad`, mapping `ui_logic` types to drawing calls.

## 3. Multithreading Model
The application runs on two primary threads:
1.  **Main/UI Thread:** Runs the `macroquad` loop, handles user input, and renders the viewport.
2.  **Simulation Thread:** Dispatches compute shaders via `wgpu`. It uses non-blocking `try_lock` patterns to read/write `SharedData`, ensuring simulation spikes never cause UI "stutter."

## 4. GPGPU Simulation Pipeline
The WGSL kernels (split across `src/shaders/` for maintainability) form the "hot loop" of the project, primarily located in `src/shaders/sim.wgsl`.

### Expanded Cognitive Model (Human-Approximate)
To move beyond simple reactive behaviors, the agent brain has been expanded based on biological and psychological scaling laws:
- **Representational Capacity:** Hidden layer width increased to **128 nodes**, allowing for complex non-linear decision-making and persistence of multiple internal states.
- **Working Memory:** Expanded to **24 channels** (grounded in Miller's Law), enabling the agent to maintain concurrent goals, spatial anchors, and social signatures.
- **Vocal/Signal Bandwidth:** Increased to **12 communication channels**, providing a high-dimensional semantic space for emergent tribal and resource-related signals.
- **Competitive Intent Dynamics:** Implemented a "Winner-Take-All" priority system. Agents face realistic physical trade-offs; for example, construction (building) requires absolute immobility and is only performed if its neural intent exceeds the desire for movement.

- **Cooperative LDS Caching:** GPU workgroups utilize Local Device Storage (LDS) to cache a 16x16 tile patch of the environment. All 64 agents in a workgroup sample their vision from this high-speed local cache rather than global VRAM.
- **Register Pressure Optimization:** To prevent register spilling, the kernel avoids loading the massive `Agent` struct into local memory, instead operating directly on the storage buffer via pointer-style indexing.
- **Spatial Awareness:** LiDAR vision is simulated by sampling the pheromone and elevation maps in a 3x3 grid around each agent.
- **Physical Development Scaling:** Agent mobility (speed) and physical endurance (stamina consumption) are dynamically scaled based on the agent's maturity ratio ($\text{age} / \text{puberty\_age}$), ensuring newborns are appropriately vulnerable.
- **Physics:** Longitude convergence and spherical wrapping are handled mathematically at the physics step to simulate a 3D globe on a 2D array.

## 5. Telemetry & Data Export
A dedicated `TelemetryExporter` in `src/telemetry.rs` provides a high-fidelity time-series export for research.
- **Sampling:** Telemetry is captured on the simulation thread during the 60Hz agent fetch cycle, ensuring zero impact on GPU compute performance.
- **Metrics:** Aggregates population-wide biometrics (Age, Health, Wealth) and global environmental state (Infrastructure distribution).
- **Format:** Outputs to standard CSV for downstream research and statistical analysis.
