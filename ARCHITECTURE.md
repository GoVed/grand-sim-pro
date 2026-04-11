# Architecture & Implementation Detail

This document outlines the technical design decisions and internal workings of the Grand Sim Pro engine.

## 1. Hybrid CPU-GPU Compute Engine
The simulation utilizes a "Think on GPU, Synchronize on CPU" pattern to achieve massive parallelism.

### Memory Alignment & Synchronization
- **Strict Alignment:** All data structures shared with the GPU (`src/config.rs`, `src/agent.rs`) are marked with `#[repr(C)]` and use `bytemuck` for zero-copy casting. Memory is strictly 16-byte aligned to match WGSL requirements.
- **Buffer Reuse:** To avoid per-frame allocation overhead, `SimulationManager` maintains reusable `Vec` and `HashMap` buffers for genetics and birth processing.
- **Throttled Fetching:** While the GPU computes at high frequency, agent state is fetched back to the CPU at ~60Hz to conserve PCIe bandwidth.

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
The WGSL kernel (`src/sim.wgsl`) is the "hot loop" of the project.
- **Spatial Awareness:** LiDAR vision is simulated by sampling the pheromone and elevation maps in a 3x3 grid around each agent.
- **Neural Inference:** Hidden layer activations use `tanh` or `ReLU` variants implemented directly in the shader.
- **Physics:** Longitude convergence and spherical wrapping are handled mathematically at the physics step to simulate a 3D globe on a 2D array.
