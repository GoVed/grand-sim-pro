# Grand Sim Pro

Grand Sim Pro is a massively parallel, GPU-accelerated genetic survival simulator. It leverages modern Rust, the `wgpu` graphics API, and Compute Shaders to simulate thousands of autonomous neural-network-driven agents in real-time. 

By offloading the heaviest computational workloads directly to the GPU's VRAM, the engine bypasses traditional CPU bottlenecks, allowing for the simulation of complex ecosystems, terrain physics, and survival mechanics at blistering speeds.

## Architecture

This project uses a highly optimized **Hybrid CPU-GPU Compute Engine**:

- **Decoupled Threading:** The application splits into two entirely independent timelines. The UI thread runs a silky smooth 60 FPS viewport using `macroquad`, while a dedicated background thread dispatches heavy simulation workloads as fast as the hardware allows.
- **WGSL Compute Shaders (`wgpu`):** Instead of looping through agents sequentially on the CPU, the simulation math is written in WebGPU Shading Language (`sim.wgsl`). The GPU executes neural network evaluations, physics calculations, and terrain collisions for every single agent simultaneously.
- **Memory Synchronization:** Structs strictly formatted in memory via `bytemuck` are safely shuttled across the PCIe bus, guaranteeing precise alignment between the Rust CPU state and the WGSL GPU state.

## Core Features

### 🧠 GPU-Accelerated Neural Networks
Every agent contains a multi-layer neural network evaluating spatial inputs, topographical heightmaps, and biases. All matrix multiplications and `tanh` activation functions are resolved instantly across thousands of GPU cores.

### 🌍 Procedural Topography & 4D Wrapping
The environment is generated using Fractal Brownian Motion (FBM) layered over Perlin noise. 
- **Seamless Wrapping:** 2D map coordinates are mapped to 4D mathematical angles, guaranteeing that moving off the right edge of the map wraps perfectly to the left edge like a true globe.
- **Topological Contours:** The generator extracts exact heightmap elevations and visualizes them using dynamic contour lines on the rendered texture.

### ⛰️ Advanced Terrain Physics & Resource Mechanics
Agents do not just walk freely; the environment fights back.
- **Elevation Penalties:** Agents evaluate the topographical slope of the terrain. Walking uphill severely slows movement and dramatically drains their gathered resources, while walking downhill provides a speed boost.
- **Ocean Traversal:** Deep water is impassable unless an agent has passively gathered enough resources on land to overcome the "boat threshold," allowing them to cross oceans at a high resource cost.

### 📊 Real-Time Telemetry
The `macroquad` UI tracks engine performance precisely, displaying:
- Live Population Counts
- Compute Latency (ms per loop)
- Dynamic Simulation Speed Multipliers
- Average FPS & 1% Lows

## Prerequisites

To build and run this project natively on your machine, you need:
- Rust
- A Vulkan/Metal/DX12 compatible graphics card (AMD, NVIDIA, or Apple Silicon)

*(Note: If running on Linux via a sandboxed environment like Flatpak, ensure you have exposed X11/Wayland display permissions).*

## Running the Simulation

Because the simulation runs as a native application, you can compile and launch it with full optimizations using a single command:

```bash
cargo run --release
```

### Controls

- **Mouse Left Click & Drag:** Pan the camera
- **Mouse Scroll Wheel:** Zoom in and out
- **Spacebar:** Pause / Resume the simulation
- **Up Arrow:** Exponentially increase simulation speed (compute loops per frame)
- **Down Arrow:** Exponentially decrease simulation speed

## Dependencies

* `macroquad` - Hardware-accelerated 2D UI and input handling
* `wgpu` - Safe, cross-platform graphics and compute API (Vulkan backend)
* `bytemuck` - Raw memory casting for CPU-to-GPU structuring
* `noise` - Procedural 4D terrain generation
* `rand` - Mathematical RNG utilities
* `pollster` - Synchronous blocking for GPU initialization