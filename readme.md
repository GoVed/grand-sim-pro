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
Every agent contains an advanced, dynamic multi-layer neural network evaluating 12 distinct sensory inputs (including Spatial location, Cell Pheromones, personal Health, Inventory, Age, and Gender). The brain can organically mutate and grow up to 32 hidden nodes. All matrix multiplications and `tanh` activation functions are resolved instantly across thousands of GPU cores.

### 🌍 Procedural Topography & 4D Wrapping
The environment is generated using Fractal Brownian Motion (FBM) layered over Perlin noise. 
- **Seamless Wrapping:** 2D map coordinates are mapped to 4D mathematical angles, guaranteeing that moving off the right edge of the map wraps perfectly to the left edge like a true globe.
- **Topological Contours:** The generator extracts exact heightmap elevations and visualizes them using dynamic contour lines on the rendered texture.

### 🗺️ Pheromone Grid & Spatial Awareness
The map doesn't just store resources—it acts as a biological grid. As agents traverse the tiles, they leave behind continuously decaying "pheromone" traces of their speed, community-sharing intent, and desire to reproduce. Agents sense these traces, allowing them to track other groups or actively search for mates without expensive CPU-side collision loops.

### ⛰️ Advanced Terrain Physics & Resource Mechanics
Agents do not just walk freely; the environment fights back.
- **Elevation Penalties:** Agents evaluate the topographical slope of the terrain. Walking uphill severely slows movement and dramatically drains their gathered resources, while walking downhill provides a speed boost.
- **Ocean Traversal:** Deep water is impassable unless an agent has passively gathered enough resources on land to overcome the "boat threshold," allowing them to cross oceans at a high resource cost.

### 🧬 Biological Lifecycle & Genetics
Agents are subject to the harsh realities of life. They constantly burn baseline resources to survive and age over time. If they run out of resources, they will starve and eventually die, leaving behind a corpse. 
- **Sexual Reproduction:** Agents possess a male/female gender. If a healthy Male and Female stand on the same tile and both signal a high neural desire to reproduce, they will mate, pool their resources, and spawn a child with a genetic crossover of their neural weights.
- **Extinction Founder System:** If an entire generation goes extinct, the simulation doesn't just throw away the progress. It sorts the dead population by age, extracts the top 8 longest-surviving "Founders", and repopulates the new world with 4,000 slightly mutated descendants of those evolutionary champions.

### ⚙️ Real-Time Configuration (`sim_config.json`)
On its first run, the simulation generates a `sim_config.json` file, mapping the environment to realistic metrics (e.g., 1 Resource = $1 USD, 1 Tick = 1 Minute).
You can freely tweak base speeds, climbing penalties, boat costs, reproduction costs, and more without recompiling the project.

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
- **R Key:** Toggle Logarithmic Resource View (depleted paths glow red, abundant resources glow green)
- **Up Arrow:** Exponentially increase simulation speed (compute loops per frame)
- **Down Arrow:** Exponentially decrease simulation speed

## Dependencies

* `macroquad` - Hardware-accelerated 2D UI and input handling
* `wgpu` - Safe, cross-platform graphics and compute API (Vulkan backend)
* `bytemuck` - Raw memory casting for CPU-to-GPU structuring
* `noise` - Procedural 4D terrain generation
* `rand` - Mathematical RNG utilities
* `pollster` - Synchronous blocking for GPU initialization