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
Every agent contains a massive **Deep Neural Network** with two hidden layers (up to 32 nodes each) evaluating **40 distinct sensory inputs** (including Recurrent Memory, Look-Ahead Vision, Local Market Prices, Encumbrance, Crowding, Health, Food, Age, Gender, and Seasons). The brain evaluates over 2,900 synaptic weights and drives **20 complex output intents** (Turn, Speed, Share, Reproduce, Attack, Rest, 4 Communication Channels, an active Hebbian Learning intent, 4 Recurrent Memory states, and 4 Economic Trading intents). 

### ⚡ In-Lifetime Neuroplasticity & Memory
- **Hebbian Learning Engine:** Agents don't just rely on Darwinian genetics; they can learn on the fly. By firing a specific "Learn Intent" node, an agent triggers an active Hebbian gradient update inside the compute shader, dynamically rewiring its own synaptic weights based on real-time environmental context.
- **Recurrent Memory Loops:** Agents feature 4 dedicated abstract memory channels. What they output to these memory states in one frame is fed directly back into their sensory inputs on the next, allowing them to recall context (like the direction of a shoreline or the location of an attacker).

### 🌍 Procedural Topography & 4D Wrapping
The environment is generated using Fractal Brownian Motion (FBM) layered over Perlin noise. 
- **Seamless Wrapping:** 2D map coordinates are mapped to 4D mathematical angles, guaranteeing that moving off the right edge of the map wraps perfectly to the left edge like a true globe.
- **Topological Contours:** The generator extracts exact heightmap elevations and visualizes them using dynamic contour lines on the rendered texture.

### 🗺️ Pheromone Grid, Spatial Awareness & Communication
The map doesn't just store resources—it acts as a biological grid. As agents traverse the tiles, they leave behind continuously decaying "pheromone" traces of their speed, community-sharing intent, aggression, and pregnancy status.
- **Pseudo-Communication:** Agents feature 4 dedicated abstract output channels (`comm1..4`). These signals mix directly into the tile's pheromones, which are then read by neighboring agents on the next frame. The neural networks must autonomously figure out how to invent and decode their own localized languages!

### 💹 Localized AMM Economies & Trade
The simulation implements an Automated Market Maker (AMM) style liquidity pool on every single tile, separating physical **Food** from weightless **Wealth** (USD).
- **Micro-Economies:** Agents read the local `Ask` and `Bid` prices of the cell they stand on, and can output their own intended prices alongside a `Buy` or `Sell` intent.
- **Capitalism & Survival:** Because hoarding heavy physical food severely encumbers movement, agents are organically incentivized to invent commerce—farming food, carrying it to a profitable market tile, selling it for weightless USD, and using that Wealth to pay for boat travel or reproduction without being crushed by weight penalties.

### ⛰️ Advanced Terrain Physics & Resource Mechanics
Agents do not just walk freely; the environment fights back.
- **Elevation & Seasons:** Agents evaluate the topographical slope of the terrain. Walking uphill severely slows movement. Additionally, a global seasonal clock dictates temperatures. Poles and high elevations are freezing, burning agent calories exponentially faster.
- **Hydration & Satiation:** Biological needs are split. Agents must gather Food from the land and return to the coastline to drink Water. 
- **Ocean Traversal:** Deep water is impassable unless an agent has passively gathered enough resources on land to overcome the "boat threshold," allowing them to cross oceans.
- **Encumbrance & Crowding:** Inventory represents physical weight. Hoarding hundreds of kilograms of food severely encumbers agents, slowing their movement speed. Additionally, high populations on a single tile create a physical crowding penalty, organically forcing herds to spread out.

### 🧬 Biological Lifecycle & Genetics
Agents are subject to the harsh realities of life mapped to a realistic timeline (Years/Months). They constantly burn baseline calories to survive, and running depletes Stamina, forcing them to rest. If they run out of resources, they will starve and eventually die. 
- **Combat & Parasitism:** Agents can evolve to output an "Attack" intent, actively stealing food from abstract populations on their current tile, simulating predator/prey dynamics.
- **Sexual Reproduction & Gestation:** Agents possess a male/female gender and must reach puberty to mate. If a healthy Male and Female mate, the female becomes pregnant, entering a Gestation period where she moves slower and consumes significantly more food/water before birthing the genetically crossed child.
- **Extinction Founder System:** If an entire generation goes extinct, the simulation doesn't just throw away the progress. It sorts the dead population by age, extracts the top 8 longest-surviving "Founders", and repopulates the new world with 4,000 slightly mutated descendants of those evolutionary champions.

### ⚙️ Real-Time Configuration (`sim_config.json`)
On its first run, the simulation generates a `sim_config.json` file, mapping the environment to realistic metrics (e.g., 1 Resource = $1 USD, 1 Tick = 1 Minute).
You can freely tweak base speeds, climbing penalties, boat costs, reproduction costs, and more without recompiling the project.

### 📊 Real-Time Telemetry
The `macroquad` UI tracks engine performance precisely, displaying:
- Live Population Counts
- Compute Latency (ms per loop)
- Dynamic Simulation Speed Multipliers
- Formatted Biological Timeline
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
- **Mouse Scroll Wheel:** Zoom in and out (when Inspector is closed)
- **Spacebar:** Pause / Resume the simulation
- **R Key:** Open Visuals Panel to toggle map views (Resources, Market Prices, Age, Gender, Pregnancy overlays)
- **TAB Key:** Open the Live Inspector. Click an agent's row to inspect its live Neural Network Heatmap, or click `[Locate]` to lock the camera and open the side-panel Agent Tracker.
- **Up Arrow:** Exponentially increase simulation speed (compute loops per frame)
- **Down Arrow:** Exponentially decrease simulation speed

## Dependencies

* `macroquad` - Hardware-accelerated 2D UI and input handling
* `wgpu` - Safe, cross-platform graphics and compute API (Vulkan backend)
* `bytemuck` - Raw memory casting for CPU-to-GPU structuring
* `noise` - Procedural 4D terrain generation
* `rand` - Mathematical RNG utilities
* `pollster` - Synchronous blocking for GPU initialization

## Citation & Academic Use

This project is an independent research initiative exploring neural plasticity and evolutionary dynamics in GPU-accelerated environments.

If you utilize this engine, its neural architecture, or the simulation logic in an academic or professional capacity, please cite the work as follows:

Suthar, V. H. (2026). Grand Sim Pro: A GPGPU Framework for Evolutionary Agent-Based Modeling. GitHub Repository. https://github.com/GoVed/grand-sim-pro

## AI Assistance Acknowledgment

In accordance with emerging standards for transparency in software development and academic research:

This project was developed with the assistance of Google's Gemini AI for code generation, refactoring, and architectural brainstorming. All AI-generated outputs were rigorously reviewed, tested, and guided by the human author to ensure strict GPU memory alignment, computational accuracy, and alignment with the project's core research objectives. The author assumes full responsibility for the final codebase, architecture, and simulation logic.