<div align="center">
  <h1>🌍 Grand Sim Pro</h1>
  <p><b>A massively parallel, GPU-accelerated genetic survival simulator.</b></p>
</div>

Grand Sim Pro leverages modern Rust, the `wgpu` graphics API, and Compute Shaders to simulate thousands of autonomous neural-network-driven agents in real-time. 

By offloading the heaviest computational workloads directly to the GPU's VRAM, the engine bypasses traditional CPU bottlenecks, allowing for the simulation of complex ecosystems, terrain physics, and survival mechanics at blistering speeds.

![Main UI showing the simulation running with procedural terrain, biomes, and a massive agent population](readme_imgs/main_ui.png)

## 🏗️ Architecture
The engine is built on a high-performance **Hybrid CPU-GPU Compute Engine**. Detailed technical specifications regarding threading, memory alignment, and the Logic-View UI pattern can be found in [ARCHITECTURE.md](./ARCHITECTURE.md).

---

## ✨ Core Features

### 🧠 Direct Neural Influence Mapping
<p align="center">
  <img src="readme_imgs/agent_info_ui.png" alt="Neural Influence UI" width="80%">
</p>

Every agent contains a massive **Deep Neural Network** evaluating **160 sensory inputs** against **31 complex output intents**. 

Instead of viewing opaque hidden layers, the Inspector provides a **Linearized Sensory-Behavioral Influence Map**. This bipartite graph computes the direct impact of every sensory input (e.g., "Hunger", "Crowding") on every behavioral output (e.g., "Attack", "Build Road") by multiplying the weight matrices ($W_3 \times W_2 \times W_{sparse1}$).

- **Smart Filtering:** Automatically identifies and displays the top 30 most "influential" inputs to maintain visual clarity.
- **Interactive Hover:** Hover over any node to isolate its specific neural pathways and see the exact numerical influence values.
- **Direct Insight:** Instantly see if an agent is innately aggressive, altruistic, or industrious based on its neural wiring.

### ⚡ In-Lifetime Neuroplasticity & Memory
- **Hebbian Learning Engine:** Agents don't just rely on Darwinian genetics; they can learn on the fly. By firing a specific "Learn Intent" node, an agent triggers an active Hebbian gradient update inside the compute shader, dynamically rewiring its own synaptic weights based on real-time environmental context.
- **Recurrent Memory Loops:** Agents feature 8 dedicated abstract memory channels. What they output to these memory states in one frame is fed directly back into their sensory inputs on the next, allowing them to recall context (like the direction of a shoreline or the location of an attacker).
- **Sleep State & Dreaming:** When an agent rests or passes out from exhaustion, they enter a realistic sleep state. Because the Hebbian learning intent can remain active, agents can literally *dream*—consolidating memories and updating synapses while asleep!

### 🧪 Situational Behavioral Probing
The Inspector features a behavioral simulation panel that executes "forward passes" of an agent's neural network against hypothetical archetypes: *Crowded*, *Starving*, and *Prosperous*. This provides deep behavioral insights into an agent's innate propensities before the situation actually arises.

### ⚡ High-Performance GPGPU Parallelism
Grand Sim Pro is built for extreme scale, utilizing modern GPGPU techniques to bypass CPU bottlenecks:
- **Spatial Sorting:** Agents are sorted by map position on the CPU before every GPU dispatch, ensuring optimal memory locality.
- **LDS (Local Device Storage) Caching:** GPU workgroups cooperatively load map "patches" into fast on-chip memory, reducing vision-sampling latency by up to 8x.
- **Pointer-Style Memory Access:** Minimal register pressure architecture prevents VRAM "spilling," allowing for massive agent populations (10,000+) on consumer hardware.
- **Atomic Micro-Economics:** Resource gathering and trading utilize hardware-level atomics for thread-safe, high-frequency interaction.

### 📊 Research Data Export Pipeline (Telemetry)
Grand Sim Pro now includes a high-fidelity **CSV Telemetry Pipeline** designed for longitudinal research.
- **Automated Logging:** Periodically exports 14+ population and economic metrics (Population, Avg Age, Health, Wealth, Infrastructure count, etc.).
- **Research Ready:** Data is formatted for direct import into R, Python (Pandas), or Excel for statistical analysis of evolutionary drift and economic emergence.
- **Configurable:** Adjust the `export_interval_ticks` and toggle the system via the Live Configuration panel.

### 🌍 Procedural Continents & Spherical Projection
- **3D Spherical Mapping:** 2D map coordinates are mapped to a mathematically perfect 3D spherical projection. landmasses generate North and South poles where longitude shifts 180 degrees seamlessly.
- **Dynamic Biomes:** Moisture noise layers interact with global latitude temperatures to paint diverse biomes: Snow, Tundra, Deserts, Savannas, Jungles, and Forests.
- **Topological Contours:** The generator extracts heightmap elevations and visualizes them using dynamic contour lines and 2.5D directional shading.

### 💹 Localized AMM Economies & Trade
The simulation implements an **Automated Market Maker (AMM)** style liquidity pool on every tile, separating physical **Food** from weightless **Wealth** (USD).
- **Micro-Economies:** Agents read local `Ask`/`Bid` prices and can output `Buy`/`Sell` intents.
- **Capitalism & Survival:** Because physical food causes encumbrance and rots, agents are incentivized to farm food, sell it for weightless USD, and use that wealth for life-extending healthcare or reproduction.

### 🏗️ Civilization & Infrastructure
Agents can expend wealth to construct mutually exclusive infrastructure: **Roads, Houses, Farms, and Granaries**.
- Structures grant massive bonuses (e.g., 2x movement speed on roads, 90% rot reduction in granaries).
- **Decay & Maintenance:** Infrastructure is subjected to weathering and wear-and-tear, requiring continuous economic maintenance.

### 🧬 Biological Lifecycle & Reproduction
- **Age-Based Physical Development:** Newborn agents start with significantly reduced physical capabilities (10% speed, 20% stamina). Their mobility and endurance scale linearly as they mature toward puberty, accurately modeling the vulnerability of youth.
- **Combat & Parasitism:** Agents can evolve to "Attack," actively stealing resources from others.
- **Sexual Reproduction:** Agents possess male/female genders and must reach puberty to mate. Females enter a Gestation period with specific encumbrance and metabolic costs before birthing genetically crossed offspring.
- **Extinction Founder System:** If a generation goes extinct, the simulation extracts the longest-surviving "Founders," generates a new world map, and repopulates it with their mutated descendants.

---

### ⚙️ Deep Configuration Engine
Press **C** to open the Live Configuration Panel. 
- **Full Exposure:** Over 30+ physical parameters are exposed, including world regeneration rates, combat damage, and infrastructure bonuses.
- **Isolated Search:** Find settings by label or internal key. Global hotkeys are disabled while searching to prevent accidental triggers.
- **Interactive Editing:** Fine-tune with +/- buttons and instant change highlighting (Yellow).

### 📊 Evolutionary Telemetry
- **Survival Graph:** Tracks generational survival time with a Y-axis formatted in **Years and Months**.
- **Real-Time Clock:** A functional day/night cycle affects visibility and temperature.
- **Biometric Tracking:** Follow specific agents to watch their health, wealth, and inventory levels in real-time.

---

## 🧪 Testing & Performance

### Unit & UI Verification Tests
To run the standard unit tests and the headless UI logic verification suite:
```bash
cargo test
```
The UI verification suite ensures that situational probing math, direct influence calculations, and panel layouts are structurally sound and regression-free.

### Performance Benchmarking
To run the performance suite and log results into `PERFORMANCE_LOG.md`:
```bash
bash scripts/test_perf.sh
```

---

## 🚀 Running the Simulation

```bash
cargo run --release
```

### Controls

- **Mouse Left Click & Drag:** Pan the camera
- **Mouse Scroll Wheel:** Zoom in and out
- **Spacebar:** Pause / Resume (Disabled when Config is open)
- **S Key:** Save top agents (Disabled when Config is open)
- **C Key:** Toggle Configuration Panel
- **G Key:** Toggle Generational Survival Graph
- **R Key:** Open Visuals Panel (Toggle Resources, Health, Age, Tribes views)
- **T/N/I/W Keys:** Instant Map Toggles (Temperature, Day/Night, Identity, Water)
- **TAB Key:** Open Agent Inspector
- **Up/Down Arrows:** Exponentially adjust simulation speed

## Citation & Academic Use

This project is an independent research initiative exploring neural plasticity and evolutionary dynamics in GPU-accelerated environments.

If you utilize this engine, its neural architecture, or the simulation logic in an academic or professional capacity, please cite the work as follows:

Suthar, V. H. (2026). Grand Sim Pro: A GPGPU Framework for Evolutionary Agent-Based Modeling. GitHub Repository. https://github.com/GoVed/grand-sim-pro

## AI Assistance Acknowledgment
This project was developed with the assistance of Google's Gemini AI. All AI-generated outputs were rigorously reviewed and tested by the human author. 
