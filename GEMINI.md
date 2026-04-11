# Grand Sim Pro - Project Status & Roadmap

## 1. Architectural Overview
- **Core Engine:** GPGPU-accelerated evolutionary simulation using `wgpu` (WGSL).
- **UI Framework:** Macroquad-based immediate mode GUI, following a **Logic-View Separation** pattern.
- **Testing Strategy:** 
    - **Performance:** Release-mode benchmarks in `tests/performance.rs`.
    - **UI Logic:** Headless verification of situational probing, layout, and configuration in `tests/ui_verification.rs`.
    - **No-Panic Policy:** `macroquad` is strictly isolated from logic to prevent thread-local context panics during `cargo test`.
    - **Coverage Mandate:** UI Logic must maintain >90% code coverage. All new UI features MUST include corresponding tests in `tests/ui_verification.rs`.

## 2. Key Modules
- `src/ui_logic.rs`: Pure-logic backend for UI. Handles situational behavior math, configuration filtering, and panel layout calculations.
- `src/ui.rs`: Rendering layer. Maps `ui_logic` types and layout calculations to `macroquad` drawing calls.
- `src/agent.rs`: Contains `Person` struct and `mental_simulation` for neural probing.
- `src/simulation.rs`: Manages the simulation lifecycle and CPU-side buffers.
- `src/sim_thread.rs`: Background simulation thread. Handles high-speed tick computation and state syncing.
- `src/sim.wgsl`: GPU kernel for population dynamics and resource flows.

## 3. Latest Achievements
- **Deep Behavioral Profiling:** Implemented "Situational Probing" bars in the Agent Detail view. Bars are correctly sized and spaced to prevent panel overflow.
- **Advanced Config Panel:** Searchable settings with real-time change highlighting. Supports sub-string matching and instant visual feedback.
- **Improved Data Visualization:** `Generation Survival Time` graph now includes proper axes, labels, and titles.
- **Day/Night Cycle Fixed:** Resolved issue where time was stuck; `current_tick` now increments correctly across all simulation modes.
- **Testable UI Logic:** Expanded verification suite to cover layout and filtering, ensuring structural integrity without an active window.

## 4. Immediate Roadmap
- [ ] Extend behavioral simulation with more probes (e.g., "In Combat", "Aging").
- [ ] Add "Tribe" visual mode to show emergent social clusters.
- [ ] Update `readme.md` with new high-resolution screenshots of the Neural Profile UI and Graph.
