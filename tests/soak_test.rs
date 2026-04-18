use world_sim::simulation::SimulationManager;
use world_sim::config::SimConfig;
use world_sim::shared::SharedData;
use world_sim::gpu_engine::GpuEngine;
use world_sim::sim_thread;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[tokio::test]
async fn test_soak_stability() {
    let mut config = SimConfig::default();
    config.world.map_width = 800;
    config.world.map_height = 600;
    config.sim.agent_count = 2000;
    config.sim.founder_count = 100;
    config.telemetry.enabled = 0; // Disable telemetry to focus purely on GPU stability

    let sim = SimulationManager::new(config.world.map_width, config.world.map_height, 12345, config.sim.agent_count, &config, Vec::new());
    let shared = Arc::new(Mutex::new(SharedData {
        sim,
        config: config.clone(),
        last_saved_config: config.clone(),
        is_paused: false,
        restart_message_active: false,
        ticks_per_loop: 200, // HIGH SPEED
        total_ticks: 0,
        cumulative_ticks: 0,
        last_telemetry_tick: 0,
        cumulative_births: 2000,
        cumulative_deaths: 0,
        last_compute_time_micros: 0,
        ticks_per_second: 0.0,
        generation_survival_times: Vec::new(),
    }));

    let states = shared.lock().unwrap().sim.states.clone();
    let genetics = shared.lock().unwrap().sim.genetics.clone();
    let heights = shared.lock().unwrap().sim.env.height_map.clone();
    let cells = shared.lock().unwrap().sim.env.map_cells.clone();
    let gpu = Arc::new(GpuEngine::new(&states, &genetics, &heights, &cells, &config));

    // Spawn sim thread in headless mode for maximum stress
    sim_thread::spawn(shared.clone(), gpu.clone(), true);

    println!("--- Starting 10-Generation Soak Test ---");
    let mut generations = 0;
    let max_generations = 10;
    let mut timeout_secs = 60; // 1 minute limit

    while generations < max_generations && timeout_secs > 0 {
        tokio::time::sleep(Duration::from_secs(1)).await;
        let (current_gen, ticks) = {
            let data = shared.lock().unwrap();
            (data.generation_survival_times.len(), data.cumulative_ticks)
        };
        
        if current_gen > generations {
            println!("Soak Progress: Reached Generation {} (Total Ticks: {})", current_gen, ticks);
            generations = current_gen;
        }
        timeout_secs -= 1;
    }

    let final_gen = { shared.lock().unwrap().generation_survival_times.len() };
    assert!(final_gen >= 2, "Simulation did not progress through enough generations. Stuck at {}", final_gen);
    println!("Soak Test Successful: {} generations survived without GPU crash.", final_gen);
}
