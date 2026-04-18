use world_sim::simulation::SimulationManager;
use world_sim::config::SimConfig;
use world_sim::shared::SharedData;
use world_sim::gpu_engine::GpuEngine;
use world_sim::sim_thread;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[tokio::test]
async fn test_e2e_telemetry_activity() {
    let mut config = SimConfig::default();
    config.world.map_width = 200;
    config.world.map_height = 200;
    config.sim.agent_count = 100;
    config.telemetry.enabled = 1;
    config.telemetry.export_interval_ticks = 10; // Export every 10 ticks for the test
    
    // Ensure we start with a clean slate
    let _ = std::fs::remove_file("telemetry.csv");

    let sim = SimulationManager::new(config.world.map_width, config.world.map_height, 12345, config.sim.agent_count, &config, Vec::new());
    let shared = Arc::new(Mutex::new(SharedData {
        sim,
        config: config.clone(),
        last_saved_config: config.clone(),
        is_paused: false,
        restart_message_active: false,
        ticks_per_loop: 50,
        total_ticks: 0,
        cumulative_ticks: 0,
        last_telemetry_tick: 0,
        cumulative_births: 100,
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

    // Spawn sim thread in headless mode
    sim_thread::spawn(shared.clone(), gpu.clone(), true);

    // Run for a bit
    let mut ticks = 0;
    for _ in 0..100 { 
        tokio::time::sleep(Duration::from_millis(100)).await;
        ticks = shared.lock().unwrap().cumulative_ticks;
        if ticks >= 50 { break; }
    }

    assert!(ticks >= 50, "Simulation did not progress fast enough: {} ticks", ticks);
    
    // Give background telemetry thread a moment to finish writing
    tokio::time::sleep(Duration::from_millis(500)).await;
    
    assert!(std::path::Path::new("telemetry.csv").exists(), "Telemetry file was not created");

    // Read telemetry and verify non-zero social metrics
    let content = std::fs::read_to_string("telemetry.csv").unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert!(lines.len() >= 2, "Telemetry has no data lines");

    let last_line = lines.last().unwrap();
    println!("DEBUG: Raw CSV line: {}", last_line);
    let cols: Vec<&str> = last_line.split(',').collect();
    
    // Header check (ensure columns are where we expect)
    // 0:Gen, 1:Tick, 2:Pop, ... 15:Aggr, 16:Altruism, 17:Ask, 18:Bid
    let aggr: f32 = cols[15].parse().unwrap();
    let altruism: f32 = cols[16].parse().unwrap();
    let ask: f32 = cols[17].parse().unwrap();
    let bid: f32 = cols[18].parse().unwrap();

    println!("E2E Results - Aggr: {}, Altr: {}, Ask: {}, Bid: {}", aggr, altruism, ask, bid);

    // With our "kickstart" bias, these should definitely be non-zero
    assert!(aggr.abs() > 0.0, "Average aggression is still zero");
    assert!(altruism.abs() > 0.0, "Average altruism is still zero");
    assert!(ask > 0.0, "Average ask price is still zero");
    assert!(bid > 0.0, "Average bid price is still zero");

    let _ = std::fs::remove_file("telemetry.csv");
}
