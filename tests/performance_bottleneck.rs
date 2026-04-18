use world_sim::simulation::SimulationManager;
use world_sim::config::SimConfig;
use world_sim::gpu_engine::GpuEngine;
use std::time::Instant;

#[test]
fn test_bottleneck_analysis() {
    let mut config = SimConfig::default();
    config.world.map_width = 800;
    config.world.map_height = 600;
    config.sim.agent_count = 10000;
    
    let states = vec![world_sim::agent::AgentState { id: 0, ..Default::default() }; config.sim.agent_count as usize];
    let genetics = vec![world_sim::agent::Genetics { w1_weights: [0.0; world_sim::agent::W1_SIZE], w1_indices: [0; world_sim::agent::W1_SIZE], w2: [0.0; world_sim::agent::W2_SIZE], w3: [0.0; world_sim::agent::W3_SIZE] }; config.sim.agent_count as usize];
    let heights = vec![0.0f32; (config.world.map_width * config.world.map_height) as usize];
    let cells = vec![world_sim::environment::CellState { res_value: 0, ..Default::default() }; (config.world.map_width * config.world.map_height) as usize];
    
    println!("Initializing GPU Engine...");
    let start_init = Instant::now();
    let gpu = GpuEngine::new(&states, &genetics, &heights, &cells, &config);
    println!("GPU Init took: {:?}", start_init.elapsed());

    // Test 1: Agent Fetch Latency
    println!("Testing Agent Fetch...");
    let start_fetch = Instant::now();
    let _ = gpu.fetch_agents();
    let fetch_time = start_fetch.elapsed();
    println!("fetch_agents took: {:?}", fetch_time);

    // Test 2: Cell Fetch Latency (The suspected bottleneck)
    println!("Testing Cell Fetch...");
    let start_cells = Instant::now();
    let _ = gpu.fetch_cells();
    let cell_fetch_time = start_cells.elapsed();
    println!("fetch_cells took: {:?}", cell_fetch_time);

    // Test 3: Telemetry Calculation Overhead
    let mut sim = SimulationManager::new(config.world.map_width, config.world.map_height, 1, config.sim.agent_count, &config, Vec::new());
    sim.env.map_cells = vec![world_sim::environment::CellState { infra_roads: 100, ..Default::default() }; (config.world.map_width * config.world.map_height) as usize];
    
    println!("Testing Telemetry Logic Overhead...");
    let mut exporter = world_sim::telemetry::TelemetryExporter::new("perf_test_telemetry.csv");
    let start_telemetry = Instant::now();
    let _ = exporter.export(&sim, &config, 0, 0);
    let telemetry_time = start_telemetry.elapsed();
    println!("telemetry.export took: {:?}", telemetry_time);
    
    let _ = std::fs::remove_file("perf_test_telemetry.csv");

    // Threshold checks (typical frame budget is 16ms)
    assert!(fetch_time.as_millis() < 50, "Agent fetch is too slow: {:?}", fetch_time);
    assert!(cell_fetch_time.as_millis() < 100, "Cell fetch is extremely slow: {:?}", cell_fetch_time);
    assert!(telemetry_time.as_millis() < 50, "Telemetry calculation is too slow: {:?}", telemetry_time);
}
