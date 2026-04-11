
use world_sim::simulation::SimulationManager;
use world_sim::config::SimConfig;
use std::time::Instant;

#[test]
fn bench_simulation_step() {
    let mut config = SimConfig::default();
    config.world.map_width = 800;
    config.world.map_height = 600;
    config.sim.agent_count = 10000;
    
    let mut sim = SimulationManager::new(800, 600, 12345, 10000, &config);
    
    let start = Instant::now();
    for _ in 0..500 {
        sim.process_genetics_and_births(&config);
    }
    let duration = start.elapsed();
    
    println!("Processed 500 steps with 10000 agents in {:?}", duration);
    // Ensure it doesn't take more than 5 seconds
    assert!(duration.as_secs() < 5, "Performance is too low: {:?}", duration);
}

#[test]
fn bench_world_generation() {
    let config = SimConfig::default();
    let start = Instant::now();
    let _sim = SimulationManager::new(800, 600, 12345, 1000, &config);
    let duration = start.elapsed();
    println!("World generation (800x600) took {:?}", duration);
    assert!(duration.as_secs() < 10, "World generation is too slow: {:?}", duration);
}

#[test]
fn bench_agent_reproduction() {
    let config = SimConfig::default();
    let mut p1 = world_sim::agent::Person::new(0.0, 0.0, &config);
    let mut p2 = world_sim::agent::Person::new(0.0, 0.0, &config);
    let mut rng = rand::thread_rng();
    
    let start = Instant::now();
    for _ in 0..10000 {
        world_sim::agent::Person::reproduce_sexual_with_rng(&mut p1, &mut p2, 500.0, &mut rng);
    }
    let duration = start.elapsed();
    println!("10000 reproductions in {:?}", duration);
    assert!(duration.as_millis() < 500, "Reproduction is too slow: {:?}", duration);
}
