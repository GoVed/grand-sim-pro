
use world_sim::simulation::SimulationManager;
use world_sim::config::SimConfig;
use std::time::Instant;

#[test]
fn bench_simulation_step() {
    let mut config = SimConfig::default();
    config.world.map_width = 800;
    config.world.map_height = 600;
    config.sim.agent_count = 10000;
    
    let mut sim = SimulationManager::new(800, 600, 12345, 10000, &config, Vec::new());
    
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
    let _sim = SimulationManager::new(800, 600, 12345, 1000, &config, Vec::new());
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

#[test]
fn bench_massive_scale_simulation() {
    let mut config = SimConfig::default();
    config.world.map_width = 1600;
    config.world.map_height = 1200;
    config.sim.agent_count = 50000;
    
    let mut sim = SimulationManager::new(1600, 1200, 12345, 50000, &config, Vec::new());
    
    let start = Instant::now();
    // Simulate 100 steps of birth/genetics logic (CPU side)
    for _ in 0..100 {
        sim.process_genetics_and_births(&config);
    }
    let duration = start.elapsed();
    
    println!("Massive Scale: Processed 100 CPU steps with 50000 agents in {:?}", duration);
    assert!(duration.as_secs() < 10, "Massive scale CPU logic is too slow: {:?}", duration);
}

#[test]
fn bench_high_density_sorting() {
    let mut config = SimConfig::default();
    config.world.map_width = 100;
    config.world.map_height = 100;
    config.sim.agent_count = 20000;
    
    let mut sim = SimulationManager::new(100, 100, 12345, 20000, &config, Vec::new());
    
    let start = Instant::now();
    for _ in 0..100 {
        // Mock spatial sorting as done in sim_thread
        let map_w = config.world.map_width as usize;
        sim.agents.sort_by_key(|a| {
            if a.health <= 0.0 { return usize::MAX; }
            let ty = (a.y as usize).clamp(0, config.world.map_height as usize - 1);
            let tx = (a.x as usize).clamp(0, config.world.map_width as usize - 1);
            ty * map_w + tx
        });
    }
    let duration = start.elapsed();
    
    println!("High Density: 100 Spatial Sorts of 20000 agents in {:?}", duration);
    assert!(duration.as_secs() < 2, "High density sorting is too slow: {:?}", duration);
}
