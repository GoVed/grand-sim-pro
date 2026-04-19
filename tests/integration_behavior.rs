/*
 * Grand Sim Pro: Integration Tests for Agent Behavior
 */

use world_sim::agent::{Person, NUM_OUTPUTS};
use world_sim::config::SimConfig;
use world_sim::environment::{CellState};
use world_sim::gpu_engine::GpuEngine;

#[test]
fn test_agent_behavior_integration() {
    let mut config = SimConfig::default();
    config.world.map_width = 10;
    config.world.map_height = 10;
    config.sim.agent_count = 2; 
    
    let mut states = Vec::new();
    let mut genetics = Vec::new();
    
    // --- Agent 0: The "Mover" ---
    let mut p0 = Person::new(5.0, 5.0, 0, &config);
    p0.state.age = config.bio.puberty_age + 100.0; // Adult
    p0.state.food = 1000.0;
    p0.state.water = 10.0;
    p0.state.stamina = 100.0;
    p0.state.heading = 0.0;
    
    p0.genetics.w1_weights.fill(0.0);
    p0.genetics.w1_indices.fill(0);
    p0.genetics.w2.fill(0.0);
    p0.genetics.w3.fill(0.0);
    p0.genetics.w1_weights[0] = 1.0; 
    p0.genetics.w1_indices[0] = 0;
    p0.genetics.w2[0] = 1.0;
    p0.genetics.w3[0 * NUM_OUTPUTS + 0] = 1.0; // Turn
    p0.genetics.w3[0 * NUM_OUTPUTS + 1] = 1.0; // Speed
    
    // --- Agent 1: The "Static Observer" ---
    let mut p1 = Person::new(5.1, 5.1, 1, &config);
    p1.state.age = config.bio.puberty_age + 100.0;
    
    states.push(p0.state);
    states.push(p1.state);
    genetics.push(p0.genetics);
    genetics.push(p1.genetics);
    
    let heights = vec![0.0; 100];
    let cells = vec![CellState::default(); 100];
    
    let gpu = GpuEngine::new(&states, &genetics, &heights, &cells, &config);
    
    // 2. Execute Ticks
    gpu.compute_ticks(1);
    
    // 3. Verify Results
    let (after_states, _) = gpu.fetch_agents();
    let a0 = &after_states[0];
    
    println!("Agent 0: Final Pos ({}, {})", a0.x, a0.y);
    println!("Agent 0: Final Food {}", a0.food);
    println!("Agent 0: Target Identity Feature 1: {}", a0.nearest_id_f1);

    // A. Verify Metabolism
    assert!(a0.food < 1000.0, "Metabolism: Food should decrease");
    assert!(a0.water < 10.0, "Metabolism: Water should decrease");
    assert!(a0.age > p0.state.age, "Metabolism: Age should increase");

    // B. Verify Movement & Turn
    assert!(a0.x != 5.0 || a0.y != 5.0, "Physics: Agent 0 should have moved");
    assert!(a0.heading != 0.0, "Physics: Agent 0 should have turned");
    
    // C. Verify Identity recognition
    assert!(a0.id_f1 != 0.0, "Identity: Self-features should be initialized");
    assert!(a0.nearest_id_f1 != 0.0, "Identity: Should recognize nearby agent");
}
