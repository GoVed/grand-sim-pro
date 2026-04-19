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
    
    // D. Verify Identity recognition
    assert!(a0.id_f1 != 0.0, "Identity: Self-features should be initialized");
    assert!(a0.nearest_id_f1 != 0.0, "Identity: Should recognize nearby agent");
    }

    #[test]
    fn test_founder_logic_initialization() {
    use world_sim::simulation::SimulationManager;
    use world_sim::agent::{NUM_INPUTS};

    let mut config = SimConfig::default();
    config.sim.agent_count = 10;
    config.genetics.random_spawn_percentage = 0.2; // 2 random, 8 from founders

    // Create a specific "Super-Mover" founder
    let mut founder = Person::new(0.0, 0.0, 0, &config);
    // Bias Input 0 to Output 1 (Speed)
    founder.genetics.w1_weights.fill(0.0);
    founder.genetics.w1_weights[0] = 5.0; // Strong weight
    founder.genetics.w1_indices[0] = 0;
    founder.genetics.w3.fill(0.0);
    founder.genetics.w3[0 * NUM_OUTPUTS + 1] = 5.0; 

    let founders = vec![founder];
    let sim = SimulationManager::new(100, 100, 42, 10, &config, founders);

    assert_eq!(sim.states.len(), 10);
    assert_eq!(sim.genetics.len(), 10);

    // Verify that most agents inherited the strong weight at index 0
    let mut inherited_count = 0;
    for g in &sim.genetics {
        // Check if W1 index 0 points to input 0 and has non-zero weight
        if g.w1_indices[0] == 0 && g.w1_weights[0].abs() > 1.0 {
            inherited_count += 1;
        }
    }

    // 8 should be descendants of the founder
    assert!(inherited_count >= 7, "Most agents should have inherited founder traits. Found: {}", inherited_count);

    // Verify NUM_INPUTS is consistent
    assert_eq!(NUM_INPUTS, 420);
    }

    #[test]
    fn test_cnn_kernel_inheritance() {
    let config = SimConfig::default();
    let mut parent = Person::new(0.0, 0.0, 0, &config);

    // Set unique kernels for parent
    for (i, val) in parent.genetics.cnn_kernels.iter_mut().enumerate() {
        *val = i as f32 * 0.1;
    }

    let child = parent.clone_as_descendant(1.0, 1.0, 1, 0.0, 0.0, &config); // 0 mutation

    // Child should have identical kernels
    for i in 0..72 {
        assert_eq!(child.genetics.cnn_kernels[i], parent.genetics.cnn_kernels[i], "CNN Kernel {} should be inherited", i);
    }
    }
