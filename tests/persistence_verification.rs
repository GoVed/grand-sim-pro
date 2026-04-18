use world_sim::simulation::SimulationManager;
use world_sim::config::SimConfig;
use world_sim::shared::{SharedData, FullState};
use std::io::Read;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;

#[test]
fn test_persistence_roundtrip() {
    let config = SimConfig::default();
    let map_width = 100;
    let map_height = 100;
    let agent_count = 50;
    let random_seed = 12345;

    let sim = SimulationManager::new(map_width, map_height, random_seed, agent_count, &config, Vec::new());
    
    let shared = SharedData {
        sim,
        config: config.clone(),
        last_saved_config: config.clone(),
        is_paused: false,
        restart_message_active: false,
        ticks_per_loop: 5,
        total_ticks: 1234,
        cumulative_ticks: 5678,
        last_telemetry_tick: 5000,
        cumulative_births: 10000,
        cumulative_deaths: 9000,
        last_compute_time_micros: 5678,
        ticks_per_second: 60.0,
        generation_survival_times: vec![1000, 2000, 3000],
    };

    let original_state = FullState { shared };

    // Serialize to Bincode + Gzip
    let mut buffer = Vec::new();
    {
        let mut encoder = GzEncoder::new(&mut buffer, Compression::default());
        bincode::serialize_into(&mut encoder, &original_state).expect("Failed to serialize");
        encoder.finish().expect("Failed to finish compression");
    }

    // Deserialize from Bincode + Gzip
    let mut decoder = GzDecoder::new(&buffer[..]);
    let mut decompressed_data = Vec::new();
    decoder.read_to_end(&mut decompressed_data).expect("Failed to decompress");
    
    let loaded_state: FullState = bincode::deserialize_from(&decompressed_data[..]).expect("Failed to deserialize");

    // Compare
    assert_eq!(original_state, loaded_state, "Loaded state does not match original state");
    
    // Additional granular checks for peace of mind
    assert_eq!(original_state.shared.total_ticks, loaded_state.shared.total_ticks);
    assert_eq!(original_state.shared.sim.states.len(), loaded_state.shared.sim.states.len());
    assert_eq!(original_state.shared.sim.genetics.len(), loaded_state.shared.sim.genetics.len());
    assert_eq!(original_state.shared.sim.env.map_cells.len(), loaded_state.shared.sim.env.map_cells.len());
}

#[test]
fn test_genetic_array_persistence() {
    // Specifically test large arrays in Genetics
    use world_sim::agent::Genetics;
    use world_sim::agent::{W1_SIZE, W2_SIZE, W3_SIZE};
    
    let mut g = Genetics {
        w1_weights: [0.0; W1_SIZE],
        w1_indices: [0; W1_SIZE],
        w2: [0.0; W2_SIZE],
        w3: [0.0; W3_SIZE],
    };
    
    // Set some non-zero values
    g.w1_weights[123] = 0.5;
    g.w1_indices[456] = 42;
    g.w2[789] = -1.2;
    g.w3[1011] = 0.8;
    
    let encoded = bincode::serialize(&g).expect("Failed to serialize genetics");
    let decoded: Genetics = bincode::deserialize(&encoded).expect("Failed to deserialize genetics");
    
    assert_eq!(g.w1_weights[123], decoded.w1_weights[123]);
    assert_eq!(g.w1_indices[456], decoded.w1_indices[456]);
    assert_eq!(g.w2[789], decoded.w2[789]);
    assert_eq!(g.w3[1011], decoded.w3[1011]);
}
