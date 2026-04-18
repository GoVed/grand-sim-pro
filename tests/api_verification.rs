use world_sim::simulation::SimulationManager;
use world_sim::config::SimConfig;
use world_sim::shared::SharedData;
use world_sim::gpu_engine::GpuEngine;
use world_sim::api;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[tokio::test]
async fn test_api_status_sync() {
    let config = SimConfig::default();
    let sim = SimulationManager::new(100, 100, 12345, 100, &config, Vec::new());
    let shared = Arc::new(Mutex::new(SharedData {
        sim,
        config: config.clone(),
        last_saved_config: config.clone(),
        is_paused: false,
        restart_message_active: false,
        ticks_per_loop: 5,
        total_ticks: 1000,
        cumulative_ticks: 1000,
        last_telemetry_tick: 0,
        cumulative_births: 100,
        cumulative_deaths: 0,
        last_compute_time_micros: 0,
        ticks_per_second: 100.0,
        generation_survival_times: Vec::new(),
    }));
    
    // We need a dummy GPU engine
    let states = shared.lock().unwrap().sim.states.clone();
    let genetics = shared.lock().unwrap().sim.genetics.clone();
    let heights = shared.lock().unwrap().sim.env.height_map.clone();
    let cells = shared.lock().unwrap().sim.env.map_cells.clone();
    let gpu = Arc::new(GpuEngine::new(&states, &genetics, &heights, &cells, &config));

    let shared_api = shared.clone();
    let gpu_api = gpu.clone();
    
    // Start API on a random port
    tokio::spawn(async move {
        api::start_api(shared_api, gpu_api, 3030).await;
    });

    // Give it a moment to start
    tokio::time::sleep(Duration::from_millis(500)).await;

    let client = reqwest::Client::new();
    let res = client.get("http://127.0.0.1:3030/status")
        .send()
        .await
        .expect("Failed to send request");

    assert_eq!(res.status(), 200);
    let status: serde_json::Value = res.json().await.expect("Failed to parse JSON");
    
    assert_eq!(status["ticks"], 1000);
    assert_eq!(status["population"], 100);

    // Now modify state and check if API reflects it
    {
        let mut data = shared.lock().unwrap();
        data.total_ticks = 2000;
        data.sim.states[0].health = -1.0; // Kill one
    }

    let res2 = client.get("http://127.0.0.1:3030/status")
        .send()
        .await
        .expect("Failed to send request");
    
    let status2: serde_json::Value = res2.json().await.expect("Failed to parse JSON");
    assert_eq!(status2["ticks"], 2000);
    assert_eq!(status2["population"], 99);
}
