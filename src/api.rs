/*
 * Grand Sim Pro: A high-performance GPGPU evolutionary agent simulation.
 * Part of an independent research project into emergent biological complexity.
 *
 * Copyright (C) 2026 Ved Hirenkumar Suthar
 * Licensed under the GNU General Public License v3.0 or later.
 * * This software is provided "as is", without warranty of any kind.
 * See the LICENSE file in the project root for full license details.
 */

use axum::{
    routing::{get, post},
    extract::{State, Json},
    Router,
    response::IntoResponse,
    http::StatusCode,
};
use std::sync::{Arc, Mutex};
use crate::shared::{SharedData, save_everything};
use crate::gpu_engine::GpuEngine;
use serde::Serialize;
use tower_http::cors::CorsLayer;

#[derive(Clone)]
struct ApiState {
    shared: Arc<Mutex<SharedData>>,
    gpu: Arc<GpuEngine>,
}

#[derive(Serialize)]
struct StatusResponse {
    ticks: u64,
    cumulative_ticks: u64,
    population: usize,
    ticks_per_second: f32,
    is_paused: bool,
    generation: u32,
}

pub async fn start_api(shared: Arc<Mutex<SharedData>>, gpu: Arc<GpuEngine>, port: u16) {
    let state = ApiState { shared, gpu };

    let app = Router::new()
        .route("/status", get(get_status))
        .route("/config", get(get_config).post(update_config))
        .route("/pause", post(pause_sim))
        .route("/resume", post(resume_sim))
        .route("/save", post(trigger_save))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    println!("Research API listening on http://{}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn get_status(State(state): State<ApiState>) -> Json<StatusResponse> {
    let data = state.shared.lock().unwrap();
    Json(StatusResponse {
        ticks: data.total_ticks,
        cumulative_ticks: data.cumulative_ticks,
        population: data.sim.states.iter().filter(|s| s.health > 0.0).count(),
        ticks_per_second: data.ticks_per_second,
        is_paused: data.is_paused,
        generation: data.generation_survival_times.len() as u32,
    })
}

async fn get_config(State(state): State<ApiState>) -> Json<crate::config::SimConfig> {
    let data = state.shared.lock().unwrap();
    Json(data.config.clone())
}

async fn update_config(
    State(state): State<ApiState>,
    Json(new_config): Json<crate::config::SimConfig>,
) -> impl IntoResponse {
    let mut data = state.shared.lock().unwrap();
    data.config = new_config;
    // Notify GPU if needed (main loop usually handles this via update_config every tick, but let's be explicit if we add a flag)
    StatusCode::OK
}

async fn pause_sim(State(state): State<ApiState>) -> impl IntoResponse {
    let mut data = state.shared.lock().unwrap();
    data.is_paused = true;
    StatusCode::OK
}

async fn resume_sim(State(state): State<ApiState>) -> impl IntoResponse {
    let mut data = state.shared.lock().unwrap();
    data.is_paused = false;
    StatusCode::OK
}

async fn trigger_save(State(state): State<ApiState>) -> impl IntoResponse {
    let data = state.shared.lock().unwrap();
    let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let save_name = format!("api_save_{}", timestamp);
    
    // We can't await save_everything here easily because it needs GPU cells which might be locked
    // But we can spawn it.
    let shared_clone = data.clone();
    let gpu_clone = state.gpu.clone();
    
    std::thread::spawn(move || {
        let mut s = shared_clone;
        s.sim.env.map_cells = gpu_clone.fetch_cells();
        pollster::block_on(save_everything(&s, &save_name, false, None));
    });

    StatusCode::ACCEPTED
}
