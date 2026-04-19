/*
 * Grand Sim Pro: A high-performance GPGPU evolutionary agent simulation.
 * Part of an independent research project into emergent biological complexity.
 *
 * Copyright (C) 2026 Ved Hirenkumar Suthar
 * Licensed under the GNU General Public License v3.0 or later.
 * * This software is provided "as is", without warranty of any kind.
 * See the LICENSE file in the project root for full license details.
 */

use world_sim::simulation::SimulationManager;
use world_sim::config::SimConfig;
use world_sim::gpu_engine::GpuEngine;
use world_sim::agent::{AgentState, Genetics, W1_SIZE, W2_SIZE, W3_SIZE};
use world_sim::environment::CellState;
use std::time::Instant;

#[test]
fn test_stress_performance() {
    let mut config = SimConfig::default();
    // High-Resolution Research Map
    config.world.map_width = 4000;
    config.world.map_height = 2500;
    config.sim.agent_count = 20000; // Stress population
    
    let states = vec![AgentState::default(); config.sim.agent_count as usize];
    let genetics = vec![Genetics { 
        w1_weights: [0.0; W1_SIZE], w1_indices: [0; W1_SIZE], 
        w2: [0.0; W2_SIZE], w3: [0.0; W3_SIZE], cnn_kernels: [0.0; 72]
    }; config.sim.agent_count as usize];
    
    let map_size = (config.world.map_width * config.world.map_height) as usize;
    let heights = vec![0.0f32; map_size];
    let cells = vec![CellState::default(); map_size];
    
    println!("--- Stress Performance Analysis (4000x2500 Map, 20k Agents) ---");
    
    let start_init = Instant::now();
    let gpu = GpuEngine::new(&states, &genetics, &heights, &cells, &config);
    println!("GPU Stress Init: {:?}", start_init.elapsed());

    // Test 1: High-Density GPU Compute
    let start_submit = Instant::now();
    gpu.compute_ticks(100);
    let submit_time = start_submit.elapsed();
    
    gpu.wait_idle(); // Wait for compute to finish
    let compute_time = start_submit.elapsed();
    
    println!("Compute 100 ticks (Submission): {:?}", submit_time);
    println!("Compute 100 ticks (Total Execution): {:?}", compute_time);
    println!("Theoretical TPS (Execution): {:.0}", 100.0 / compute_time.as_secs_f32());

    // Test 2: Bus Bandwidth (Agent Sync)
    let start_fetch = Instant::now();
    let _ = gpu.fetch_agents();
    println!("Agent State Fetch (20k): {:?}", start_fetch.elapsed());

    // Test 3: Large-Scale Map Telemetry Fetch
    let start_cells = Instant::now();
    let _ = gpu.fetch_cells();
    println!("Full Map Cell Fetch (10M tiles): {:?}", start_cells.elapsed());

    // Test 4: Heavy Spatial Sort
    use rayon::prelude::*;
    let mut indices: Vec<usize> = (0..states.len()).collect();
    let start_sort = Instant::now();
    indices.par_sort_by_key(|&i| {
        let s = &states[i];
        let ty = (s.y as usize).clamp(0, config.world.map_height as usize - 1);
        let tx = (s.x as usize).clamp(0, config.world.map_width as usize - 1);
        ty * config.world.map_width as usize + tx
    });
    println!("Parallel Spatial Sort (20k agents): {:?}", start_sort.elapsed());
}
