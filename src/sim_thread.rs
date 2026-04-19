/*
 * Grand Sim Pro: A high-performance GPGPU evolutionary agent simulation.
 * Part of an independent research project into emergent biological complexity.
 *
 * Copyright (C) 2026 Ved Hirenkumar Suthar
 * Licensed under the GNU General Public License v3.0 or later.
 * * This software is provided "as is", without warranty of any kind.
 * See the LICENSE file in the project root for full license details.
 */

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::shared::SharedData;
use crate::gpu_engine::GpuEngine;
use crate::agent::Person;

pub fn spawn(sim_thread_data: Arc<Mutex<SharedData>>, gpu: Arc<GpuEngine>, is_headless: bool) {
    thread::spawn(move || {
        let mut last_fetch_time = Instant::now();
        let mut last_auto_save_tick = 0;
        
        let mut last_perf_calc_time = Instant::now();
        let mut ticks_this_second = 0u64;

        // Background worker state
        let mut is_worker_active = false;
        let (worker_tx, worker_rx) = std::sync::mpsc::channel::<crate::config::SimConfig>();
        let (result_tx, result_rx) = std::sync::mpsc::channel::<(Vec<crate::agent::AgentState>, u64)>();

        // Spawn background worker for heavy tasks (GPU Fetch + Spatial Sorting)
        let gpu_worker = gpu.clone();
        thread::spawn(move || {
            while let Ok(config) = worker_rx.recv() {
                let start_worker = Instant::now();

                // 1. Fetch from GPU (Heavy)
                let mut states = gpu_worker.fetch_agents();                
                // 2. Spatial Sorting (Heavy CPU)
                use rayon::prelude::*;
                let map_w = config.world.map_width as usize;
                let map_h = config.world.map_height as usize;
                let mut indices: Vec<usize> = (0..states.len()).collect();
                indices.par_sort_by_key(|&i| {
                    let s = &states[i];
                    if s.health <= 0.0 { return usize::MAX; }
                    let ty = (s.y as usize).clamp(0, map_h - 1);
                    let tx = (s.x as usize).clamp(0, map_w - 1);
                    ty * map_w + tx
                });

                let mut sorted_states = Vec::with_capacity(states.len());
                for &i in &indices { sorted_states.push(states[i]); }
                states = sorted_states;

                let worker_time = start_worker.elapsed().as_micros() as u64;
                let _ = result_tx.send((states, worker_time));
            }
        });

        loop {
            let (is_paused, ticks_per_loop, config) = {
                if let Ok(data) = sim_thread_data.lock() {
                    (data.is_paused, data.ticks_per_loop, data.config)
                } else {
                    (true, 1, crate::config::SimConfig::default())
                }
            };

            if !is_paused {
                gpu.update_config(&config);
                
                // 1. Dispatch GPU Ticks (Pure Fire-and-Forget for maximum TPS)
                let batch_size = if is_headless {
                    ticks_per_loop * 5 
                } else {
                    ticks_per_loop
                };

                gpu.compute_ticks(batch_size as u32);
                
                // Throttle CPU to prevent WGPU queue flooding in headless mode
                if is_headless { gpu.wait_idle(); }
                
                ticks_this_second += batch_size as u64;
                
                // Update shared counters (Non-blocking)
                if let Ok(mut data) = sim_thread_data.try_lock() {
                    data.total_ticks += batch_size as u64;
                    data.cumulative_ticks += batch_size as u64;
                    data.config.sim.current_tick += batch_size as u32;
                }

                // 2. Trigger Worker if not busy
                let sync_interval = if is_headless { 100 } else { 64 }; // ms
                if !is_worker_active && last_fetch_time.elapsed().as_millis() >= sync_interval {
                    let _ = worker_tx.send(config);
                    is_worker_active = true;
                    last_fetch_time = Instant::now();
                }

                // 3. Process Worker Results (Non-blocking)
                if let Ok((states, worker_micros)) = result_rx.try_recv() {
                    if let Ok(mut data) = sim_thread_data.lock() {
                        let prev_births = data.sim.total_births;
                        let prev_deaths = data.sim.total_deaths;
                        data.sim.states = states;
                        data.last_compute_time_micros = worker_micros as u128;
                        data.sim.process_genetics_and_births(&config);
                        
                        let new_births = data.sim.total_births.saturating_sub(prev_births);
                        let new_deaths = data.sim.total_deaths.saturating_sub(prev_deaths);
                        data.cumulative_births += new_births;
                        data.cumulative_deaths += new_deaths;
                        
                        gpu.update_agents(&data.sim.states, &data.sim.genetics);

                        // Telemetry trigger
                        let telemetry_interval = data.config.telemetry.export_interval_ticks as u64;
                        if data.config.telemetry.enabled != 0 && data.cumulative_ticks >= data.last_telemetry_tick + telemetry_interval {
                            let config_clone = data.config.clone();
                            let states_clone = data.sim.states.clone();
                            let cumulative_ticks = data.cumulative_ticks;
                            let cumulative_births = data.cumulative_births;
                            let cumulative_deaths = data.cumulative_deaths;
                            let r#gen = data.generation_survival_times.len() as u32;
                            let gpu_c = gpu.clone();
                            
                            std::thread::spawn(move || {
                                let mut telemetry_exporter = crate::telemetry::TelemetryExporter::new("telemetry.csv");
                                let cells = gpu_c.fetch_cells();
                                let _ = telemetry_exporter.export_optimized(&config_clone, &states_clone, &cells, cumulative_ticks, cumulative_births, cumulative_deaths, r#gen);
                            });
                            data.last_telemetry_tick = data.cumulative_ticks;
                        }
                    }
                    is_worker_active = false;
                }

                // 4. Performance Tracking
                if last_perf_calc_time.elapsed().as_secs() >= 1 {
                    if let Ok(mut data) = sim_thread_data.lock() {
                        data.ticks_per_second = ticks_this_second as f32 / last_perf_calc_time.elapsed().as_secs_f32();
                    }
                    ticks_this_second = 0;
                    last_perf_calc_time = Instant::now();
                }
                
                // 5. Background Maintenance
                if let Ok(mut data) = sim_thread_data.try_lock() {
                     // Auto-Restart
                    let survivors_count = data.sim.states.iter().filter(|s| s.health > 0.0).count();
                    if survivors_count < data.config.sim.founder_count as usize {
                        let survivors: Vec<_> = data.sim.states.iter().enumerate()
                            .filter(|(_, s)| s.health > 0.0)
                            .map(|(_, s)| Person { state: *s, genetics: data.sim.genetics[s.genetics_index as usize] })
                            .collect();

                        data.restart_message_active = true;
                        let total_ticks = data.total_ticks;
                        if total_ticks > 0 { data.generation_survival_times.push(total_ticks); }
                        
                        let mut rng = ::rand::thread_rng();
                        data.sim = crate::simulation::SimulationManager::new(
                            data.config.world.map_width, 
                            data.config.world.map_height, 
                            rand::Rng::r#gen(&mut rng), 
                            data.config.sim.agent_count, 
                            &data.config, 
                            survivors
                        );
                        
                        data.total_ticks = 0; data.restart_message_active = false;
                        gpu.update_agents(&data.sim.states, &data.sim.genetics);
                        gpu.update_heights(&data.sim.env.height_map);
                        gpu.update_cells(&data.sim.env.map_cells);
                    }
                    
                    // Auto-Save
                    if data.config.sim.auto_save_interval_ticks > 0 && data.total_ticks >= last_auto_save_tick + data.config.sim.auto_save_interval_ticks as u64 {
                        let mut save_shared = data.clone();
                        save_shared.sim.env.map_cells = Vec::new();
                        let gpu_c = gpu.clone();
                        std::thread::spawn(move || {
                            let mut s = save_shared;
                            s.sim.env.map_cells = gpu_c.fetch_cells();
                            let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
                            let save_name = format!("auto_save_{}", timestamp);
                            pollster::block_on(crate::shared::save_everything(&s, &save_name, false, None));
                        });
                        last_auto_save_tick = data.total_ticks;
                    }
                }
            }

            if is_paused { thread::sleep(Duration::from_millis(16)); }
            else { thread::sleep(Duration::from_millis(1)); }
        }
    });
}
