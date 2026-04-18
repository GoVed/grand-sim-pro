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
        let (worker_tx, worker_rx) = std::sync::mpsc::channel::<(Vec<crate::agent::AgentState>, Vec<crate::agent::Genetics>, crate::config::SimConfig)>();
        let (result_tx, result_rx) = std::sync::mpsc::channel::<(Vec<crate::agent::AgentState>, Vec<crate::agent::Genetics>, u64)>();

        // Spawn background worker for heavy CPU tasks (Spatial Sorting)
        thread::spawn(move || {
            while let Ok((mut states, genetics, config)) = worker_rx.recv() {
                let start_worker = Instant::now();
                
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
                let _ = result_tx.send((states, genetics, worker_time));
            }
        });

        loop {
            let (is_paused, ticks_per_loop, config) = {
                let data = sim_thread_data.lock().unwrap();
                (data.is_paused, data.ticks_per_loop, data.config)
            };

            if !is_paused {
                gpu.update_config(&config);
                
                // 1. Dispatch GPU Ticks
                let batch_size = if is_headless {
                    if !is_worker_active { ticks_per_loop * 10 } else { 0 }
                } else {
                    if !is_worker_active { ticks_per_loop * 2 } else { ticks_per_loop }
                };

                if batch_size > 0 {
                    gpu.compute_ticks(batch_size);
                    ticks_this_second += batch_size as u64;
                    
                    // Update shared counters immediately so API/UI see progress
                    if let Ok(mut data) = sim_thread_data.try_lock() {
                        data.total_ticks += batch_size as u64;
                        data.cumulative_ticks += batch_size as u64;
                        data.config.sim.current_tick += batch_size as u32;
                    }

                    // In headless mode, if we just ran a batch, we MUST sync now to avoid stale sorting
                    if is_headless && !is_worker_active {
                        let (states, genetics) = gpu.fetch_agents();
                        let _ = worker_tx.send((states, genetics, config));
                        is_worker_active = true;
                    }
                }

                // 2. Performance Tracking
                if last_perf_calc_time.elapsed().as_secs() >= 1 {
                    if let Ok(mut data) = sim_thread_data.lock() {
                        data.ticks_per_second = ticks_this_second as f32 / last_perf_calc_time.elapsed().as_secs_f32();
                    }
                    ticks_this_second = 0;
                    last_perf_calc_time = Instant::now();
                }

                // 3. UI Mode Periodic Sync
                if !is_headless && !is_worker_active && last_fetch_time.elapsed().as_millis() >= 64 { 
                    let (states, genetics) = gpu.fetch_agents();
                    let _ = worker_tx.send((states, genetics, config));
                    is_worker_active = true;
                    last_fetch_time = Instant::now();
                }

                // 4. Process Worker Results
                if let Ok((states, genetics, worker_micros)) = result_rx.try_recv() {
                    if let Ok(mut data) = sim_thread_data.lock() {
                        let prev_births = data.sim.total_births;
                        let prev_deaths = data.sim.total_deaths;
                        data.sim.states = states;
                        data.sim.genetics = genetics;
                        data.last_compute_time_micros = worker_micros as u128;
                        data.sim.process_genetics_and_births(&config);
                        
                        let new_births = data.sim.total_births.saturating_sub(prev_births);
                        let new_deaths = data.sim.total_deaths.saturating_sub(prev_deaths);
                        data.cumulative_births += new_births;
                        data.cumulative_deaths += new_deaths;
                        
                        gpu.update_agents(&data.sim.states, &data.sim.genetics);

                        // TELEMETRY TRIGGER (Inside result block to ensure we have fresh GPU data)
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

                        // Debug Check
                        if data.cumulative_ticks % 100 == 0 {
                            let living: Vec<_> = data.sim.states.iter().filter(|s| s.health > 0.0).collect();
                            if !living.is_empty() {
                                let mut sum_aggr = 0.0; let mut sum_repro = 0.0;
                                for s in &living { sum_aggr += s.attack_intent; sum_repro += s.reproduce_desire; }
                                println!("T:{} | Pop:{} | Debug Aggr:{:.6} Repro:{:.6} | Sample[0] Aggr:{:.6} Repro:{:.6}", 
                                    data.cumulative_ticks, living.len(), sum_aggr / living.len() as f32, sum_repro / living.len() as f32, 
                                    living[0].attack_intent, living[0].reproduce_desire);
                            }
                        }
                    }
                    is_worker_active = false;
                }
                
                // 5. Background Maintenance (Restart and Save)
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
            else if !is_headless { thread::sleep(Duration::from_millis(1)); }
        }
    });
}
