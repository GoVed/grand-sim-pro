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

pub fn spawn(sim_thread_data: Arc<Mutex<SharedData>>, gpu: Arc<GpuEngine>) {
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
                gpu.compute_ticks(ticks_per_loop);
                
                ticks_this_second += ticks_per_loop as u64;
                if last_perf_calc_time.elapsed().as_secs() >= 1 {
                    if let Ok(mut data) = sim_thread_data.lock() {
                        data.ticks_per_second = ticks_this_second as f32 / last_perf_calc_time.elapsed().as_secs_f32();
                    }
                    ticks_this_second = 0;
                    last_perf_calc_time = Instant::now();
                }

                // Async Fetch & Process
                if !is_worker_active && last_fetch_time.elapsed().as_millis() >= 32 { 
                    let (states, genetics) = gpu.fetch_agents();
                    let _ = worker_tx.send((states, genetics, config));
                    is_worker_active = true;
                    last_fetch_time = Instant::now();
                }

                // Check for worker results
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
                        
                        data.total_ticks += ticks_per_loop as u64;
                        data.cumulative_ticks += ticks_per_loop as u64;

                        // Telemetry (Seamless & Backgrounded)
                        let telemetry_interval = data.config.telemetry.export_interval_ticks as u64;
                        if data.config.telemetry.enabled != 0 && data.cumulative_ticks >= data.last_telemetry_tick + telemetry_interval {
                            let config_clone = data.config.clone();
                            let states_clone = data.sim.states.clone();
                            let cumulative_ticks = data.cumulative_ticks;
                            let cumulative_births = data.cumulative_births;
                            let cumulative_deaths = data.cumulative_deaths;
                            let generation = data.generation_survival_times.len() as u32;
                            let gpu_c = gpu.clone();
                            
                            std::thread::spawn(move || {
                                let mut telemetry_exporter = crate::telemetry::TelemetryExporter::new("telemetry.csv");
                                let cells = gpu_c.fetch_cells();
                                let _ = telemetry_exporter.export_optimized(&config_clone, &states_clone, &cells, cumulative_ticks, cumulative_births, cumulative_deaths, generation);
                            });
                            data.last_telemetry_tick = data.cumulative_ticks;
                        }
                    }
                    is_worker_active = false;
                }
                
                // Shared maintenance
                let mut auto_save_maintenance = None;
                if let Ok(mut data) = sim_thread_data.try_lock() {
                     // Check for auto-restart
                    let survivors: Vec<_> = data.sim.states.iter().enumerate()
                        .filter(|(_, s)| s.health > 0.0)
                        .map(|(_, s)| Person { state: *s, genetics: data.sim.genetics[s.genetics_index as usize] })
                        .collect();
                    
                    if survivors.len() < data.config.sim.founder_count as usize {
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
                    let auto_save_interval = data.config.sim.auto_save_interval_ticks as u64;
                    if auto_save_interval > 0 && data.total_ticks >= last_auto_save_tick + auto_save_interval {
                        let mut save_clone = data.clone();
                        save_clone.sim.env.map_cells = Vec::new();
                        auto_save_maintenance = Some(save_clone);
                        last_auto_save_tick = data.total_ticks;
                    }
                }

                if let Some(shared_clone) = auto_save_maintenance {
                    let gpu_c = gpu.clone();
                    std::thread::spawn(move || {
                        let mut s = shared_clone;
                        s.sim.env.map_cells = gpu_c.fetch_cells();
                        let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
                        let save_name = format!("auto_save_{}", timestamp);
                        pollster::block_on(crate::shared::save_everything(&s, &save_name, false, None));
                    });
                }
            }

            if is_paused { thread::sleep(Duration::from_millis(16)); }
            else { thread::sleep(Duration::from_millis(1)); }
        }
    });
}
