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
use ::rand::Rng;

pub fn spawn(sim_thread_data: Arc<Mutex<SharedData>>, gpu: Arc<GpuEngine>) {
    thread::spawn(move || {
        let mut last_fetch_time = Instant::now();
        let mut telemetry = crate::telemetry::TelemetryExporter::new("telemetry.csv");
        let mut last_auto_save_tick = 0;
        
        let mut last_perf_calc_time = Instant::now();
        let mut ticks_this_second = 0u64;

        loop {
            let (is_paused, ticks_per_loop, config) = {
                let data = sim_thread_data.lock().unwrap();
                (data.is_paused, data.ticks_per_loop, data.config)
            };

            if !is_paused {
                let start = Instant::now();

                gpu.update_config(&config);
                gpu.compute_ticks(ticks_per_loop);
                
                ticks_this_second += ticks_per_loop as u64;
                if last_perf_calc_time.elapsed().as_secs() >= 1 {
                    let mut data = sim_thread_data.lock().unwrap();
                    data.ticks_per_second = ticks_this_second as f32 / last_perf_calc_time.elapsed().as_secs_f32();
                    ticks_this_second = 0;
                    last_perf_calc_time = Instant::now();
                }

                if last_fetch_time.elapsed().as_millis() >= 16 {
                    let (states, genetics) = gpu.fetch_agents();

                    let mut data = sim_thread_data.lock().unwrap();
                    data.sim.states = states;
                    data.sim.genetics = genetics;
                    let modifications = data.sim.process_genetics_and_births(&config);
                    data.config.sim.current_tick += ticks_per_loop as u32;

                    // High-Performance Parallel Spatial Sorting
                    use rayon::prelude::*;
                    let map_w = config.world.map_width as usize;
                    let map_h = config.world.map_height as usize;
                    
                    let mut indices: Vec<usize> = (0..data.sim.states.len()).collect();
                    indices.par_sort_by_key(|&i| {
                        let s = &data.sim.states[i];
                        if s.health <= 0.0 { return usize::MAX; }
                        let ty = (s.y as usize).clamp(0, map_h - 1);
                        let tx = (s.x as usize).clamp(0, map_w - 1);
                        ty * map_w + tx
                    });

                    let mut sorted_states = Vec::with_capacity(data.sim.states.len());
                    for &i in &indices { sorted_states.push(data.sim.states[i]); }
                    data.sim.states = sorted_states;

                    if modifications || true { gpu.update_agents(&data.sim.states, &data.sim.genetics); }
                    last_fetch_time = Instant::now();

                    data.total_ticks += ticks_per_loop as u64;
                    data.cumulative_ticks += ticks_per_loop as u64;
                    data.last_compute_time_micros = start.elapsed().as_micros();

                    // Telemetry Export (Seamless)
                    let telemetry_interval = data.config.telemetry.export_interval_ticks as u64;
                    if data.config.telemetry.enabled != 0 && data.cumulative_ticks >= data.last_telemetry_tick + telemetry_interval {
                        data.sim.env.map_cells = gpu.fetch_cells();
                        // Generation is 1-indexed for humans, but 0-indexed in length. 
                        // Let's use len + 1 if we want it to be current generation.
                        let generation = data.generation_survival_times.len() as u32;
                        let _ = telemetry.export(&data.sim, &data.config, data.cumulative_ticks, generation);
                        data.last_telemetry_tick = data.cumulative_ticks;
                    }

                    // Auto-Save
                    let auto_save_interval = data.config.sim.auto_save_interval_ticks as u64;
                    if auto_save_interval > 0 && data.total_ticks >= last_auto_save_tick + auto_save_interval {
                        data.sim.env.map_cells = gpu.fetch_cells();
                        let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
                        let save_name = format!("auto_save_{}", timestamp);
                        let shared_clone = data.clone();
                        std::thread::spawn(move || { pollster::block_on(crate::shared::save_everything(&shared_clone, &save_name, false, None)); });
                        last_auto_save_tick = data.total_ticks;
                    }

                    // Auto-Restart
                    let living_count = data.sim.states.iter().filter(|s| s.health > 0.0).count();
                    if living_count < data.config.sim.founder_count as usize {
                        data.restart_message_active = true;
                        drop(data);
                        thread::sleep(Duration::from_millis(1000));
                        let mut data = sim_thread_data.lock().unwrap();

                        let total_ticks = data.total_ticks;
                        if total_ticks > 0 { data.generation_survival_times.push(total_ticks); }

                        data.sim.states.sort_by(|a, b| (b.health > 0.0).cmp(&(a.health > 0.0)));
                        let target_pop = data.config.sim.agent_count as usize;
                        let map_w = data.config.world.map_width as f32;
                        let map_h = data.config.world.map_height as f32;
                        let mut rng = ::rand::thread_rng();
                        data.sim.env = crate::environment::Environment::new(data.config.world.map_width, data.config.world.map_height, rng.r#gen(), &data.config);

                        let mut new_states = Vec::with_capacity(target_pop);
                        let mut new_genetics = Vec::with_capacity(target_pop);
                        let founders_count = data.sim.states.iter().filter(|s| s.health > 0.0).count().max(1).min(data.config.sim.founder_count as usize);
                        let random_agents_count = (target_pop as f32 * data.config.genetics.random_spawn_percentage) as usize;
                        let children_per_founder = (target_pop - random_agents_count) / founders_count;

                        for _ in 0..random_agents_count {
                            let random_agent = Person::new(rng.gen_range(0.0..map_w), rng.gen_range(0.0..map_h), new_states.len() as u32, &data.config);
                            new_states.push(random_agent.state); new_genetics.push(random_agent.genetics);
                        }
                        for i in 0..founders_count {
                            let founder = Person { state: data.sim.states[i], genetics: data.sim.genetics[data.sim.states[i].genetics_index as usize] };
                            for _ in 0..children_per_founder {
                                let child = founder.clone_as_descendant(rng.gen_range(0.0..map_w), rng.gen_range(0.0..map_h), new_states.len() as u32, data.config.genetics.mutation_rate, data.config.genetics.mutation_strength, &data.config);
                                new_states.push(child.state); new_genetics.push(child.genetics);
                            }
                        }
                        while new_states.len() < target_pop {
                            let random_agent = Person::new(rng.gen_range(0.0..map_w), rng.gen_range(0.0..map_h), new_states.len() as u32, &data.config);
                            new_states.push(random_agent.state); new_genetics.push(random_agent.genetics);
                        }

                        data.sim.states = new_states; data.sim.genetics = new_genetics;
                        data.sim.total_births = data.config.sim.agent_count as u64; data.sim.total_deaths = 0;
                        data.total_ticks = 0; data.restart_message_active = false;
                        gpu.update_agents(&data.sim.states, &data.sim.genetics);
                        gpu.update_heights(&data.sim.env.height_map);
                        gpu.update_cells(&data.sim.env.map_cells);
                    }
                }
            }

            if is_paused { thread::sleep(Duration::from_millis(16)); }
            else { thread::sleep(Duration::from_millis(1)); }
        }
    });
}
