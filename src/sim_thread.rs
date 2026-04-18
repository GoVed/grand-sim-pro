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
        let mut last_telemetry_tick = 0;
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
                // Dispatch heavy work entirely to the GPU
                gpu.compute_ticks(ticks_per_loop);
                
                ticks_this_second += ticks_per_loop as u64;
                if last_perf_calc_time.elapsed().as_secs() >= 1 {
                    let mut data = sim_thread_data.lock().unwrap();
                    data.ticks_per_second = ticks_this_second as f32 / last_perf_calc_time.elapsed().as_secs_f32();
                    ticks_this_second = 0;
                    last_perf_calc_time = Instant::now();
                }

                // Throttle PCIe bandwidth: Fetch the massive Agent state roughly at 60Hz instead of every compute loop.
                if last_fetch_time.elapsed().as_millis() >= 16 {
                    let (states, genetics) = gpu.fetch_agents();

                    let mut data = sim_thread_data.lock().unwrap();
                    data.sim.states = states;
                    data.sim.genetics = genetics;
                    let modifications = data.sim.process_genetics_and_births(&config);
                    data.config.sim.current_tick += ticks_per_loop as u32;

                    // Update local agents after genetics
                    let mut states = data.sim.states.clone();
                    
                    // High-Performance Parallel Spatial Sorting
                    // We sort indices instead of the 28KB structs to avoid massive memory bandwidth overhead
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

                    // Reorder states based on sorted indices. genetics remains UNCHANGED.
                    let mut sorted_states = Vec::with_capacity(states.len());
                    for &i in &indices {
                        sorted_states.push(states[i]);
                    }
                    states = sorted_states;

                    data.sim.states = states.clone();

                    if modifications || true { gpu.update_agents(&data.sim.states, &data.sim.genetics); }
                    last_fetch_time = Instant::now();

                    data.total_ticks += ticks_per_loop as u64;
                    data.last_compute_time_micros = start.elapsed().as_micros();

                    // Telemetry Export
                    if data.config.telemetry.enabled != 0 && data.total_ticks >= last_telemetry_tick + data.config.telemetry.export_interval_ticks as u64 {
                        let generation = data.generation_survival_times.len() as u32;
                        let _ = telemetry.export(&data.sim, &data.config, data.total_ticks, generation);
                        last_telemetry_tick = data.total_ticks;
                    }

                    // Auto-Save
                    let auto_save_interval = data.config.sim.auto_save_interval_ticks as u64;
                    if auto_save_interval > 0 && data.total_ticks >= last_auto_save_tick + auto_save_interval {
                        data.sim.env.map_cells = gpu.fetch_cells();
                        let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
                        let save_name = format!("auto_save_{}", timestamp);
                        let shared_clone = data.clone();
                        // Run save in background thread to avoid blocking simulation
                        std::thread::spawn(move || {
                            pollster::block_on(crate::shared::save_everything(&shared_clone, &save_name, false, None));
                        });
                        last_auto_save_tick = data.total_ticks;
                    }

                    // Check for auto-restart while we have the lock
                    let living_count = data.sim.states.iter().filter(|s| s.health > 0.0).count();
                    if living_count < data.config.sim.founder_count as usize {
                        // ... (keep the restart logic inside the lock as it modifies everything)
                        data.restart_message_active = true;
                        drop(data);
                        thread::sleep(Duration::from_millis(1000));
                        let mut data = sim_thread_data.lock().unwrap();

                        // [EXISTING RESTART LOGIC START]
                        if data.total_ticks > 0 {
                            let ticks = data.total_ticks;
                            data.generation_survival_times.push(ticks);
                        }

                        // Sort states so living agents (founders) are at the front
                        data.sim.states.sort_by(|a, b| (b.health > 0.0).cmp(&(a.health > 0.0)));

                        let target_pop = data.config.sim.agent_count as usize;
                        let map_w = data.config.world.map_width as f32;
                        let map_h = data.config.world.map_height as f32;
                        let mutation_rate = data.config.genetics.mutation_rate;
                        let mutation_strength = data.config.genetics.mutation_strength;
                        let spawn_group_size = data.config.sim.spawn_group_size as usize;

                        let mut rng = ::rand::thread_rng();
                        let new_seed = rng.r#gen::<u32>();
                        data.sim.env = crate::environment::Environment::new(data.config.world.map_width, data.config.world.map_height, new_seed, &data.config);

                        let mut new_states = Vec::with_capacity(target_pop);
                        let mut new_genetics = Vec::with_capacity(target_pop);
                        let living_count = data.sim.states.iter().filter(|s| s.health > 0.0).count();
                        let founders_count = living_count.max(1).min(data.config.sim.founder_count as usize);

                        let random_agents_count = (target_pop as f32 * data.config.genetics.random_spawn_percentage) as usize;
                        let descendant_agents_count = target_pop - random_agents_count;
                        let children_per_founder = descendant_agents_count / founders_count.max(1);

                        let mut current_spawn_pt = loop {
                            let px = rng.gen_range(0.0f32..map_w);
                            let u = rng.gen_range(0.0f32..1.0f32);
                            let phi = (1.0 - 2.0 * u).acos();
                            let py = (phi / std::f32::consts::PI) * map_h;
                            let idx = (py as usize).clamp(0, (data.config.world.map_height - 1) as usize) * (data.config.world.map_width as usize) + (px as usize).clamp(0, (data.config.world.map_width - 1) as usize);
                            if data.sim.env.height_map[idx] >= 0.0 { break (px, py); }
                        };
                        let mut current_base_color = ((rng.r#gen::<f32>() * 2.0) - 1.0, (rng.r#gen::<f32>() * 2.0) - 1.0, (rng.r#gen::<f32>() * 2.0) - 1.0);
                        let mut spawn_count = 0;

                        for _ in 0..random_agents_count {
                            if spawn_count >= spawn_group_size { 
                                loop {
                                    let px = rng.gen_range(0.0f32..map_w);
                                    let u = rng.gen_range(0.0f32..1.0f32);
                                    let phi = (1.0 - 2.0 * u).acos();
                                    let py = (phi / std::f32::consts::PI) * map_h;
                                    let idx = (py as usize).clamp(0, (data.config.world.map_height - 1) as usize) * (data.config.world.map_width as usize) + (px as usize).clamp(0, (data.config.world.map_width - 1) as usize);
                                    if data.sim.env.height_map[idx] >= 0.0 { current_spawn_pt = (px, py); break; }
                                }
                                current_base_color = ((rng.r#gen::<f32>() * 2.0) - 1.0, (rng.r#gen::<f32>() * 2.0) - 1.0, (rng.r#gen::<f32>() * 2.0) - 1.0);
                                spawn_count = 0; 
                            }
                            let mut px = current_spawn_pt.0 + rng.gen_range(-5.0f32..5.0f32);
                            let mut py = current_spawn_pt.1 + rng.gen_range(-5.0f32..5.0f32);
                            if py < 0.0 { py = -py; px += map_w / 2.0; }
                            else if py >= map_h { py = 2.0 * map_h - 1.0 - py; px += map_w / 2.0; }
                            px = px.rem_euclid(map_w);

                            let random_agent = Person::new(px, py, new_states.len() as u32, &data.config);
                            let mut s = random_agent.state;
                            s.age = rng.gen_range(0.0f32..data.config.bio.max_age * 0.8);
                            s.pheno_r = (current_base_color.0 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
                            s.pheno_g = (current_base_color.1 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
                            s.pheno_b = (current_base_color.2 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
                            new_states.push(s);
                            new_genetics.push(random_agent.genetics);
                            spawn_count += 1;
                        }

                        for i in 0..founders_count {
                            let founder_state = &data.sim.states[i];
                            let founder_genetics = &data.sim.genetics[founder_state.genetics_index as usize];
                            let founder = Person { state: *founder_state, genetics: *founder_genetics };
                            
                            for _ in 0..children_per_founder {
                                if spawn_count >= spawn_group_size { 
                                    loop {
                                        let px = rng.gen_range(0.0f32..map_w);
                                        let u = rng.gen_range(0.0f32..1.0f32);
                                        let phi = (1.0 - 2.0 * u).acos();
                                        let py = (phi / std::f32::consts::PI) * map_h;
                                        let idx = (py as usize).clamp(0, (data.config.world.map_height - 1) as usize) * (data.config.world.map_width as usize) + (px as usize).clamp(0, (data.config.world.map_width - 1) as usize);
                                        if data.sim.env.height_map[idx] >= 0.0 { current_spawn_pt = (px, py); break; }
                                    }
                                    current_base_color = ((rng.r#gen::<f32>() * 2.0) - 1.0, (rng.r#gen::<f32>() * 2.0) - 1.0, (rng.r#gen::<f32>() * 2.0) - 1.0);
                                    spawn_count = 0; 
                                }
                                let mut px = current_spawn_pt.0 + rng.gen_range(-5.0f32..5.0f32);
                                let mut py = current_spawn_pt.1 + rng.gen_range(-5.0f32..5.0f32);
                                if py < 0.0 { py = -py; px += map_w / 2.0; }
                                else if py >= map_h { py = 2.0 * map_h - 1.0 - py; px += map_w / 2.0; }
                                px = px.rem_euclid(map_w);

                                let child = founder.clone_as_descendant(px, py, new_states.len() as u32, mutation_rate, mutation_strength, &data.config);
                                let mut s = child.state;
                                s.age = rng.gen_range(0.0f32..data.config.bio.max_age * 0.8);
                                s.pheno_r = (current_base_color.0 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
                                s.pheno_g = (current_base_color.1 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
                                s.pheno_b = (current_base_color.2 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
                                new_states.push(s);
                                new_genetics.push(child.genetics);
                                spawn_count += 1;
                            }
                        }
                        while new_states.len() < target_pop {
                            if spawn_count >= spawn_group_size { 
                                loop {
                                    let px = rng.gen_range(0.0f32..map_w);
                                    let u = rng.gen_range(0.0f32..1.0f32);
                                    let phi = (1.0 - 2.0 * u).acos();
                                    let py = (phi / std::f32::consts::PI) * map_h;
                                    let idx = (py as usize).clamp(0, (data.config.world.map_height - 1) as usize) * (data.config.world.map_width as usize) + (px as usize).clamp(0, (data.config.world.map_width - 1) as usize);
                                    if data.sim.env.height_map[idx] >= 0.0 { current_spawn_pt = (px, py); break; }
                                }
                                current_base_color = ((rng.r#gen::<f32>() * 2.0) - 1.0, (rng.r#gen::<f32>() * 2.0) - 1.0, (rng.r#gen::<f32>() * 2.0) - 1.0);
                                spawn_count = 0; 
                            }
                            let mut px = current_spawn_pt.0 + rng.gen_range(-5.0f32..5.0f32);
                            let mut py = current_spawn_pt.1 + rng.gen_range(-5.0f32..5.0f32);
                            if py < 0.0 { py = -py; px += map_w / 2.0; }
                            else if py >= map_h { py = 2.0 * map_h - 1.0 - py; px += map_w / 2.0; }
                            px = px.rem_euclid(map_w);

                            let founder_state = &data.sim.states[0];
                            let founder_genetics = &data.sim.genetics[founder_state.genetics_index as usize];
                            let founder = Person { state: *founder_state, genetics: *founder_genetics };

                            let child = founder.clone_as_descendant(px, py, new_states.len() as u32, mutation_rate, mutation_strength, &data.config);
                            let mut s = child.state;
                            s.age = rng.gen_range(0.0f32..data.config.bio.max_age * 0.8);
                            s.pheno_r = (current_base_color.0 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
                            s.pheno_g = (current_base_color.1 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
                            s.pheno_b = (current_base_color.2 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
                            new_states.push(s);
                            new_genetics.push(child.genetics);
                            spawn_count += 1;
                        }
                        data.sim.states = new_states;
                        data.sim.genetics = new_genetics;
                        data.sim.total_births = data.config.sim.agent_count as u64;
                        data.sim.total_deaths = 0;

                        data.total_ticks = 0;
                        last_telemetry_tick = 0;
                        data.restart_message_active = false;
                        gpu.update_agents(&data.sim.states, &data.sim.genetics);
                        gpu.update_heights(&data.sim.env.height_map);
                        gpu.update_cells(&data.sim.env.map_cells);
                        // [EXISTING RESTART LOGIC END]
                    }
                } else {
                    // Even if we don't fetch agents, we should update ticks
                    let mut data = sim_thread_data.lock().unwrap();
                    data.total_ticks += ticks_per_loop as u64;
                    data.config.sim.current_tick += ticks_per_loop as u32;
                    data.last_compute_time_micros = start.elapsed().as_micros();
                }
            }

            if is_paused {
                thread::sleep(Duration::from_millis(16));
            } else {
                // Sleep for 1ms to give UI thread a chance to take the lock
                thread::sleep(Duration::from_millis(1));
            }
        }
    });
}