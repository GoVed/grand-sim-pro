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
    thread::spawn(move || loop {
        let mut ran_ticks = false;
        {
            let mut data = sim_thread_data.lock().unwrap();
            if !data.is_paused {
                let start = Instant::now();
                
                data.config.current_tick = data.total_ticks as u32;
                gpu.update_config(&data.config);

                // Dispatch heavy work entirely to the GPU
                gpu.compute_ticks(data.ticks_per_loop);
                
                // Option 1: Decoupled State Fetching (No longer downloads 165MB of Cell data across PCIe)
                data.sim.agents = gpu.fetch_agents();

                let config = data.config;
                let modifications = data.sim.process_genetics_and_births(&config);

                // Send mutated population back to VRAM
                if modifications { gpu.update_agents(&data.sim.agents); }
                
                data.total_ticks += data.ticks_per_loop as u64;
                data.last_compute_time_ms = start.elapsed().as_millis();
                ran_ticks = true;

                // --- Auto-Restart Logic ---
                let living_count = data.sim.agents.iter().filter(|a| a.health > 0.0).count();
                if living_count < data.config.founder_count as usize {
                    data.restart_message_active = true;
                    drop(data); // Release lock instantly so the UI can render the message
                    thread::sleep(Duration::from_millis(1000));
                    
                    let mut data = sim_thread_data.lock().unwrap();

                    // Record the survival time of the generation that just ended.
                    if data.total_ticks > 0 {
                        let ticks = data.total_ticks;
                        data.generation_survival_times.push(ticks);
                    }
                    
                    // Prioritize alive agents. Since survival is the only metric that matters,
                    // we just move all living agents to the front of the array to be used as founders.
                    data.sim.agents.sort_by(|a, b| (b.health > 0.0).cmp(&(a.health > 0.0)));
                    
                    let target_pop = data.config.agent_count as usize;
                    let map_w = data.config.map_width as f32;
                    let map_h = data.config.map_height as f32;
                    let mutation_rate = data.config.mutation_rate;
                    let mutation_strength = data.config.mutation_strength;
                    let spawn_group_size = data.config.spawn_group_size as usize;

                    let mut rng = ::rand::thread_rng();
                    
                    // Regenerate a completely new procedural map
                    let new_seed = rng.r#gen::<u32>();
                    data.sim.env = crate::environment::Environment::new(data.config.map_width, data.config.map_height, new_seed, &data.config);

                    let mut new_population = Vec::with_capacity(target_pop);
                    let living_count = data.sim.agents.iter().filter(|a| a.health > 0.0).count();
                    let founders_count = living_count.max(1).min(data.config.founder_count as usize);
                    
                    let random_agents_count = (target_pop as f32 * data.config.random_spawn_percentage) as usize;
                    let descendant_agents_count = target_pop - random_agents_count;
                    let children_per_founder = descendant_agents_count / founders_count.max(1);
                    
                    let get_land_spawn_point = |rng: &mut rand::rngs::ThreadRng| -> (f32, f32) {
                        loop {
                            let px = rng.gen_range(0.0f32..map_w);
                            let py = rng.gen_range(0.0f32..map_h);
                            let idx = (py as usize).clamp(0, (data.config.map_height - 1) as usize) * (data.config.map_width as usize) + (px as usize).clamp(0, (data.config.map_width - 1) as usize);
                            if data.sim.env.height_map[idx] >= 0.0 { return (px, py); }
                        }
                    };

                    let mut current_spawn_pt = get_land_spawn_point(&mut rng);
                    let mut spawn_count = 0;

                    // Create totally random agents
                    for _ in 0..random_agents_count {
                        if spawn_count >= spawn_group_size { current_spawn_pt = get_land_spawn_point(&mut rng); spawn_count = 0; }
                        let px = (current_spawn_pt.0 + rng.gen_range(-5.0f32..5.0f32)).rem_euclid(map_w);
                        let py = (current_spawn_pt.1 + rng.gen_range(-5.0f32..5.0f32)).rem_euclid(map_h);

                        let mut random_agent = Person::new(px, py, &data.config);
                        random_agent.age = rng.gen_range(0.0f32..data.config.max_age * 0.8);
                        new_population.push(random_agent);
                        spawn_count += 1;
                    }

                    // Create descendants from founders
                    for i in 0..founders_count {
                        let founder = &data.sim.agents[i];
                        for _ in 0..children_per_founder {
                            if spawn_count >= spawn_group_size { current_spawn_pt = get_land_spawn_point(&mut rng); spawn_count = 0; }
                            let px = (current_spawn_pt.0 + rng.gen_range(-5.0f32..5.0f32)).rem_euclid(map_w);
                            let py = (current_spawn_pt.1 + rng.gen_range(-5.0f32..5.0f32)).rem_euclid(map_h);

                            let mut child = founder.clone_as_descendant(px, py, mutation_rate, mutation_strength, &data.config);
                            child.age = rng.gen_range(0.0f32..data.config.max_age * 0.8);
                            new_population.push(child);
                            spawn_count += 1;
                        }
                    }
                    // Fill any remaining slots with descendants of the first founder
                    while new_population.len() < target_pop { // This handles cases where target_pop is not perfectly divisible
                        if spawn_count >= spawn_group_size { current_spawn_pt = get_land_spawn_point(&mut rng); spawn_count = 0; }
                        let px = (current_spawn_pt.0 + rng.gen_range(-5.0f32..5.0f32)).rem_euclid(map_w);
                        let py = (current_spawn_pt.1 + rng.gen_range(-5.0f32..5.0f32)).rem_euclid(map_h);

                        let mut child = data.sim.agents[0].clone_as_descendant(px, py, mutation_rate, mutation_strength, &data.config);
                        child.age = rng.gen_range(0.0f32..data.config.max_age * 0.8);
                        new_population.push(child);
                        spawn_count += 1;
                    }
                    data.sim.agents = new_population;
                    
                    data.total_ticks = 0;
                    data.restart_message_active = false;
                    gpu.update_agents(&data.sim.agents);
                    gpu.update_heights(&data.sim.env.height_map); // Force a topography re-upload
                    gpu.update_cells(&data.sim.env.map_cells);
                }
            }
        }
        
        if !ran_ticks {
            thread::sleep(Duration::from_millis(16)); // Prevent CPU hogging when paused
        } else {
            thread::sleep(Duration::from_millis(1)); // Context switch
        }
    });
}