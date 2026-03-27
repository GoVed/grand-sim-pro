/*
 * Grand Sim Pro: A high-performance GPGPU evolutionary agent simulation.
 * Part of an independent research project into emergent biological complexity.
 *
 * Copyright (C) 2026 Ved Hirenkumar Suthar
 * Licensed under the GNU General Public License v3.0 or later.
 * * This software is provided "as is", without warranty of any kind.
 * See the LICENSE file in the project root for full license details.
 */

use crate::agent::Person;
use crate::environment::Environment;
use ::rand::Rng;

pub struct SimulationManager {
    pub env: Environment,
    pub agents: Vec<Person>,
    pub pending_births: std::collections::HashMap<u32, Person>,
}

impl SimulationManager {
    pub fn new(width: u32, height: u32, seed: u32, count: u32, config: &crate::config::SimConfig) -> Self {
        let env = Environment::new(width, height, seed, config);
        
        let mut rng = ::rand::thread_rng();
        let spawn_group_size = config.spawn_group_size as usize;
        
        let get_land_spawn_point = |rng: &mut rand::rngs::ThreadRng| -> (f32, f32) {
            loop {
                let px = rng.gen_range(0.0..width as f32);
                let py = rng.gen_range(0.0..height as f32);
                let idx = (py as usize).clamp(0, (height - 1) as usize) * (width as usize) + (px as usize).clamp(0, (width - 1) as usize);
                if env.height_map[idx] >= 0.0 { return (px, py); }
            }
        };

        let mut agents = Vec::with_capacity(count as usize);
        let mut current_spawn_pt = get_land_spawn_point(&mut rng);
        let mut spawn_count = 0;

        for _ in 0..count {
            if spawn_count >= spawn_group_size { current_spawn_pt = get_land_spawn_point(&mut rng); spawn_count = 0; }
            let px = (current_spawn_pt.0 + rng.gen_range(-5.0f32..5.0f32)).rem_euclid(width as f32);
            let py = (current_spawn_pt.1 + rng.gen_range(-5.0f32..5.0f32)).rem_euclid(height as f32);
            agents.push(Person::new(px, py, config));
            spawn_count += 1;
        }
        
        Self { env, agents, pending_births: std::collections::HashMap::new() }
    }
}