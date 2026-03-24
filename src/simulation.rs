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

pub struct SimulationManager {
    pub env: Environment,
    pub agents: Vec<Person>,
    pub pending_births: std::collections::HashMap<u32, Person>,
}

impl SimulationManager {
    pub fn new(width: u32, height: u32, seed: u32, count: u32, config: &crate::config::SimConfig) -> Self {
        let env = Environment::new(width, height, seed, config);
        let agents = (0..count).map(|_| Person::new((width/2) as f32, (height/2) as f32, config)).collect();
        Self { env, agents, pending_births: std::collections::HashMap::new() }
    }
}