/*
 * Grand Sim Pro: A high-performance GPGPU evolutionary agent simulation.
 * Part of an independent research project into emergent biological complexity.
 *
 * Copyright (C) 2026 Ved Hirenkumar Suthar
 * Licensed under the GNU General Public License v3.0 or later.
 * * This software is provided "as is", without warranty of any kind.
 * See the LICENSE file in the project root for full license details.
 */

use crate::agent::{Person, AgentState, Genetics};
use crate::environment::Environment;
use ::rand::Rng;

use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct SimulationManager {
    pub env: Environment,
    pub states: Vec<AgentState>,
    pub genetics: Vec<Genetics>,
    pub pending_births: std::collections::HashMap<u32, Person>,
    pub total_births: u64,
    pub total_deaths: u64,
    
    // Reusable buffers to avoid allocations in process_genetics_and_births
    #[serde(skip)]
    cell_occupants: std::collections::HashMap<usize, Vec<usize>>,
    #[serde(skip)]
    living_ids: std::collections::HashSet<u32>,
    #[serde(skip)]
    dead_indices: Vec<usize>,
    #[serde(skip)]
    births_to_process: Vec<(u32, f32, f32)>,
    #[serde(skip)]
    males: Vec<usize>,
    #[serde(skip)]
    females: Vec<usize>,
}

pub fn load_founders(config: &crate::config::SimConfig) -> Vec<Person> {
    let mut founders = Vec::new();
    if let Ok(entries) = std::fs::read_dir("saved_agents_weights") {
        for entry in entries.flatten() {
            if entry.path().extension().and_then(|s| s.to_str()) == Some("json") {
                if let Ok(contents) = std::fs::read_to_string(entry.path()) {
                    if let Ok(weights) = serde_json::from_str::<crate::agent::AgentWeights>(&contents) {
                        let mut dummy = Person::new(0.0, 0.0, 0, config);
                        dummy.apply_weights(&weights);
                        founders.push(dummy);
                    }
                }
            }
        }
    }
    founders
}

impl SimulationManager {
    pub fn new(width: u32, height: u32, seed: u32, count: u32, config: &crate::config::SimConfig, founders: Vec<Person>) -> Self {
        let env = Environment::new(width, height, seed, config);
        
        let mut rng = ::rand::thread_rng();
        let spawn_group_size = config.sim.spawn_group_size as usize;
        
        let get_land_spawn_point = |rng: &mut rand::rngs::ThreadRng| -> (f32, f32) {
            loop {
                let px = rng.gen_range(0.0..width as f32);
                // Spherical area-weighted Y selection (Inverse Cosine mapping)
                let u = rng.gen_range(0.0f32..1.0f32);
                let phi = (1.0 - 2.0 * u).acos();
                let py = (phi / std::f32::consts::PI) * height as f32;
                let idx = (py as usize).clamp(0, (height - 1) as usize) * (width as usize) + (px as usize).clamp(0, (width - 1) as usize);
                if env.height_map[idx] >= 0.0 { return (px, py); }
            }
        };

        let mut states = Vec::with_capacity(count as usize);
        let mut genetics = Vec::with_capacity(count as usize);
        let mut current_spawn_pt = get_land_spawn_point(&mut rng);
        let mut current_base_color = ((rng.r#gen::<f32>() * 2.0) - 1.0, (rng.r#gen::<f32>() * 2.0) - 1.0, (rng.r#gen::<f32>() * 2.0) - 1.0);
        let mut spawn_count = 0;

        let random_agents_count = if founders.is_empty() { count as usize } else { (count as f32 * config.genetics.random_spawn_percentage) as usize };
        let descendant_agents_count = count as usize - random_agents_count;
        let children_per_founder = if founders.is_empty() { 0 } else { descendant_agents_count / founders.len().max(1) };

        for _ in 0..random_agents_count {
            if spawn_count >= spawn_group_size { 
                current_spawn_pt = get_land_spawn_point(&mut rng); 
                current_base_color = ((rng.r#gen::<f32>() * 2.0) - 1.0, (rng.r#gen::<f32>() * 2.0) - 1.0, (rng.r#gen::<f32>() * 2.0) - 1.0);
                spawn_count = 0; 
            }
            let px = (current_spawn_pt.0 + rng.gen_range(-5.0f32..5.0f32)).rem_euclid(width as f32);
            let py = (current_spawn_pt.1 + rng.gen_range(-5.0f32..5.0f32)).rem_euclid(height as f32);
            let mut agent = Person::new(px, py, states.len() as u32, config);
            agent.state.pheno_r = (current_base_color.0 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
            agent.state.pheno_g = (current_base_color.1 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
            agent.state.pheno_b = (current_base_color.2 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
            states.push(agent.state);
            genetics.push(agent.genetics);
            spawn_count += 1;
        }

        if !founders.is_empty() {
            let mutation_rate = config.genetics.mutation_rate;
            let mutation_strength = config.genetics.mutation_strength;

            for founder in &founders {
                for _ in 0..children_per_founder {
                    if spawn_count >= spawn_group_size { 
                        current_spawn_pt = get_land_spawn_point(&mut rng); 
                        current_base_color = ((rng.r#gen::<f32>() * 2.0) - 1.0, (rng.r#gen::<f32>() * 2.0) - 1.0, (rng.r#gen::<f32>() * 2.0) - 1.0);
                        spawn_count = 0; 
                    }
                    let px = (current_spawn_pt.0 + rng.gen_range(-5.0f32..5.0f32)).rem_euclid(width as f32);
                    let py = (current_spawn_pt.1 + rng.gen_range(-5.0f32..5.0f32)).rem_euclid(height as f32);
                    let mut child = founder.clone_as_descendant(px, py, states.len() as u32, mutation_rate, mutation_strength, config);
                    child.state.age = rng.gen_range(0.0f32..config.bio.max_age * 0.8);
                    child.state.pheno_r = (current_base_color.0 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
                    child.state.pheno_g = (current_base_color.1 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
                    child.state.pheno_b = (current_base_color.2 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
                    states.push(child.state);
                    genetics.push(child.genetics);
                    spawn_count += 1;
                }
            }

            // Fill remaining slots
            while states.len() < count as usize {
                if spawn_count >= spawn_group_size { 
                    current_spawn_pt = get_land_spawn_point(&mut rng); 
                    current_base_color = ((rng.r#gen::<f32>() * 2.0) - 1.0, (rng.r#gen::<f32>() * 2.0) - 1.0, (rng.r#gen::<f32>() * 2.0) - 1.0);
                    spawn_count = 0; 
                }
                let px = (current_spawn_pt.0 + rng.gen_range(-5.0f32..5.0f32)).rem_euclid(width as f32);
                let py = (current_spawn_pt.1 + rng.gen_range(-5.0f32..5.0f32)).rem_euclid(height as f32);
                let mut child = founders[0].clone_as_descendant(px, py, states.len() as u32, mutation_rate, mutation_strength, config);
                child.state.age = rng.gen_range(0.0f32..config.bio.max_age * 0.8);
                child.state.pheno_r = (current_base_color.0 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
                child.state.pheno_g = (current_base_color.1 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
                child.state.pheno_b = (current_base_color.2 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
                states.push(child.state);
                genetics.push(child.genetics);
                spawn_count += 1;
            }
        }
        
        Self { 
            env, 
            states, 
            genetics,
            pending_births: std::collections::HashMap::new(),
            total_births: count as u64,
            total_deaths: 0,
            cell_occupants: std::collections::HashMap::with_capacity(count as usize / 10),
            living_ids: std::collections::HashSet::with_capacity(count as usize),
            dead_indices: Vec::with_capacity(count as usize / 4),
            births_to_process: Vec::with_capacity(count as usize / 10),
            males: Vec::with_capacity(64),
            females: Vec::with_capacity(64),
        }
    }

    pub fn process_genetics_and_births(&mut self, config: &crate::config::SimConfig) -> bool {
        self.living_ids.clear();
        for (_, vec) in self.cell_occupants.iter_mut() { vec.clear(); }
        self.dead_indices.clear();
        self.births_to_process.clear();
        
        let puberty = config.bio.puberty_age;
        let menopause = config.bio.menopause_age;

        for i in 0..self.states.len() {
            let s = &self.states[i];
            if s.health <= 0.0 {
                self.dead_indices.push(i);
            } else {
                self.living_ids.insert(s.id);
                let map_w = config.world.map_width as usize;
                let map_h = config.world.map_height as usize;
                let idx = (s.y as usize).clamp(0, map_h.saturating_sub(1)) * map_w + (s.x as usize).clamp(0, map_w.saturating_sub(1));
                self.cell_occupants.entry(idx).or_default().push(i);
            }
        }

        let living_count = self.living_ids.len();
        self.total_deaths = self.total_births.saturating_sub(living_count as u64);

        let living_ids_ref = &self.living_ids;
        self.pending_births.retain(|id, _| living_ids_ref.contains(id));

        let mut modifications = false;
        for s in &self.states {
            if s.gestation_timer <= 0.0 && self.pending_births.contains_key(&s.id) {
                self.births_to_process.push((s.id, s.x, s.y));
            }
        }

        // Use a temporary clone or separate logic to avoid borrow issues
        let births = self.births_to_process.clone();
        for (mother_id, mx, my) in births {
            if let Some(dead_idx) = self.dead_indices.pop() {
                if let Some(mut child) = self.pending_births.remove(&mother_id) {
                    child.state.x = mx;
                    child.state.y = my;
                    self.total_births += 1;
                    
                    // The child's state will be placed at dead_idx in self.states
                    // Its genetics will be placed at child.state.genetics_index in self.genetics
                    let g_idx = child.state.genetics_index as usize;
                    if g_idx < self.genetics.len() {
                        self.genetics[g_idx] = child.genetics;
                    }
                    self.states[dead_idx] = child.state;
                    
                    // Reset mother's pregnancy state
                    if let Some(mother_state) = self.states.iter_mut().find(|s| s.id == mother_id) {
                        mother_state.is_pregnant = 0.0;
                        mother_state.gestation_timer = 0.0;
                    }

                    modifications = true;
                }
            }
        }

        let reproduction_cost = config.eco.reproduction_cost;
        let mut rng = rand::thread_rng();

        // Use a temporary vector of cell indices to avoid borrowing self.cell_occupants while modifying self.states
        let active_cells: Vec<usize> = self.cell_occupants.iter()
            .filter(|(_, occupants)| occupants.len() >= 2)
            .map(|(&idx, _)| idx)
            .collect();

        for cell_idx in active_cells {
            let occupants = &self.cell_occupants[&cell_idx];
            self.males.clear();
            self.females.clear();

            for &idx in occupants {
                let s = &self.states[idx];
                let is_mature = s.age >= puberty && s.age <= menopause;
                if is_mature && s.reproduce_desire > 0.5 && s.wealth >= reproduction_cost / 2.0 && s.health > config.bio.max_health * 0.5 {
                    if s.gender > 0.5 { self.males.push(idx); } else if s.gestation_timer <= 0.0 && !self.pending_births.contains_key(&s.id) { self.females.push(idx); }
                }
            }

            while let (Some(m_idx), Some(f_idx)) = (self.males.pop(), self.females.pop()) {
                let (child, p2_id) = {
                    let m_g_idx = self.states[m_idx].genetics_index as usize;
                    let f_g_idx = self.states[f_idx].genetics_index as usize;
                    
                    let mut p1 = Person { state: self.states[m_idx], genetics: self.genetics[m_g_idx] };
                    let mut p2 = Person { state: self.states[f_idx], genetics: self.genetics[f_g_idx] };
                    
                    let child = Person::reproduce_sexual_with_rng(&mut p1, &mut p2, f_g_idx as u32, reproduction_cost, &mut rng);
                    
                    // Update parents
                    self.states[m_idx] = p1.state;
                    self.states[f_idx] = p2.state;
                    
                    (child, p2.state.id)
                };
                
                let p2_state = &mut self.states[f_idx];
                p2_state.is_pregnant = 1.0;
                p2_state.gestation_timer = config.bio.gestation_period;
                
                self.pending_births.insert(p2_id, child); 
                modifications = true;
            }
        }
        modifications
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SimConfig;

    #[test]
    fn test_simulation_manager_new() {
        let config = SimConfig::default();
        let sim = SimulationManager::new(800, 600, 12345, 100, &config, Vec::new());
        
        assert_eq!(sim.states.len(), 100);
        assert_eq!(sim.genetics.len(), 100);
        assert_eq!(sim.env.height_map.len(), 800 * 600);
        assert_eq!(sim.env.map_cells.len(), 800 * 600);
    }

    #[test]
    fn test_load_founders() {
        let config = SimConfig::default();
        let founders = load_founders(&config);
        // It should either be empty or load some default files if they exist in the workspace, 
        // but it won't crash.
        let _ = founders.len();
    }

    #[test]
    fn test_process_genetics_and_births() {
        let mut config = SimConfig::default();
        config.world.map_width = 800;
        config.world.map_height = 600;
        config.bio.puberty_age = 0.0;
        config.bio.menopause_age = 100.0;
        config.eco.reproduction_cost = 10.0;
        config.bio.max_health = 100.0;

        let mut sim = SimulationManager::new(800, 600, 12345, 10, &config, Vec::new());
        
        // Setup two agents in the same cell ready to reproduce
        sim.states[0].x = 100.0;
        sim.states[0].y = 100.0;
        sim.states[0].gender = 1.0; // Male
        sim.states[0].reproduce_desire = 1.0;
        sim.states[0].wealth = 100.0;
        sim.states[0].age = 20.0;
        sim.states[0].health = 100.0;

        sim.states[1].x = 100.1;
        sim.states[1].y = 100.1;
        sim.states[1].gender = 0.0; // Female
        sim.states[1].reproduce_desire = 1.0;
        sim.states[1].wealth = 100.0;
        sim.states[1].age = 20.0;
        sim.states[1].health = 100.0;
        sim.states[1].gestation_timer = 0.0;

        // Ensure no other agents can reproduce
        for i in 2..10 {
            sim.states[i].wealth = 0.0;
            sim.states[i].health = 100.0;
        }

        let modified = sim.process_genetics_and_births(&config);
        assert!(modified);
        assert!(!sim.pending_births.is_empty());
        
        // Advance gestation and check for birth
        sim.states[1].gestation_timer = -1.0; // Force birth
        // Set desire to 0 to prevent immediate re-conception during birth turn
        sim.states[0].reproduce_desire = 0.0;
        sim.states[1].reproduce_desire = 0.0;
        
        // Kill one agent to make space for birth
        sim.states[5].health = -1.0; 
        
        let modified_birth = sim.process_genetics_and_births(&config);
        println!("Modified birth: {}, pending births: {}", modified_birth, sim.pending_births.len());
        assert!(modified_birth);
        assert!(sim.pending_births.is_empty());
        assert!(sim.states[5].health > 0.0); // Birth should have replaced dead agent
    }
}
