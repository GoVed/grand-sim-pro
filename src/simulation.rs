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
        
        let mut founders = Vec::new();
        if config.load_saved_agents_on_start > 0 {
            if let Ok(entries) = std::fs::read_dir("saved_agents_weights") {
                for entry in entries.flatten() {
                    if entry.path().extension().and_then(|s| s.to_str()) == Some("json") {
                        if let Ok(contents) = std::fs::read_to_string(entry.path()) {
                            if let Ok(weights) = serde_json::from_str::<crate::agent::AgentWeights>(&contents) {
                                let mut dummy = Person::new(0.0, 0.0, config);
                                dummy.apply_weights(&weights);
                                founders.push(dummy);
                            }
                        }
                    }
                }
            }
        }

        let mut rng = ::rand::thread_rng();
        let spawn_group_size = config.spawn_group_size as usize;
        
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

        let mut agents = Vec::with_capacity(count as usize);
        let mut current_spawn_pt = get_land_spawn_point(&mut rng);
        let mut current_base_color = ((rng.r#gen::<f32>() * 2.0) - 1.0, (rng.r#gen::<f32>() * 2.0) - 1.0, (rng.r#gen::<f32>() * 2.0) - 1.0);
        let mut spawn_count = 0;

        let random_agents_count = if founders.is_empty() { count as usize } else { (count as f32 * config.random_spawn_percentage) as usize };
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
            let mut agent = Person::new(px, py, config);
            agent.pheno_r = (current_base_color.0 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
            agent.pheno_g = (current_base_color.1 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
            agent.pheno_b = (current_base_color.2 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
            agents.push(agent);
            spawn_count += 1;
        }

        if !founders.is_empty() {
            let mutation_rate = config.mutation_rate;
            let mutation_strength = config.mutation_strength;

            for founder in &founders {
                for _ in 0..children_per_founder {
                    if spawn_count >= spawn_group_size { 
                        current_spawn_pt = get_land_spawn_point(&mut rng); 
                        current_base_color = ((rng.r#gen::<f32>() * 2.0) - 1.0, (rng.r#gen::<f32>() * 2.0) - 1.0, (rng.r#gen::<f32>() * 2.0) - 1.0);
                        spawn_count = 0; 
                    }
                    let px = (current_spawn_pt.0 + rng.gen_range(-5.0f32..5.0f32)).rem_euclid(width as f32);
                    let py = (current_spawn_pt.1 + rng.gen_range(-5.0f32..5.0f32)).rem_euclid(height as f32);
                    let mut child = founder.clone_as_descendant(px, py, mutation_rate, mutation_strength, config);
                    child.age = rng.gen_range(0.0f32..config.max_age * 0.8);
                    child.pheno_r = (current_base_color.0 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
                    child.pheno_g = (current_base_color.1 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
                    child.pheno_b = (current_base_color.2 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
                    agents.push(child);
                    spawn_count += 1;
                }
            }

            // Fill remaining slots
            while agents.len() < count as usize {
                if spawn_count >= spawn_group_size { 
                    current_spawn_pt = get_land_spawn_point(&mut rng); 
                    current_base_color = ((rng.r#gen::<f32>() * 2.0) - 1.0, (rng.r#gen::<f32>() * 2.0) - 1.0, (rng.r#gen::<f32>() * 2.0) - 1.0);
                    spawn_count = 0; 
                }
                let px = (current_spawn_pt.0 + rng.gen_range(-5.0f32..5.0f32)).rem_euclid(width as f32);
                let py = (current_spawn_pt.1 + rng.gen_range(-5.0f32..5.0f32)).rem_euclid(height as f32);
                let mut child = founders[0].clone_as_descendant(px, py, mutation_rate, mutation_strength, config);
                child.age = rng.gen_range(0.0f32..config.max_age * 0.8);
                child.pheno_r = (current_base_color.0 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
                child.pheno_g = (current_base_color.1 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
                child.pheno_b = (current_base_color.2 + rng.gen_range(-0.15f32..0.15f32)).clamp(-1.0, 1.0);
                agents.push(child);
                spawn_count += 1;
            }
        }
        
        Self { env, agents, pending_births: std::collections::HashMap::new() }
    }

    pub fn process_genetics_and_births(&mut self, config: &crate::config::SimConfig) -> bool {
        use std::collections::{HashMap, HashSet};
        let mut living_ids = HashSet::new();
        let mut cell_occupants: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut dead_indices = Vec::new();
        
        let puberty = config.puberty_age;
        let menopause = config.menopause_age;

        for i in 0..self.agents.len() {
            let a = &self.agents[i];
            if a.health <= 0.0 {
                dead_indices.push(i);
            } else {
                living_ids.insert(a.id);
                let map_w = config.map_width as usize;
                let map_h = config.map_height as usize;
                let idx = (a.y as usize).clamp(0, map_h.saturating_sub(1)) * map_w + (a.x as usize).clamp(0, map_w.saturating_sub(1));
                cell_occupants.entry(idx).or_default().push(i);
            }
        }

        self.pending_births.retain(|id, _| living_ids.contains(id));

        let mut modifications = false;
        let mut births_to_process = Vec::new();
        for mother in &self.agents {
            if mother.gestation_timer <= 0.0 && self.pending_births.contains_key(&mother.id) {
                births_to_process.push((mother.id, mother.x, mother.y));
            }
        }

        for (mother_id, mx, my) in births_to_process {
            if let Some(dead_idx) = dead_indices.pop() {
                if let Some(mut child) = self.pending_births.remove(&mother_id) {
                    child.x = mx;
                    child.y = my;
                    self.agents[dead_idx] = child;
                    modifications = true;
                }
            }
        }

        let reproduction_cost = config.reproduction_cost;
        for (_cell_idx, occupants) in cell_occupants {
            if occupants.len() < 2 { continue; } 

            let mut males = Vec::new();
            let mut females = Vec::new();

            for &idx in &occupants {
                let a = &self.agents[idx];
                let is_mature = a.age >= puberty && a.age <= menopause;
                if is_mature && a.reproduce_desire > 0.5 && a.wealth >= reproduction_cost / 2.0 && a.health > config.max_health * 0.5 {
                    if a.gender > 0.5 { males.push(idx); } else if a.gestation_timer <= 0.0 && !self.pending_births.contains_key(&a.id) { females.push(idx); }
                }
            }

            while let (Some(m_idx), Some(f_idx)) = (males.pop(), females.pop()) {
                let mut p1 = self.agents[m_idx].clone();
                let mut p2 = self.agents[f_idx].clone();
                let child = Person::reproduce_sexual(&mut p1, &mut p2, reproduction_cost);
                self.agents[m_idx] = p1; 
                p2.is_pregnant = 1.0;
                p2.gestation_timer = config.gestation_period;
                self.agents[f_idx] = p2;
                self.pending_births.insert(p2.id, child); 
                modifications = true;
            }
        }
        modifications
    }
}