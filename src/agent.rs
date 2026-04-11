/*
 * Grand Sim Pro: A high-performance GPGPU evolutionary agent simulation.
 * Part of an independent research project into emergent biological complexity.
 *
 * Copyright (C) 2026 Ved Hirenkumar Suthar
 * Licensed under the GNU General Public License v3.0 or later.
 * * This software is provided "as is", without warranty of any kind.
 * See the LICENSE file in the project root for full license details.
 */

use std::f32::consts::PI;
use rand::Rng;
use serde::{Serialize, Deserialize};

pub const NUM_INPUTS: usize = 160;
pub const NUM_HIDDEN_MAX: usize = 64;
pub const NUM_OUTPUTS: usize = 31;
pub const W1_SIZE: usize = NUM_HIDDEN_MAX * 8; // Sparse Fixed-K Connectivity
pub const W2_SIZE: usize = NUM_HIDDEN_MAX * NUM_HIDDEN_MAX;
pub const W3_SIZE: usize = NUM_HIDDEN_MAX * NUM_OUTPUTS; // 64 * 31 = 1984

pub const INPUT_LABELS: [&str; NUM_INPUTS] = [
    "Bias", "Local Res", "Local Pop", "Avg Speed", "Avg Share", "Avg Repro", "Avg Aggr", "Avg Preg",
    "Avg Turn", "Avg Rest", "Comm 1", "Comm 2", "Comm 3", "Comm 4", "Health", "Food", "Water", "Stamina",
    "Age", "Gender", "Temp", "Season", "Is Preg", "Encumbrance", "Crowding",
    "Mem 1", "Mem 2", "Mem 3", "Mem 4", "Mem 5", "Mem 6", "Mem 7", "Mem 8", "Wealth", "Avg Ask", "Avg Bid", "Daylight",
    "Own Pheno R", "Own Pheno G", "Own Pheno B",
    "Loc Pheno R", "Loc Pheno G", "Loc Pheno B",
    "FL Res", "FL Elev", "FL Pop", "FL C1", "FL C2", "FL C3", "FL C4", "FL PR", "FL PG", "FL PB", "FL Road", "FL House", "FL Farm", "FL Store",
    "F Res", "F Elev", "F Pop", "F C1", "F C2", "F C3", "F C4", "F PR", "F PG", "F PB", "F Road", "F House", "F Farm", "F Store",
    "FR Res", "FR Elev", "FR Pop", "FR C1", "FR C2", "FR C3", "FR C4", "FR PR", "FR PG", "FR PB", "FR Road", "FR House", "FR Farm", "FR Store",
    "L Res", "L Elev", "L Pop", "L C1", "L C2", "L C3", "L C4", "L PR", "L PG", "L PB", "L Road", "L House", "L Farm", "L Store",
    "R Res", "R Elev", "R Pop", "R C1", "R C2", "R C3", "R C4", "R PR", "R PG", "R PB", "R Road", "R House", "R Farm", "R Store",
    "BL Res", "BL Elev", "BL Pop", "BL C1", "BL C2", "BL C3", "BL C4", "BL PR", "BL PG", "BL PB", "BL Road", "BL House", "BL Farm", "BL Store",
    "B Res", "B Elev", "B Pop", "B C1", "B C2", "B C3", "B C4", "B PR", "B PG", "B PB", "B Road", "B House", "B Farm", "B Store",
    "BR Res", "BR Elev", "BR Pop", "BR C1", "BR C2", "BR C3", "BR C4", "BR PR", "BR PG", "BR PB", "BR Road", "BR House", "BR Farm", "BR Store",
    "Loc Road", "Loc House", "Loc Farm", "Loc Storage", "Pad 1"
];

pub const OUTPUT_LABELS: [&str; NUM_OUTPUTS] = [
    "Turn", "Speed", "Drop Res", "Reproduce", "Attack", "Rest", "Comm 1", "Comm 2", "Comm 3", "Comm 4",
    "Learn", "Mem 1", "Mem 2", "Mem 3", "Mem 4", "Mem 5", "Mem 6", "Mem 7", "Mem 8",
    "Buy Intent", "Sell Intent", "Ask Price", "Bid Price", "Drop H2O", "Pickup H2O", "Defend Intent", "Build Road", "Build House", "Build Farm", "Build Storage",
    "Destroy Infra"
];

#[derive(Serialize, Deserialize)]
pub struct AgentWeights {
    pub hidden_count: u32,
    #[serde(default)]
    pub inputs: std::collections::HashMap<String, Vec<f32>>,
    #[serde(default)]
    pub w1_weights: Vec<f32>,
    #[serde(default)]
    pub w1_indices: Vec<u32>,
    pub w2: Vec<f32>,
    #[serde(default)]
    pub outputs: std::collections::HashMap<String, Vec<f32>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub w1: Vec<f32>, // Kept for backwards compatibility with older saves
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub w3: Vec<f32>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AgentState {
    pub x: f32,
    pub y: f32,
    pub heading: f32,
    pub speed: f32,
    pub hidden_count: u32,
    pub genetics_index: u32, // Indirection to Genetics buffer
    pub gender: f32, 
    pub reproduce_desire: f32,
    pub attack_intent: f32,
    pub rest_intent: f32,
    pub comm1: f32,
    pub comm2: f32,
    pub comm3: f32,
    pub comm4: f32,
    pub mem1: f32,
    pub mem2: f32,
    pub mem3: f32,
    pub mem4: f32,
    pub mem5: f32,
    pub mem6: f32,
    pub mem7: f32,
    pub mem8: f32,
    pub buy_intent: f32,
    pub sell_intent: f32,
    pub ask_price: f32,
    pub bid_price: f32,
    pub wealth: f32,
    pub drop_water_intent: f32,
    pub pickup_water_intent: f32,
    pub defend_intent: f32,
    pub build_road_intent: f32,
    pub build_house_intent: f32,
    pub build_farm_intent: f32,
    pub build_storage_intent: f32,
    pub destroy_infra_intent: f32,
    pub pheno_r: f32, 
    pub pheno_g: f32,
    pub pheno_b: f32,
    pub _pad_agent1: f32,
    pub _pad_agent2: f32,
    pub _pad_agent3: f32,
    pub food: f32,     
    pub water: f32,
    pub stamina: f32,
    pub health: f32,
    pub age: f32,
    pub id: u32,       
    pub gestation_timer: f32,
    pub is_pregnant: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Genetics {
    pub w1_weights: [f32; W1_SIZE],
    pub w1_indices: [u32; W1_SIZE],
    pub w2: [f32; W2_SIZE],
    pub w3: [f32; W3_SIZE],
}

#[derive(Clone, Copy)]
pub struct Person {
    pub state: AgentState,
    pub genetics: Genetics,
}

impl Person {
    pub fn new(x: f32, y: f32, genetics_index: u32, config: &crate::config::SimConfig) -> Self {
        let hidden_count = 16;

        let mut rng = rand::thread_rng();
        
        let w1_limit = (6.0 / (8.0 + hidden_count as f32)).sqrt(); 
        let w2_limit = (6.0 / (hidden_count as f32 + hidden_count as f32)).sqrt();
        let w3_limit = (6.0 / (hidden_count as f32 + NUM_OUTPUTS as f32)).sqrt();

        let mut w1_weights = [0.0; W1_SIZE];
        let mut w1_indices = [0; W1_SIZE];
        for h in 0..NUM_HIDDEN_MAX {
            let mut available: Vec<u32> = (0..NUM_INPUTS as u32).collect();
            for k in 0..8 {
                let idx = rng.gen_range(0..available.len());
                w1_indices[h * 8 + k] = available.remove(idx);
                w1_weights[h * 8 + k] = (rng.r#gen::<f32>() * 2.0 * w1_limit) - w1_limit;
            }
        }

        let mut w2 = [0.0; W2_SIZE];
        for i in 0..hidden_count {
            for j in 0..hidden_count {
                w2[i * NUM_HIDDEN_MAX + j] = (rng.r#gen::<f32>() * 2.0 * w2_limit) - w2_limit;
            }
        }
        let mut w3 = [0.0; W3_SIZE];
        for i in 0..hidden_count {
            for j in 0..NUM_OUTPUTS {
                w3[i * NUM_OUTPUTS + j] = (rng.r#gen::<f32>() * 2.0 * w3_limit) - w3_limit;
            }
        }

        Self {
            state: AgentState {
                x,
                y,
                heading: rng.r#gen::<f32>() * PI * 2.0,
                speed: 1.4,
                hidden_count: hidden_count as u32,
                genetics_index,
                gender: if rng.r#gen::<f32>() > 0.5 { 1.0 } else { 0.0 }, 
                reproduce_desire: 0.0,
                attack_intent: 0.0,
                rest_intent: 0.0,
                comm1: 0.0,
                comm2: 0.0,
                comm3: 0.0,
                comm4: 0.0,
                mem1: 0.0,
                mem2: 0.0,
                mem3: 0.0,
                mem4: 0.0,
                mem5: 0.0,
                mem6: 0.0,
                mem7: 0.0,
                mem8: 0.0,
                buy_intent: 0.0,
                sell_intent: 0.0,
                ask_price: 1.0,
                bid_price: 1.0,
                wealth: 500.0,
                drop_water_intent: 0.0,
                pickup_water_intent: 0.0,
                defend_intent: 0.0,
                build_road_intent: 0.0,
                build_house_intent: 0.0,
                build_farm_intent: 0.0,
                build_storage_intent: 0.0,
                destroy_infra_intent: 0.0,
                pheno_r: (rng.r#gen::<f32>() * 2.0) - 1.0,
                pheno_g: (rng.r#gen::<f32>() * 2.0) - 1.0,
                pheno_b: (rng.r#gen::<f32>() * 2.0) - 1.0,
                _pad_agent1: 0.0,
                _pad_agent2: 0.0,
                _pad_agent3: 0.0,
                food: 50000.0, 
                water: config.bio.max_water,
                stamina: 100.0,
                health: 100.0,
                age: rng.r#gen::<f32>() * config.bio.max_age * 0.5, 
                id: rng.r#gen::<u32>(),
                gestation_timer: 0.0,
                is_pregnant: 0.0,
            },
            genetics: Genetics {
                w1_weights,
                w1_indices,
                w2,
                w3,
            }
        }
    }

    pub fn reproduce_sexual(parent1: &mut Self, parent2: &mut Self, genetics_index: u32, cost: f32) -> Self {
        let mut rng = rand::thread_rng();
        Self::reproduce_sexual_with_rng(parent1, parent2, genetics_index, cost, &mut rng)
    }

    pub fn reproduce_sexual_with_rng<R: Rng>(parent1: &mut Self, parent2: &mut Self, genetics_index: u32, cost: f32, rng: &mut R) -> Self {
        parent1.state.wealth -= cost / 2.0; 
        parent2.state.wealth -= cost / 2.0; 
        
        let mut child = *parent1;
        child.state.age = 0.0;
        child.state.health = 100.0;
        child.state.food = 50000.0;
        child.state.wealth = cost / 2.0; // Inherit seed money
        child.state.water = parent1.state.water; // Child inherits water from parent, or could be config.bio.max_water
        child.state.stamina = 100.0;
        
        child.state.gender = if rng.r#gen::<f32>() > 0.5 { 1.0 } else { 0.0 };
        child.state.genetics_index = genetics_index;
        child.state.reproduce_desire = 0.0;
        child.state.attack_intent = 0.0;
        child.state.rest_intent = 0.0;
        child.state.comm1 = 0.0;
        child.state.comm2 = 0.0;
        child.state.comm3 = 0.0;
        child.state.comm4 = 0.0;
        child.state.mem1 = 0.0;
        child.state.mem2 = 0.0;
        child.state.mem3 = 0.0;
        child.state.mem4 = 0.0;
        child.state.mem5 = 0.0;
        child.state.mem6 = 0.0;
        child.state.mem7 = 0.0;
        child.state.mem8 = 0.0;
        child.state.buy_intent = 0.0;
        child.state.sell_intent = 0.0;
        child.state.ask_price = 1.0;
        child.state.bid_price = 1.0;
        child.state.drop_water_intent = 0.0;
        child.state.pickup_water_intent = 0.0;
        child.state.defend_intent = 0.0;
        child.state.build_road_intent = 0.0;
        child.state.build_house_intent = 0.0;
        child.state.build_farm_intent = 0.0;
        child.state.build_storage_intent = 0.0;
        child.state.destroy_infra_intent = 0.0;
        child.state._pad_agent1 = 0.0;
        child.state._pad_agent2 = 0.0;
        child.state._pad_agent3 = 0.0;
        child.state.id = rng.r#gen::<u32>();
        child.state.gestation_timer = 0.0;
        child.state.is_pregnant = 0.0;
        
        child.state.pheno_r = if rng.r#gen::<f32>() > 0.5 { parent1.state.pheno_r } else { parent2.state.pheno_r };
        child.state.pheno_g = if rng.r#gen::<f32>() > 0.5 { parent1.state.pheno_g } else { parent2.state.pheno_g };
        child.state.pheno_b = if rng.r#gen::<f32>() > 0.5 { parent1.state.pheno_b } else { parent2.state.pheno_b };
        if rng.r#gen::<f32>() < 0.1 { child.state.pheno_r = (child.state.pheno_r + (rng.r#gen::<f32>() * 0.4) - 0.2).clamp(-1.0, 1.0); }
        if rng.r#gen::<f32>() < 0.1 { child.state.pheno_g = (child.state.pheno_g + (rng.r#gen::<f32>() * 0.4) - 0.2).clamp(-1.0, 1.0); }
        if rng.r#gen::<f32>() < 0.1 { child.state.pheno_b = (child.state.pheno_b + (rng.r#gen::<f32>() * 0.4) - 0.2).clamp(-1.0, 1.0); }

        // 1. Crossover Genetics and Mutate
        // High-performance block crossover to reduce RNG calls
        let crossover_rate = 0.5;
        let mutation_rate = 0.1;
        
        // Block-based crossover for W1
        for i in (0..child.genetics.w1_weights.len()).step_by(64) {
            if rng.r#gen::<f32>() < crossover_rate {
                let end = (i + 64).min(child.genetics.w1_weights.len());
                child.genetics.w1_weights[i..end].copy_from_slice(&parent2.genetics.w1_weights[i..end]);
                child.genetics.w1_indices[i..end].copy_from_slice(&parent2.genetics.w1_indices[i..end]);
            }
            if rng.r#gen::<f32>() < mutation_rate {
                let end = (i + 64).min(child.genetics.w1_weights.len());
                let m = (rng.r#gen::<f32>() * 0.5) - 0.25;
                for j in i..end { child.genetics.w1_weights[j] = (child.genetics.w1_weights[j] + m).clamp(-2.0, 2.0); }
                if rng.r#gen::<f32>() < 0.05 {
                    let k = rng.gen_range(i..end);
                    child.genetics.w1_indices[k] = rng.gen_range(0..NUM_INPUTS as u32);
                }
            }
        }
        
        // Use larger strides for w2 and w3 to minimize overhead
        for i in (0..child.genetics.w2.len()).step_by(256) {
            if rng.r#gen::<f32>() < crossover_rate {
                let end = (i + 256).min(child.genetics.w2.len());
                child.genetics.w2[i..end].copy_from_slice(&parent2.genetics.w2[i..end]);
            }
        }
        for i in (0..child.genetics.w2.len()).step_by(64) {
            if rng.r#gen::<f32>() < mutation_rate { 
                let end = (i + 64).min(child.genetics.w2.len());
                let m = (rng.r#gen::<f32>() * 0.5) - 0.25;
                for j in i..end { child.genetics.w2[j] = (child.genetics.w2[j] + m).clamp(-2.0, 2.0); }
            }
        }

        for i in (0..child.genetics.w3.len()).step_by(128) {
            if rng.r#gen::<f32>() < crossover_rate {
                let end = (i + 128).min(child.genetics.w3.len());
                child.genetics.w3[i..end].copy_from_slice(&parent2.genetics.w3[i..end]);
            }
        }
        for i in (0..child.genetics.w3.len()).step_by(64) {
            if rng.r#gen::<f32>() < mutation_rate { 
                let end = (i + 64).min(child.genetics.w3.len());
                let m = (rng.r#gen::<f32>() * 0.5) - 0.25;
                for j in i..end { child.genetics.w3[j] = (child.genetics.w3[j] + m).clamp(-2.0, 2.0); }
            }
        }

        // 2. Structural mutation
        if rng.r#gen::<f32>() < 0.05 && child.state.hidden_count < NUM_HIDDEN_MAX as u32 {
            let h = child.state.hidden_count as usize;
            
            let w1_limit = (6.0 / (8.0 + h as f32 + 1.0)).sqrt();
            let w2_limit = (6.0 / ((h as f32 + 1.0) * 2.0)).sqrt();
            let w3_limit = (6.0 / (h as f32 + 1.0 + NUM_OUTPUTS as f32)).sqrt();

            let mut available: Vec<u32> = (0..NUM_INPUTS as u32).collect();
            for k in 0..8 {
                let idx = rng.gen_range(0..available.len());
                child.genetics.w1_indices[h * 8 + k] = available.remove(idx);
                child.genetics.w1_weights[h * 8 + k] = (rng.r#gen::<f32>() * 2.0 * w1_limit) - w1_limit;
            }

            for i in 0..NUM_HIDDEN_MAX { child.genetics.w2[h * NUM_HIDDEN_MAX + i] = (rng.r#gen::<f32>() * 2.0 * w2_limit) - w2_limit; } // H1 out
            for i in 0..NUM_HIDDEN_MAX { child.genetics.w2[i * NUM_HIDDEN_MAX + h] = (rng.r#gen::<f32>() * 2.0 * w2_limit) - w2_limit; } // H2 in
            for i in 0..NUM_OUTPUTS { child.genetics.w3[h * NUM_OUTPUTS + i] = (rng.r#gen::<f32>() * 2.0 * w3_limit) - w3_limit; } 
            child.state.hidden_count += 1;
        }
        
        child
    }

    pub fn clone_as_descendant(&self, x: f32, y: f32, genetics_index: u32, mutation_rate: f32, mutation_strength: f32, config: &crate::config::SimConfig) -> Self {
        let mut child = *self;
        child.state.age = 0.0;
        child.state.health = 100.0;
        child.state.food = 50000.0; 
        child.state.water = config.bio.max_water;
        child.state.stamina = 100.0;
        child.state.wealth = 500.0;
        child.state.x = x;
        child.state.y = y;
        child.state.genetics_index = genetics_index;
        
        let mut rng = rand::thread_rng();
        child.state.gender = if rng.r#gen::<f32>() > 0.5 { 1.0 } else { 0.0 };
        child.state.reproduce_desire = 0.0;
        child.state.attack_intent = 0.0;
        child.state.rest_intent = 0.0;
        child.state.comm1 = 0.0;
        child.state.comm2 = 0.0;
        child.state.comm3 = 0.0;
        child.state.comm4 = 0.0;
        child.state.mem1 = 0.0;
        child.state.mem2 = 0.0;
        child.state.mem3 = 0.0;
        child.state.mem4 = 0.0;
        child.state.mem5 = 0.0;
        child.state.mem6 = 0.0;
        child.state.mem7 = 0.0;
        child.state.mem8 = 0.0;
        child.state.buy_intent = 0.0;
        child.state.sell_intent = 0.0;
        child.state.ask_price = 1.0;
        child.state.bid_price = 1.0;
        child.state.drop_water_intent = 0.0;
        child.state.pickup_water_intent = 0.0;
        child.state.defend_intent = 0.0;
        child.state.build_road_intent = 0.0;
        child.state.build_house_intent = 0.0;
        child.state.build_farm_intent = 0.0;
        child.state.build_storage_intent = 0.0;
        child.state.destroy_infra_intent = 0.0;
        child.state._pad_agent1 = 0.0;
        child.state._pad_agent2 = 0.0;
        child.state._pad_agent3 = 0.0;
        child.state.id = rng.r#gen::<u32>();
        child.state.gestation_timer = 0.0;
        child.state.is_pregnant = 0.0;
        child.state.heading = rng.r#gen::<f32>() * std::f32::consts::PI * 2.0;
        
        if rng.r#gen::<f32>() < mutation_rate { child.state.pheno_r = (child.state.pheno_r + (rng.r#gen::<f32>() * 2.0 * mutation_strength) - mutation_strength).clamp(-1.0, 1.0); }
        if rng.r#gen::<f32>() < mutation_rate { child.state.pheno_g = (child.state.pheno_g + (rng.r#gen::<f32>() * 2.0 * mutation_strength) - mutation_strength).clamp(-1.0, 1.0); }
        if rng.r#gen::<f32>() < mutation_rate { child.state.pheno_b = (child.state.pheno_b + (rng.r#gen::<f32>() * 2.0 * mutation_strength) - mutation_strength).clamp(-1.0, 1.0); }

        for w in child.genetics.w1_weights.iter_mut() {
            if rng.r#gen::<f32>() < mutation_rate { *w = (*w + (rng.r#gen::<f32>() * 2.0 * mutation_strength) - mutation_strength).clamp(-2.0, 2.0); }
        }
        for idx in child.genetics.w1_indices.iter_mut() {
            if rng.r#gen::<f32>() < mutation_rate * 0.1 { *idx = rng.gen_range(0..NUM_INPUTS as u32); }
        }
        for w in child.genetics.w2.iter_mut() {
            if rng.r#gen::<f32>() < mutation_rate { *w = (*w + (rng.r#gen::<f32>() * 2.0 * mutation_strength) - mutation_strength).clamp(-2.0, 2.0); }
        }
        for w in child.genetics.w3.iter_mut() {
            if rng.r#gen::<f32>() < mutation_rate { *w = (*w + (rng.r#gen::<f32>() * 2.0 * mutation_strength) - mutation_strength).clamp(-2.0, 2.0); }
        }
        child
    }

    pub fn extract_weights(&self) -> AgentWeights {
        let mut inputs = std::collections::HashMap::new();
        for i in 0..NUM_INPUTS {
            inputs.insert(INPUT_LABELS[i].to_string(), vec![0.0; NUM_HIDDEN_MAX]);
        }

        for h in 0..NUM_HIDDEN_MAX {
            if (h as u32) < self.state.hidden_count {
                for k in 0..8 {
                    let conn_idx = h * 8 + k;
                    let input_idx = self.genetics.w1_indices[conn_idx] as usize;
                    let weight = self.genetics.w1_weights[conn_idx];

                    if let Some(label) = INPUT_LABELS.get(input_idx) {
                        if let Some(weights_vec) = inputs.get_mut(*label) {
                            weights_vec[h] = weight;
                        }
                    }
                }
            }
        }

        let mut outputs = std::collections::HashMap::new();
        for o in 0..NUM_OUTPUTS {
            let mut w = Vec::with_capacity(NUM_HIDDEN_MAX);
            for h in 0..NUM_HIDDEN_MAX {
                w.push(self.genetics.w3[h * NUM_OUTPUTS + o]);
            }
            outputs.insert(OUTPUT_LABELS[o].to_string(), w);
        }

        AgentWeights {
            hidden_count: self.state.hidden_count,
            inputs,
            w1_weights: self.genetics.w1_weights.to_vec(),
            w1_indices: self.genetics.w1_indices.to_vec(),
            w2: self.genetics.w2.to_vec(),
            outputs,
            w1: Vec::new(),
            w3: Vec::new(),
        }
    }

    pub fn apply_weights(&mut self, weights: &AgentWeights) {
        self.state.hidden_count = weights.hidden_count.min(NUM_HIDDEN_MAX as u32);
        
        if !weights.w1_weights.is_empty() && !weights.w1_indices.is_empty() {
            let w1_len = self.genetics.w1_weights.len().min(weights.w1_weights.len());
            self.genetics.w1_weights[..w1_len].copy_from_slice(&weights.w1_weights[..w1_len]);
            let idx_len = self.genetics.w1_indices.len().min(weights.w1_indices.len());
            self.genetics.w1_indices[..idx_len].copy_from_slice(&weights.w1_indices[..idx_len]);
        }

        let w2_len = self.genetics.w2.len().min(weights.w2.len());
        self.genetics.w2[..w2_len].copy_from_slice(&weights.w2[..w2_len]);
        
        if !weights.outputs.is_empty() {
            for o in 0..NUM_OUTPUTS {
                if let Some(w) = weights.outputs.get(OUTPUT_LABELS[o]) {
                    for h in 0..NUM_HIDDEN_MAX {
                        if h < w.len() { self.genetics.w3[h * NUM_OUTPUTS + o] = w[h]; }
                    }
                }
            }
        } else if !weights.w3.is_empty() {
            let w3_len = self.genetics.w3.len().min(weights.w3.len());
            self.genetics.w3[..w3_len].copy_from_slice(&weights.w3[..w3_len]);
        }
    }

    /// Performs a CPU-side forward pass of the agent's neural network.
    /// Used for behavioral prediction and UI probing.
    pub fn calculate_input_output_influence(&self) -> Vec<f32> {
        let h_count = self.state.hidden_count as usize;
        let mut influence = vec![0.0f32; NUM_INPUTS * NUM_OUTPUTS];
        
        // 1. Calculate Input -> Hidden2 influence (W2 * sparse W1)
        // Hidden1 = activation(W1 * Input)
        // Hidden2 = activation(W2 * Hidden1)
        // Output  = activation(W3 * Hidden2)
        
        let mut w2_w1 = vec![0.0f32; NUM_INPUTS * h_count];
        for h2 in 0..h_count {
            for h1 in 0..h_count {
                let w2_val = self.genetics.w2[h1 * NUM_HIDDEN_MAX + h2];
                for k in 0..8 {
                    let w1_val = self.genetics.w1_weights[h1 * 8 + k];
                    let input_idx = self.genetics.w1_indices[h1 * 8 + k] as usize;
                    w2_w1[h2 * NUM_INPUTS + input_idx] += w2_val * w1_val;
                }
            }
        }
        
        // 2. Calculate Input -> Output influence (W3 * w2_w1)
        for o in 0..NUM_OUTPUTS {
            for h2 in 0..h_count {
                let w3_val = self.genetics.w3[h2 * NUM_OUTPUTS + o];
                for i in 0..NUM_INPUTS {
                    let prev_inf = w2_w1[h2 * NUM_INPUTS + i];
                    influence[i * NUM_OUTPUTS + o] += w3_val * prev_inf;
                }
            }
        }
        
        influence
    }

    pub fn mental_simulation(&self, inputs: &[f32; NUM_INPUTS]) -> [f32; NUM_OUTPUTS] {
        let mut hidden = [0.0f32; NUM_HIDDEN_MAX];
        
        // Layer 1: Sparse Input -> Hidden
        for h in 0..self.state.hidden_count as usize {
            let mut sum = 0.0;
            for k in 0..8 {
                let idx = self.genetics.w1_indices[h * 8 + k] as usize;
                if idx < NUM_INPUTS {
                    sum += inputs[idx] * self.genetics.w1_weights[h * 8 + k];
                }
            }
            hidden[h] = sum.max(0.0); // ReLU
        }

        // Layer 2: Hidden -> Hidden (Dense)
        let mut hidden2 = [0.0f32; NUM_HIDDEN_MAX];
        for h in 0..self.state.hidden_count as usize {
            let mut sum = 0.0;
            for prev_h in 0..self.state.hidden_count as usize {
                sum += hidden[prev_h] * self.genetics.w2[prev_h * NUM_HIDDEN_MAX + h];
            }
            hidden2[h] = sum.max(0.0); // ReLU
        }

        // Layer 3: Hidden -> Output (Dense)
        let mut outputs = [0.0f32; NUM_OUTPUTS];
        for o in 0..NUM_OUTPUTS {
            let mut sum = 0.0;
            for h in 0..self.state.hidden_count as usize {
                sum += hidden2[h] * self.genetics.w3[h * NUM_OUTPUTS + o];
            }
            // Use tanh for directional/bipolar outputs, sigmoid for intents
            // For simplicity in UI probing, we just return the raw sum or clamped value
            outputs[o] = sum.clamp(-1.0, 1.0);
        }
        
        outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SimConfig;

    #[test]
    fn test_person_new() {
        let config = SimConfig::default();
        let p = Person::new(10.0, 20.0, 0, &config);
        assert_eq!(p.state.x, 10.0);
        assert_eq!(p.state.y, 20.0);
        assert_eq!(p.state.genetics_index, 0);
        assert!(p.state.health > 0.0);
        assert!(p.state.food > 0.0);
        assert!(p.state.water > 0.0);
        assert!(p.state.wealth > 0.0);
    }

    #[test]
    fn test_reproduce_sexual() {
        let config = SimConfig::default();
        let mut p1 = Person::new(10.0, 20.0, 0, &config);
        let mut p2 = Person::new(15.0, 25.0, 1, &config);
        p1.state.wealth = 1000.0;
        p2.state.wealth = 1000.0;
        let cost = 500.0;
        
        let child = Person::reproduce_sexual(&mut p1, &mut p2, 2, cost);
        
        assert_eq!(p1.state.wealth, 750.0);
        assert_eq!(p2.state.wealth, 750.0);
        assert_eq!(child.state.wealth, 250.0);
        assert_eq!(child.state.genetics_index, 2);
        assert_eq!(child.state.age, 0.0);
        assert_eq!(child.state.health, 100.0);
    }

    #[test]
    fn test_clone_as_descendant() {
        let config = SimConfig::default();
        let p = Person::new(10.0, 20.0, 0, &config);
        let child = p.clone_as_descendant(30.0, 40.0, 1, 0.1, 0.1, &config);
        
        assert_eq!(child.state.x, 30.0);
        assert_eq!(child.state.y, 40.0);
        assert_eq!(child.state.genetics_index, 1);
        assert_eq!(child.state.age, 0.0);
        assert_eq!(child.state.health, 100.0);
    }

    #[test]
    fn test_extract_apply_weights() {
        let config = SimConfig::default();
        let p = Person::new(10.0, 20.0, 0, &config);
        let weights = p.extract_weights();
        
        let mut p2 = Person::new(0.0, 0.0, 1, &config);
        p2.apply_weights(&weights);
        
        assert_eq!(p2.state.hidden_count, p.state.hidden_count);
        assert_eq!(p2.genetics.w2, p.genetics.w2);
        assert_eq!(p2.genetics.w3, p.genetics.w3);
        assert_eq!(p2.genetics.w1_weights, p.genetics.w1_weights);
        assert_eq!(p2.genetics.w1_indices, p.genetics.w1_indices);
    }

    #[test]
    fn test_mental_simulation_sanity() {
        let config = SimConfig::default();
        let p = Person::new(0.0, 0.0, 0, &config);
        let mut inputs = [0.0f32; NUM_INPUTS];
        inputs[0] = 1.0; // Bias input
        
        let outputs = p.mental_simulation(&inputs);
        // Ensure outputs are within expected range [-1, 1]
        for &o in outputs.iter() {
            assert!(o >= -1.0 && o <= 1.0);
        }
    }
}
