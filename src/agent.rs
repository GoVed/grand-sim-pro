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
pub struct Person {
    pub x: f32,
    pub y: f32,
    pub heading: f32,
    pub speed: f32,
    pub hidden_count: u32,
    pub gender: f32, // 1.0 for Male, 0.0 for Female
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
    pub pheno_r: f32, // Visual/Pheromone marker
    pub pheno_g: f32,
    pub pheno_b: f32,
    pub _pad_agent1: f32, // Structural padding to keep memory strictly 16-byte aligned
    pub _pad_agent2: f32,
    pub _pad_agent3: f32,
    pub w1_weights: [f32; W1_SIZE],
    pub w1_indices: [u32; W1_SIZE],
    pub w2: [f32; W2_SIZE],
    pub w3: [f32; W3_SIZE],
    pub food: f32,      // Replaces simple inventory
    pub water: f32,
    pub stamina: f32,
    pub health: f32,
    pub age: f32,
    pub id: u32,             // Unique ID to track pregnancies
    pub gestation_timer: f32,
    pub is_pregnant: f32,    // End of Person struct
}

impl Person {
    pub fn new(x: f32, y: f32, config: &crate::config::SimConfig) -> Self {
        let hidden_count = 16;

        let mut rng = rand::thread_rng();
        
        // Xavier/Glorot Initialization limits to prevent neuron saturation
        let w1_limit = (6.0 / (8.0 + hidden_count as f32)).sqrt(); // 8 inputs per hidden node
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
        for i in 0..(hidden_count * hidden_count) {
            w2[i] = (rng.r#gen::<f32>() * 2.0 * w2_limit) - w2_limit;
        }
        let mut w3 = [0.0; W3_SIZE];
        for i in 0..(hidden_count * NUM_OUTPUTS) {
            w3[i] = (rng.r#gen::<f32>() * 2.0 * w3_limit) - w3_limit;
        }

        Self {
            x,
            y,
            heading: rng.r#gen::<f32>() * PI * 2.0,
            speed: 1.4,
            hidden_count: hidden_count as u32,
            gender: if rng.r#gen::<f32>() > 0.5 { 1.0 } else { 0.0 }, // 50/50 chance
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
            w1_weights,
            w1_indices,
            w2,
            w3,
            food: 50000.0, 
            water: config.bio.max_water,
            stamina: 100.0,
            health: 100.0,
            age: rng.r#gen::<f32>() * config.bio.max_age * 0.5, // Start between 0 and 50% of max life expectancy
            id: rng.r#gen::<u32>(),
            gestation_timer: 0.0,
            is_pregnant: 0.0,
        }
    }

    pub fn reproduce_sexual(parent1: &mut Self, parent2: &mut Self, cost: f32) -> Self {
        let mut rng = rand::thread_rng();
        Self::reproduce_sexual_with_rng(parent1, parent2, cost, &mut rng)
    }

    pub fn reproduce_sexual_with_rng<R: Rng>(parent1: &mut Self, parent2: &mut Self, cost: f32, rng: &mut R) -> Self {
        parent1.wealth -= cost / 2.0; 
        parent2.wealth -= cost / 2.0; 
        
        let mut child = *parent1;
        child.age = 0.0;
        child.health = 100.0;
        child.food = 50000.0;
        child.wealth = cost / 2.0; // Inherit seed money
        child.water = parent1.water; // Child inherits water from parent, or could be config.bio.max_water
        child.stamina = 100.0;
        
        child.gender = if rng.r#gen::<f32>() > 0.5 { 1.0 } else { 0.0 };
        child.reproduce_desire = 0.0;
        child.attack_intent = 0.0;
        child.rest_intent = 0.0;
        child.comm1 = 0.0;
        child.comm2 = 0.0;
        child.comm3 = 0.0;
        child.comm4 = 0.0;
        child.mem1 = 0.0;
        child.mem2 = 0.0;
        child.mem3 = 0.0;
        child.mem4 = 0.0;
        child.mem5 = 0.0;
        child.mem6 = 0.0;
        child.mem7 = 0.0;
        child.mem8 = 0.0;
        child.buy_intent = 0.0;
        child.sell_intent = 0.0;
        child.ask_price = 1.0;
        child.bid_price = 1.0;
        child.drop_water_intent = 0.0;
        child.pickup_water_intent = 0.0;
        child.defend_intent = 0.0;
        child.build_road_intent = 0.0;
        child.build_house_intent = 0.0;
        child.build_farm_intent = 0.0;
        child.build_storage_intent = 0.0;
        child.destroy_infra_intent = 0.0;
        child._pad_agent1 = 0.0;
        child._pad_agent2 = 0.0;
        child._pad_agent3 = 0.0;
        child.id = rng.r#gen::<u32>();
        child.gestation_timer = 0.0;
        child.is_pregnant = 0.0;
        
        child.pheno_r = if rng.r#gen::<f32>() > 0.5 { parent1.pheno_r } else { parent2.pheno_r };
        child.pheno_g = if rng.r#gen::<f32>() > 0.5 { parent1.pheno_g } else { parent2.pheno_g };
        child.pheno_b = if rng.r#gen::<f32>() > 0.5 { parent1.pheno_b } else { parent2.pheno_b };
        if rng.r#gen::<f32>() < 0.1 { child.pheno_r = (child.pheno_r + (rng.r#gen::<f32>() * 0.4) - 0.2).clamp(-1.0, 1.0); }
        if rng.r#gen::<f32>() < 0.1 { child.pheno_g = (child.pheno_g + (rng.r#gen::<f32>() * 0.4) - 0.2).clamp(-1.0, 1.0); }
        if rng.r#gen::<f32>() < 0.1 { child.pheno_b = (child.pheno_b + (rng.r#gen::<f32>() * 0.4) - 0.2).clamp(-1.0, 1.0); }

        // 1. Crossover Genetics and Mutate
        for i in 0..child.w1_weights.len() {
            child.w1_weights[i] = if rng.r#gen::<f32>() > 0.5 { parent1.w1_weights[i] } else { parent2.w1_weights[i] };
            child.w1_indices[i] = if rng.r#gen::<f32>() > 0.5 { parent1.w1_indices[i] } else { parent2.w1_indices[i] };
            if rng.r#gen::<f32>() < 0.1 { child.w1_weights[i] = (child.w1_weights[i] + (rng.r#gen::<f32>() * 0.5) - 0.25).clamp(-2.0, 2.0); }
            if rng.r#gen::<f32>() < 0.02 { child.w1_indices[i] = rng.gen_range(0..NUM_INPUTS as u32); }
        }
        
        for i in 0..child.w2.len() {
            child.w2[i] = if rng.r#gen::<f32>() > 0.5 { parent1.w2[i] } else { parent2.w2[i] };
            if rng.r#gen::<f32>() < 0.1 { child.w2[i] = (child.w2[i] + (rng.r#gen::<f32>() * 0.5) - 0.25).clamp(-2.0, 2.0); }
        }

        for i in 0..child.w3.len() {
            child.w3[i] = if rng.r#gen::<f32>() > 0.5 { parent1.w3[i] } else { parent2.w3[i] };
            if rng.r#gen::<f32>() < 0.1 { child.w3[i] = (child.w3[i] + (rng.r#gen::<f32>() * 0.5) - 0.25).clamp(-2.0, 2.0); }
        }

        // 2. Structural mutation
        if rng.r#gen::<f32>() < 0.05 && child.hidden_count < NUM_HIDDEN_MAX as u32 {
            let h = child.hidden_count as usize;
            
            let w1_limit = (6.0 / (8.0 + h as f32 + 1.0)).sqrt();
            let w2_limit = (6.0 / ((h as f32 + 1.0) * 2.0)).sqrt();
            let w3_limit = (6.0 / (h as f32 + 1.0 + NUM_OUTPUTS as f32)).sqrt();

            let mut available: Vec<u32> = (0..NUM_INPUTS as u32).collect();
            for k in 0..8 {
                let idx = rng.gen_range(0..available.len());
                child.w1_indices[h * 8 + k] = available.remove(idx);
                child.w1_weights[h * 8 + k] = (rng.r#gen::<f32>() * 2.0 * w1_limit) - w1_limit;
            }

            for i in 0..NUM_HIDDEN_MAX { child.w2[h * NUM_HIDDEN_MAX + i] = (rng.r#gen::<f32>() * 2.0 * w2_limit) - w2_limit; } // H1 out
            for i in 0..NUM_HIDDEN_MAX { child.w2[i * NUM_HIDDEN_MAX + h] = (rng.r#gen::<f32>() * 2.0 * w2_limit) - w2_limit; } // H2 in
            for i in 0..NUM_OUTPUTS { child.w3[h * NUM_OUTPUTS + i] = (rng.r#gen::<f32>() * 2.0 * w3_limit) - w3_limit; } 
            child.hidden_count += 1;
        }
        
        child
    }

    pub fn clone_as_descendant(&self, x: f32, y: f32, mutation_rate: f32, mutation_strength: f32, config: &crate::config::SimConfig) -> Self {
        let mut child = *self;
        child.age = 0.0;
        child.health = 100.0;
        child.food = 50000.0; 
        child.water = config.bio.max_water;
        child.stamina = 100.0;
        child.wealth = 500.0;
        child.x = x;
        child.y = y;
        
        let mut rng = rand::thread_rng();
        child.gender = if rng.r#gen::<f32>() > 0.5 { 1.0 } else { 0.0 };
        child.reproduce_desire = 0.0;
        child.attack_intent = 0.0;
        child.rest_intent = 0.0;
        child.comm1 = 0.0;
        child.comm2 = 0.0;
        child.comm3 = 0.0;
        child.comm4 = 0.0;
        child.mem1 = 0.0;
        child.mem2 = 0.0;
        child.mem3 = 0.0;
        child.mem4 = 0.0;
        child.mem5 = 0.0;
        child.mem6 = 0.0;
        child.mem7 = 0.0;
        child.mem8 = 0.0;
        child.buy_intent = 0.0;
        child.sell_intent = 0.0;
        child.ask_price = 1.0;
        child.bid_price = 1.0;
        child.drop_water_intent = 0.0;
        child.pickup_water_intent = 0.0;
        child.defend_intent = 0.0;
        child.build_road_intent = 0.0;
        child.build_house_intent = 0.0;
        child.build_farm_intent = 0.0;
        child.build_storage_intent = 0.0;
        child.destroy_infra_intent = 0.0;
        child._pad_agent1 = 0.0;
        child._pad_agent2 = 0.0;
        child._pad_agent3 = 0.0;
        child.id = rng.r#gen::<u32>();
        child.gestation_timer = 0.0;
        child.is_pregnant = 0.0;
        child.heading = rng.r#gen::<f32>() * std::f32::consts::PI * 2.0;
        
        if rng.r#gen::<f32>() < mutation_rate { child.pheno_r = (child.pheno_r + (rng.r#gen::<f32>() * 2.0 * mutation_strength) - mutation_strength).clamp(-1.0, 1.0); }
        if rng.r#gen::<f32>() < mutation_rate { child.pheno_g = (child.pheno_g + (rng.r#gen::<f32>() * 2.0 * mutation_strength) - mutation_strength).clamp(-1.0, 1.0); }
        if rng.r#gen::<f32>() < mutation_rate { child.pheno_b = (child.pheno_b + (rng.r#gen::<f32>() * 2.0 * mutation_strength) - mutation_strength).clamp(-1.0, 1.0); }

        for w in child.w1_weights.iter_mut() {
            if rng.r#gen::<f32>() < mutation_rate { *w = (*w + (rng.r#gen::<f32>() * 2.0 * mutation_strength) - mutation_strength).clamp(-2.0, 2.0); }
        }
        for idx in child.w1_indices.iter_mut() {
            if rng.r#gen::<f32>() < mutation_rate * 0.1 { *idx = rng.gen_range(0..NUM_INPUTS as u32); }
        }
        for w in child.w2.iter_mut() {
            if rng.r#gen::<f32>() < mutation_rate { *w = (*w + (rng.r#gen::<f32>() * 2.0 * mutation_strength) - mutation_strength).clamp(-2.0, 2.0); }
        }
        for w in child.w3.iter_mut() {
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
            if (h as u32) < self.hidden_count {
                for k in 0..8 {
                    let conn_idx = h * 8 + k;
                    let input_idx = self.w1_indices[conn_idx] as usize;
                    let weight = self.w1_weights[conn_idx];

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
                w.push(self.w3[h * NUM_OUTPUTS + o]);
            }
            outputs.insert(OUTPUT_LABELS[o].to_string(), w);
        }

        AgentWeights {
            hidden_count: self.hidden_count,
            inputs,
            w1_weights: self.w1_weights.to_vec(),
            w1_indices: self.w1_indices.to_vec(),
            w2: self.w2.to_vec(),
            outputs,
            w1: Vec::new(),
            w3: Vec::new(),
        }
    }

    pub fn apply_weights(&mut self, weights: &AgentWeights) {
        self.hidden_count = weights.hidden_count.min(NUM_HIDDEN_MAX as u32);
        
        if !weights.w1_weights.is_empty() && !weights.w1_indices.is_empty() {
            let w1_len = self.w1_weights.len().min(weights.w1_weights.len());
            self.w1_weights[..w1_len].copy_from_slice(&weights.w1_weights[..w1_len]);
            let idx_len = self.w1_indices.len().min(weights.w1_indices.len());
            self.w1_indices[..idx_len].copy_from_slice(&weights.w1_indices[..idx_len]);
        }

        let w2_len = self.w2.len().min(weights.w2.len());
        self.w2[..w2_len].copy_from_slice(&weights.w2[..w2_len]);
        
        if !weights.outputs.is_empty() {
            for o in 0..NUM_OUTPUTS {
                if let Some(w) = weights.outputs.get(OUTPUT_LABELS[o]) {
                    for h in 0..NUM_HIDDEN_MAX {
                        if h < w.len() { self.w3[h * NUM_OUTPUTS + o] = w[h]; }
                    }
                }
            }
        } else if !weights.w3.is_empty() {
            let w3_len = self.w3.len().min(weights.w3.len());
            self.w3[..w3_len].copy_from_slice(&weights.w3[..w3_len]);
        }
    }

    /// Performs a CPU-side forward pass of the agent's neural network.
    /// Used for behavioral prediction and UI probing.
    pub fn calculate_input_output_influence(&self) -> Vec<f32> {
        let h_count = self.hidden_count as usize;
        let mut influence = vec![0.0f32; NUM_INPUTS * NUM_OUTPUTS];
        
        // 1. Calculate Input -> Hidden2 influence (W2 * sparse W1)
        // Hidden1 = activation(W1 * Input)
        // Hidden2 = activation(W2 * Hidden1)
        // Output  = activation(W3 * Hidden2)
        
        let mut w2_w1 = vec![0.0f32; NUM_INPUTS * h_count];
        for h2 in 0..h_count {
            for h1 in 0..h_count {
                let w2_val = self.w2[h2 * h_count + h1];
                for k in 0..8 {
                    let w1_val = self.w1_weights[h1 * 8 + k];
                    let input_idx = self.w1_indices[h1 * 8 + k] as usize;
                    w2_w1[h2 * NUM_INPUTS + input_idx] += w2_val * w1_val;
                }
            }
        }
        
        // 2. Calculate Input -> Output influence (W3 * w2_w1)
        for o in 0..NUM_OUTPUTS {
            for h2 in 0..h_count {
                let w3_val = self.w3[h2 * NUM_OUTPUTS + o];
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
        for h in 0..self.hidden_count as usize {
            let mut sum = 0.0;
            for k in 0..8 {
                let idx = self.w1_indices[h * 8 + k] as usize;
                if idx < NUM_INPUTS {
                    sum += inputs[idx] * self.w1_weights[h * 8 + k];
                }
            }
            hidden[h] = sum.max(0.0); // ReLU
        }

        // Layer 2: Hidden -> Hidden (Dense)
        let mut hidden2 = [0.0f32; NUM_HIDDEN_MAX];
        for h in 0..self.hidden_count as usize {
            let mut sum = 0.0;
            for prev_h in 0..self.hidden_count as usize {
                sum += hidden[prev_h] * self.w2[prev_h * NUM_HIDDEN_MAX + h];
            }
            hidden2[h] = sum.max(0.0); // ReLU
        }

        // Layer 3: Hidden -> Output (Dense)
        let mut outputs = [0.0f32; NUM_OUTPUTS];
        for o in 0..NUM_OUTPUTS {
            let mut sum = 0.0;
            for h in 0..self.hidden_count as usize {
                sum += hidden2[h] * self.w3[h * NUM_OUTPUTS + o];
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
        let p = Person::new(10.0, 20.0, &config);
        assert_eq!(p.x, 10.0);
        assert_eq!(p.y, 20.0);
        assert!(p.health > 0.0);
        assert!(p.food > 0.0);
        assert!(p.water > 0.0);
        assert!(p.wealth > 0.0);
    }

    #[test]
    fn test_reproduce_sexual() {
        let config = SimConfig::default();
        let mut p1 = Person::new(10.0, 20.0, &config);
        let mut p2 = Person::new(15.0, 25.0, &config);
        p1.wealth = 1000.0;
        p2.wealth = 1000.0;
        let cost = 500.0;
        
        let child = Person::reproduce_sexual(&mut p1, &mut p2, cost);
        
        assert_eq!(p1.wealth, 750.0);
        assert_eq!(p2.wealth, 750.0);
        assert_eq!(child.wealth, 250.0);
        assert_eq!(child.age, 0.0);
        assert_eq!(child.health, 100.0);
    }

    #[test]
    fn test_clone_as_descendant() {
        let config = SimConfig::default();
        let p = Person::new(10.0, 20.0, &config);
        let child = p.clone_as_descendant(30.0, 40.0, 0.1, 0.1, &config);
        
        assert_eq!(child.x, 30.0);
        assert_eq!(child.y, 40.0);
        assert_eq!(child.age, 0.0);
        assert_eq!(child.health, 100.0);
    }

    #[test]
    fn test_extract_apply_weights() {
        let config = SimConfig::default();
        let p = Person::new(10.0, 20.0, &config);
        let weights = p.extract_weights();
        
        let mut p2 = Person::new(0.0, 0.0, &config);
        p2.apply_weights(&weights);
        
        assert_eq!(p2.hidden_count, p.hidden_count);
        assert_eq!(p2.w2, p.w2);
        assert_eq!(p2.w3, p.w3);
        assert_eq!(p2.w1_weights, p.w1_weights);
        assert_eq!(p2.w1_indices, p.w1_indices);
    }

    #[test]
    fn test_mental_simulation_sanity() {
        let config = SimConfig::default();
        let p = Person::new(0.0, 0.0, &config);
        let mut inputs = [0.0f32; NUM_INPUTS];
        inputs[0] = 1.0; // Bias input
        
        let outputs = p.mental_simulation(&inputs);
        // Ensure outputs are within expected range [-1, 1]
        for &o in outputs.iter() {
            assert!(o >= -1.0 && o <= 1.0);
        }
    }
}
