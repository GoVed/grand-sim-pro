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
use serde::{Serialize, Deserialize, Serializer, Deserializer};

pub const NUM_INPUTS: usize = 188; // 160 + 8 (Comm) + 16 (Mem) + 4 (Identity)
pub const NUM_HIDDEN_MAX: usize = 128;
pub const NUM_OUTPUTS: usize = 56; // 32 + 8 (Comm) + 16 (Mem)
pub const W1_SIZE: usize = NUM_HIDDEN_MAX * 8; // Sparse Fixed-K Connectivity
pub const W2_SIZE: usize = NUM_HIDDEN_MAX * NUM_HIDDEN_MAX;
pub const W3_SIZE: usize = NUM_HIDDEN_MAX * NUM_OUTPUTS;

pub const INPUT_LABELS: [&str; NUM_INPUTS] = [
    "Bias", "Local Res", "Local Pop", "Avg Speed", "Avg Share", "Avg Repro", "Avg Aggr", "Avg Preg",
    "Avg Turn", "Avg Rest", 
    "Comm 1", "Comm 2", "Comm 3", "Comm 4", "Comm 5", "Comm 6", "Comm 7", "Comm 8", "Comm 9", "Comm 10", "Comm 11", "Comm 12",
    "Health", "Food", "Water", "Stamina", "Age", "Gender", "Temp", "Season", "Is Preg", "Encumbrance", "Crowding",
    "Mem 1", "Mem 2", "Mem 3", "Mem 4", "Mem 5", "Mem 6", "Mem 7", "Mem 8", "Mem 9", "Mem 10", "Mem 11", "Mem 12", "Mem 13", "Mem 14", "Mem 15", "Mem 16", "Mem 17", "Mem 18", "Mem 19", "Mem 20", "Mem 21", "Mem 22", "Mem 23", "Mem 24",
    "Wealth", "Avg Ask", "Avg Bid", "Daylight",
    "ID Feat 1", "ID Feat 2", "ID Feat 3", "ID Feat 4",
    "Loc Pheno R", "Loc Pheno G", "Loc Pheno B",
    "FL Res", "FL Elev", "FL Pop", "FL C1", "FL C2", "FL C3", "FL C4", "FL PR", "FL PG", "FL PB", "FL Road", "FL House", "FL Farm", "FL Store",
    "F Res", "F Elev", "F Pop", "F C1", "F C2", "F C3", "F C4", "F PR", "F PG", "F PB", "FL Road", "FL House", "FL Farm", "FL Store",
    "FR Res", "FR Elev", "FR Pop", "FR C1", "FR C2", "FR C3", "FR C4", "FR PR", "FR PG", "FR PB", "FL Road", "FL House", "FL Farm", "FL Store",
    "L Res", "L Elev", "L Pop", "L C1", "L C2", "L C3", "L C4", "L PR", "L PG", "L PB", "L Road", "L House", "L Farm", "L Store",
    "R Res", "R Elev", "R Pop", "R C1", "R C2", "R C3", "R C4", "R PR", "R PG", "R PB", "R Road", "R House", "R Farm", "R Store",
    "BL Res", "BL Elev", "BL Pop", "BL C1", "BL C2", "BL C3", "BL C4", "BL PR", "BL PG", "BL PB", "BL Road", "BL House", "BL Farm", "BL Store",
    "B Res", "B Elev", "B Pop", "B C1", "B C2", "B C3", "B C4", "B PR", "B PG", "B PB", "B Road", "B House", "B Farm", "B Store",
    "BR Res", "BR Elev", "BR Pop", "BR C1", "BR C2", "BR C3", "BR C4", "BR PR", "BR PG", "BR PB", "BR Road", "BR House", "BR Farm", "BR Store",
    "Loc Road", "Loc House", "Loc Farm", "Loc Storage",
    "Target F1", "Target F2", "Target F3", "Target F4"
];

pub const OUTPUT_LABELS: [&str; NUM_OUTPUTS] = [
    "Turn", "Speed", "Drop Res", "Reproduce", "Attack", "Rest", 
    "Comm 1", "Comm 2", "Comm 3", "Comm 4", "Comm 5", "Comm 6", "Comm 7", "Comm 8", "Comm 9", "Comm 10", "Comm 11", "Comm 12",
    "Learn", 
    "Mem 1", "Mem 2", "Mem 3", "Mem 4", "Mem 5", "Mem 6", "Mem 7", "Mem 8", "Mem 9", "Mem 10", "Mem 11", "Mem 12", "Mem 13", "Mem 14", "Mem 15", "Mem 16", "Mem 17", "Mem 18", "Mem 19", "Mem 20", "Mem 21", "Mem 22", "Mem 23", "Mem 24",
    "Buy Intent", "Sell Intent", "Ask Price", "Bid Price", "Drop H2O", "Pickup H2O", "Defend Intent", "Build Road", "Build House", "Build Farm", "Build Storage",
    "Destroy Infra", "Emergency"
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
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Serialize, Deserialize, PartialEq, Debug, Default)]
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
    pub comms: [f32; 12],
    pub mems: [f32; 24],
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
    pub emergency_intent: f32,
    pub id_f1: f32,  // Identity feature 1 (-1 to 1)
    pub id_f2: f32,  // Identity feature 2
    pub id_f3: f32,  // Identity feature 3
    pub id_f4: f32,  // Identity feature 4
    pub nearest_id_f1: f32, // Target's Identity feature 1
    pub nearest_id_f2: f32,
    pub nearest_id_f3: f32,
    pub nearest_id_f4: f32,
    pub food: f32,     
    pub water: f32,
    pub stamina: f32,
    pub health: f32,
    pub age: f32,
    pub id: u32,       
    pub gestation_timer: f32,
    pub is_pregnant: f32,
    pub _pad_identity: f32, // Maintains 16-byte alignment (Total 64 f32 slots = 256 bytes)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, PartialEq, Debug)]
pub struct Genetics {
    pub w1_weights: [f32; W1_SIZE],
    pub w1_indices: [u32; W1_SIZE],
    pub w2: [f32; W2_SIZE],
    pub w3: [f32; W3_SIZE],
}

#[derive(Serialize, Deserialize)]
struct GeneticsSerializable {
    w1_weights: Vec<f32>,
    w1_indices: Vec<u32>,
    w2: Vec<f32>,
    w3: Vec<f32>,
}

impl Serialize for Genetics {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        GeneticsSerializable {
            w1_weights: self.w1_weights.to_vec(),
            w1_indices: self.w1_indices.to_vec(),
            w2: self.w2.to_vec(),
            w3: self.w3.to_vec(),
        }.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Genetics {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de> {
        let s = GeneticsSerializable::deserialize(deserializer)?;
        let mut res = Genetics {
            w1_weights: [0.0; W1_SIZE],
            w1_indices: [0; W1_SIZE],
            w2: [0.0; W2_SIZE],
            w3: [0.0; W3_SIZE],
        };
        if s.w1_weights.len() == W1_SIZE { res.w1_weights.copy_from_slice(&s.w1_weights); }
        if s.w1_indices.len() == W1_SIZE { res.w1_indices.copy_from_slice(&s.w1_indices); }
        if s.w2.len() == W2_SIZE { res.w2.copy_from_slice(&s.w2); }
        if s.w3.len() == W3_SIZE { res.w3.copy_from_slice(&s.w3); }
        Ok(res)
    }
}

#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Debug)]
pub struct Person {
    pub state: AgentState,
    pub genetics: Genetics,
}

impl Person {
    pub fn new(x: f32, y: f32, genetics_index: u32, config: &crate::config::SimConfig) -> Self {
        let hidden_count = 32; // Increased starting complexity

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
                // Bias towards small non-zero values to kickstart evolution
                w1_weights[h * 8 + k] = (rng.r#gen::<f32>() * 2.0 * w1_limit) - w1_limit + 0.01;
            }
        }

        let mut w2 = [0.0; W2_SIZE];
        for i in 0..hidden_count {
            for j in 0..hidden_count {
                w2[i * NUM_HIDDEN_MAX + j] = (rng.r#gen::<f32>() * 2.0 * w2_limit) - w2_limit + 0.01;
            }
        }
        let mut w3 = [0.0; W3_SIZE];
        for i in 0..hidden_count {
            for j in 0..NUM_OUTPUTS {
                w3[i * NUM_OUTPUTS + j] = (rng.r#gen::<f32>() * 2.0 * w3_limit) - w3_limit + 0.01;
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
                comms: [0.0; 12],
                mems: [0.0; 24],
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
                emergency_intent: 0.0,
                id_f1: 0.0, id_f2: 0.0, id_f3: 0.0, id_f4: 0.0,
                nearest_id_f1: 0.0, nearest_id_f2: 0.0, nearest_id_f3: 0.0, nearest_id_f4: 0.0,
                food: 50000.0, 
                water: config.bio.max_water,
                stamina: 100.0,
                health: 100.0,
                age: rng.r#gen::<f32>() * config.bio.max_age * 0.5,
                id: rng.r#gen::<u32>(),
                gestation_timer: 0.0,
                is_pregnant: 0.0,
                _pad_identity: 0.0,
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
        child.state.water = parent1.state.water; 
        child.state.stamina = 100.0;

        child.state.gender = if rng.r#gen::<f32>() > 0.5 { 1.0 } else { 0.0 };
        child.state.genetics_index = genetics_index;
        child.state.reproduce_desire = 0.0;
        child.state.attack_intent = 0.0;
        child.state.rest_intent = 0.0;
        child.state.comms = [0.0; 12];
        child.state.mems = [0.0; 24];
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
        child.state.emergency_intent = 0.0;
        child.state.id_f1 = 0.0; child.state.id_f2 = 0.0; child.state.id_f3 = 0.0; child.state.id_f4 = 0.0;
        child.state.nearest_id_f1 = -1.0; child.state.nearest_id_f2 = 0.0; child.state.nearest_id_f3 = 0.0; child.state.nearest_id_f4 = 0.0;
        child.state.id = rng.r#gen::<u32>();
        child.state.gestation_timer = 0.0;
        child.state.is_pregnant = 0.0;
        child.state._pad_identity = 0.0;
        child.state.heading = rng.r#gen::<f32>() * std::f32::consts::PI * 2.0;
        
        if rng.r#gen::<f32>() < 0.1 { child.state.pheno_r = (child.state.pheno_r + (rng.r#gen::<f32>() * 0.2) - 0.1).clamp(-1.0, 1.0); }
        if rng.r#gen::<f32>() < 0.1 { child.state.pheno_g = (child.state.pheno_g + (rng.r#gen::<f32>() * 0.2) - 0.1).clamp(-1.0, 1.0); }
        if rng.r#gen::<f32>() < 0.1 { child.state.pheno_b = (child.state.pheno_b + (rng.r#gen::<f32>() * 0.2) - 0.1).clamp(-1.0, 1.0); }

        let mutation_rate = 0.05;
        let mutation_strength = 0.05;

        // Block Crossover (Optimized)
        let cp_w1 = rng.gen_range(0..W1_SIZE);
        child.genetics.w1_weights[cp_w1..].copy_from_slice(&parent2.genetics.w1_weights[cp_w1..]);
        child.genetics.w1_indices[cp_w1..].copy_from_slice(&parent2.genetics.w1_indices[cp_w1..]);

        let cp_w2 = rng.gen_range(0..W2_SIZE);
        child.genetics.w2[cp_w2..].copy_from_slice(&parent2.genetics.w2[cp_w2..]);

        let cp_w3 = rng.gen_range(0..W3_SIZE);
        child.genetics.w3[cp_w3..].copy_from_slice(&parent2.genetics.w3[cp_w3..]);

        // Sparse Mutation Sampling (Optimized)
        let num_mut_w1 = (W1_SIZE as f32 * mutation_rate) as usize;
        for _ in 0..num_mut_w1 {
            let i = rng.gen_range(0..W1_SIZE);
            child.genetics.w1_weights[i] = (child.genetics.w1_weights[i] + (rng.r#gen::<f32>() * 2.0 * mutation_strength) - mutation_strength).clamp(-2.0, 2.0);
        }
        let num_mut_w1_idx = (W1_SIZE as f32 * mutation_rate * 0.1) as usize;
        for _ in 0..num_mut_w1_idx {
            let i = rng.gen_range(0..W1_SIZE);
            child.genetics.w1_indices[i] = rng.gen_range(0..NUM_INPUTS as u32);
        }
        
        let num_mut_w2 = (W2_SIZE as f32 * mutation_rate) as usize;
        for _ in 0..num_mut_w2 {
            let i = rng.gen_range(0..W2_SIZE);
            child.genetics.w2[i] = (child.genetics.w2[i] + (rng.r#gen::<f32>() * 2.0 * mutation_strength) - mutation_strength).clamp(-2.0, 2.0);
        }
        
        let num_mut_w3 = (W3_SIZE as f32 * mutation_rate) as usize;
        for _ in 0..num_mut_w3 {
            let i = rng.gen_range(0..W3_SIZE);
            child.genetics.w3[i] = (child.genetics.w3[i] + (rng.r#gen::<f32>() * 2.0 * mutation_strength) - mutation_strength).clamp(-2.0, 2.0);
        }

        child
    }

    pub fn clone_as_descendant(&self, x: f32, y: f32, genetics_index: u32, mutation_rate: f32, mutation_strength: f32, _config: &crate::config::SimConfig) -> Self {
        let mut child = *self;
        child.state.age = 0.0;
        child.state.health = 100.0;
        child.state.food = 50000.0; 
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
        child.state.comms = [0.0; 12];
        child.state.mems = [0.0; 24];
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
        child.state.emergency_intent = 0.0;
        child.state.id_f1 = 0.0; child.state.id_f2 = 0.0; child.state.id_f3 = 0.0; child.state.id_f4 = 0.0;
        child.state.nearest_id_f1 = -1.0; child.state.nearest_id_f2 = 0.0; child.state.nearest_id_f3 = 0.0; child.state.nearest_id_f4 = 0.0;
        child.state.id = rng.r#gen::<u32>();
        child.state.gestation_timer = 0.0;
        child.state.is_pregnant = 0.0;
        child.state._pad_identity = 0.0;
        child.state.heading = rng.r#gen::<f32>() * PI * 2.0;
        
        if rng.r#gen::<f32>() < mutation_rate { child.state.pheno_r = (child.state.pheno_r + (rng.r#gen::<f32>() * 2.0 * mutation_strength) - mutation_strength).clamp(-1.0, 1.0); }
        if rng.r#gen::<f32>() < mutation_rate { child.state.pheno_g = (child.state.pheno_g + (rng.r#gen::<f32>() * 2.0 * mutation_strength) - mutation_strength).clamp(-1.0, 1.0); }
        if rng.r#gen::<f32>() < mutation_rate { child.state.pheno_b = (child.state.pheno_b + (rng.r#gen::<f32>() * 2.0 * mutation_strength) - mutation_strength).clamp(-1.0, 1.0); }

        // Sparse Mutation Sampling (Optimized)
        let num_mut_w1 = (W1_SIZE as f32 * mutation_rate) as usize;
        for _ in 0..num_mut_w1 {
            let i = rng.gen_range(0..W1_SIZE);
            child.genetics.w1_weights[i] = (child.genetics.w1_weights[i] + (rng.r#gen::<f32>() * 2.0 * mutation_strength) - mutation_strength).clamp(-2.0, 2.0);
        }
        let num_mut_w1_idx = (W1_SIZE as f32 * mutation_rate * 0.1) as usize;
        for _ in 0..num_mut_w1_idx {
            let i = rng.gen_range(0..W1_SIZE);
            child.genetics.w1_indices[i] = rng.gen_range(0..NUM_INPUTS as u32);
        }
        
        let num_mut_w2 = (W2_SIZE as f32 * mutation_rate) as usize;
        for _ in 0..num_mut_w2 {
            let i = rng.gen_range(0..W2_SIZE);
            child.genetics.w2[i] = (child.genetics.w2[i] + (rng.r#gen::<f32>() * 2.0 * mutation_strength) - mutation_strength).clamp(-2.0, 2.0);
        }
        
        let num_mut_w3 = (W3_SIZE as f32 * mutation_rate) as usize;
        for _ in 0..num_mut_w3 {
            let i = rng.gen_range(0..W3_SIZE);
            child.genetics.w3[i] = (child.genetics.w3[i] + (rng.r#gen::<f32>() * 2.0 * mutation_strength) - mutation_strength).clamp(-2.0, 2.0);
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
        for i in 0..NUM_OUTPUTS {
            outputs.insert(OUTPUT_LABELS[i].to_string(), vec![0.0; NUM_HIDDEN_MAX]);
        }

        for h in 0..NUM_HIDDEN_MAX {
            if (h as u32) < self.state.hidden_count {
                for o in 0..NUM_OUTPUTS {
                    let weight = self.genetics.w3[h * NUM_OUTPUTS + o];
                    if let Some(weights_vec) = outputs.get_mut(OUTPUT_LABELS[o]) {
                        weights_vec[h] = weight;
                    }
                }
            }
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
        self.state.hidden_count = weights.hidden_count;
        if weights.w1_weights.len() == W1_SIZE { self.genetics.w1_weights.copy_from_slice(&weights.w1_weights); }
        if weights.w1_indices.len() == W1_SIZE { self.genetics.w1_indices.copy_from_slice(&weights.w1_indices); }
        if weights.w2.len() == W2_SIZE { self.genetics.w2.copy_from_slice(&weights.w2); }
        
        // Handle W3 with potential output label mapping for robustness
        if weights.w3.len() == W3_SIZE {
            self.genetics.w3.copy_from_slice(&weights.w3);
        } else if !weights.outputs.is_empty() {
            for (label, h_weights) in &weights.outputs {
                if let Some(o_idx) = OUTPUT_LABELS.iter().position(|&l| l == label) {
                    for (h, &w) in h_weights.iter().enumerate() {
                        if h < NUM_HIDDEN_MAX {
                            self.genetics.w3[h * NUM_OUTPUTS + o_idx] = w;
                        }
                    }
                }
            }
        }
    }

    pub fn calculate_input_output_influence(&self) -> Vec<f32> {
        let mut influence = vec![0.0f32; NUM_INPUTS * NUM_OUTPUTS];
        let h_count = self.state.hidden_count as usize;

        // 1. Sparse Input -> Hidden1 influence
        let mut h1_inf = vec![0.0f32; NUM_INPUTS * NUM_HIDDEN_MAX];
        for h in 0..h_count {
            for k in 0..8 {
                let in_idx = self.genetics.w1_indices[h * 8 + k] as usize;
                let weight = self.genetics.w1_weights[h * 8 + k];
                h1_inf[in_idx * NUM_HIDDEN_MAX + h] = weight;
            }
        }

        // 2. Multi-step propagation through layers (Approximated as matrix product W1 * W2 * W3)
        let mut h2_inf = vec![0.0f32; NUM_INPUTS * NUM_HIDDEN_MAX];
        for i in 0..NUM_INPUTS {
            for h2 in 0..h_count {
                let mut sum = 0.0;
                for h1 in 0..h_count {
                    sum += h1_inf[i * NUM_HIDDEN_MAX + h1] * self.genetics.w2[h1 * NUM_HIDDEN_MAX + h2];
                }
                h2_inf[i * NUM_HIDDEN_MAX + h2] = sum / h_count.max(1) as f32;
            }
        }

        for i in 0..NUM_INPUTS {
            for o in 0..NUM_OUTPUTS {
                let mut sum = 0.0;
                for h2 in 0..h_count {
                    let w3_val = self.genetics.w3[h2 * NUM_OUTPUTS + o];
                    let prev_inf = h2_inf[i * NUM_HIDDEN_MAX + h2];
                    sum += w3_val * prev_inf;
                }
                influence[i * NUM_OUTPUTS + o] = sum / h_count.max(1) as f32;
            }
        }

        influence
    }

    pub fn mental_simulation(&self, inputs: &[f32; NUM_INPUTS]) -> [f32; NUM_OUTPUTS] {
        let mut hidden1 = [0.0f32; NUM_HIDDEN_MAX];
        let h_count = self.state.hidden_count as usize;
        
        for h in 0..h_count {
            let mut sum = 0.0;
            for k in 0..8 {
                let in_idx = self.genetics.w1_indices[h * 8 + k] as usize;
                sum += inputs[in_idx.min(NUM_INPUTS-1)] * self.genetics.w1_weights[h * 8 + k];
            }
            hidden1[h] = sum.tanh();
        }

        let mut hidden2 = [0.0f32; NUM_HIDDEN_MAX];
        for h2 in 0..h_count {
            let mut sum = 0.0;
            for h1 in 0..h_count {
                sum += hidden1[h1] * self.genetics.w2[h1 * NUM_HIDDEN_MAX + h2];
            }
            hidden2[h2] = sum.tanh();
        }

        let mut outputs = [0.0f32; NUM_OUTPUTS];
        for o in 0..NUM_OUTPUTS {
            let mut sum = 0.0;
            for h in 0..h_count {
                sum += hidden2[h] * self.genetics.w3[h * NUM_OUTPUTS + o];
            }
            outputs[o] = sum.tanh();
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
        assert!(p.state.hidden_count >= 16);
    }

    #[test]
    fn test_reproduce_sexual() {
        let config = SimConfig::default();
        let mut p1 = Person::new(0.0, 0.0, 0, &config);
        let mut p2 = Person::new(1.0, 1.0, 1, &config);
        let child = Person::reproduce_sexual(&mut p1, &mut p2, 2, 100.0);
        assert_eq!(child.state.genetics_index, 2);
    }

    #[test]
    fn test_clone_as_descendant() {
        let config = SimConfig::default();
        let p = Person::new(0.0, 0.0, 0, &config);
        let child = p.clone_as_descendant(30.0, 40.0, 1, 0.1, 0.1, &config);
        assert_eq!(child.state.x, 30.0);
        assert_eq!(child.state.y, 40.0);
    }

    #[test]
    fn test_extract_apply_weights() {
        let config = SimConfig::default();
        let p1 = Person::new(0.0, 0.0, 0, &config);
        let weights = p1.extract_weights();
        let mut p2 = Person::new(1.0, 1.0, 1, &config);
        p2.apply_weights(&weights);
        assert_eq!(p1.state.hidden_count, p2.state.hidden_count);
    }

    #[test]
    fn test_mental_simulation_sanity() {
        let config = SimConfig::default();
        let p = Person::new(0.0, 0.0, 0, &config);
        let inputs = [0.5f32; NUM_INPUTS];
        let outputs = p.mental_simulation(&inputs);
        assert_eq!(outputs.len(), NUM_OUTPUTS);
    }

    #[test]
    fn test_calculate_input_output_influence() {
        let config = SimConfig::default();
        let p = Person::new(0.0, 0.0, 0, &config);
        let influence = p.calculate_input_output_influence();
        assert_eq!(influence.len(), NUM_INPUTS * NUM_OUTPUTS);
    }

    #[test]
    fn test_apply_weights_with_outputs_fallback() {
        let config = SimConfig::default();
        let p1 = Person::new(0.0, 0.0, 0, &config);
        let mut weights = p1.extract_weights();
        weights.w3.clear(); // Force fallback to outputs mapping
        
        let mut p2 = Person::new(1.0, 1.0, 1, &config);
        p2.apply_weights(&weights);
        // We just ensure it doesn't crash and applied some weights
        assert_eq!(p1.state.hidden_count, p2.state.hidden_count);
    }
}
