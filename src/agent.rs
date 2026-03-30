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

pub const NUM_INPUTS: usize = 48;
pub const NUM_HIDDEN_MAX: usize = 32;
pub const NUM_OUTPUTS: usize = 26;
pub const W1_SIZE: usize = NUM_INPUTS * NUM_HIDDEN_MAX;
pub const W2_SIZE: usize = NUM_HIDDEN_MAX * NUM_HIDDEN_MAX;
pub const W3_SIZE: usize = NUM_HIDDEN_MAX * NUM_OUTPUTS;

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
    pub _pad_intent1: f32, // Maintains 16-byte WGSL vector alignment before arrays
    pub _pad_intent2: f32,
    pub _pad_intent3: f32,
    pub w1: [f32; W1_SIZE],
    pub w2: [f32; W2_SIZE],
    pub w3: [f32; W3_SIZE],
    pub food: f32,      // Replaces simple inventory
    pub water: f32,
    pub stamina: f32,
    pub health: f32,
    pub age: f32,
    pub id: u32,             // Unique ID to track pregnancies
    pub gestation_timer: f32,
    pub is_pregnant: f32,    // Perfect 8784 byte alignment 
}

impl Person {
    pub fn new(x: f32, y: f32, config: &crate::config::SimConfig) -> Self {
        let hidden_count = 16;

        let mut rng = rand::thread_rng();
        
        // Xavier/Glorot Initialization limits to prevent neuron saturation
        let w1_limit = (6.0 / (NUM_INPUTS as f32 + hidden_count as f32)).sqrt();
        let w2_limit = (6.0 / (hidden_count as f32 + hidden_count as f32)).sqrt();
        let w3_limit = (6.0 / (hidden_count as f32 + NUM_OUTPUTS as f32)).sqrt();

        let mut w1 = [0.0; W1_SIZE];
        for i in 0..(NUM_INPUTS * hidden_count) {
            w1[i] = (rng.r#gen::<f32>() * 2.0 * w1_limit) - w1_limit;
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
            _pad_intent1: 0.0,
            _pad_intent2: 0.0,
            _pad_intent3: 0.0,
            w1,
            w2,
            w3,
            food: 50000.0, 
            water: config.max_water,
            stamina: 100.0,
            health: 100.0,
            age: rng.r#gen::<f32>() * config.max_age * 0.5, // Start between 0 and 50% of max life expectancy
            id: rng.r#gen::<u32>(),
            gestation_timer: 0.0,
            is_pregnant: 0.0,
        }
    }

    pub fn reproduce_sexual(parent1: &mut Self, parent2: &mut Self, cost: f32) -> Self {
        parent1.wealth -= cost / 2.0; 
        parent2.wealth -= cost / 2.0; 
        
        let mut child = *parent1;
        child.age = 0.0;
        child.health = 100.0;
        child.food = 50000.0;
        child.wealth = cost / 2.0; // Inherit seed money
        child.water = parent1.water; // Child inherits water from parent, or could be config.max_water
        child.stamina = 100.0;
        
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
        child._pad_intent1 = 0.0;
        child._pad_intent2 = 0.0;
        child._pad_intent3 = 0.0;
        child.id = rng.r#gen::<u32>();
        child.gestation_timer = 0.0;
        child.is_pregnant = 0.0;
        
        // 1. Crossover Genetics and Mutate
        for i in 0..child.w1.len() {
            child.w1[i] = if rng.r#gen::<f32>() > 0.5 { parent1.w1[i] } else { parent2.w1[i] };
            if rng.r#gen::<f32>() < 0.1 { child.w1[i] = (child.w1[i] + (rng.r#gen::<f32>() * 0.5) - 0.25).clamp(-2.0, 2.0); }
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
            
            let w1_limit = (6.0 / (NUM_INPUTS as f32 + h as f32 + 1.0)).sqrt();
            let w2_limit = (6.0 / ((h as f32 + 1.0) * 2.0)).sqrt();
            let w3_limit = (6.0 / (h as f32 + 1.0 + NUM_OUTPUTS as f32)).sqrt();

            for i in 0..NUM_INPUTS { child.w1[h * NUM_INPUTS + i] = (rng.r#gen::<f32>() * 2.0 * w1_limit) - w1_limit; }
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
        child.water = config.max_water;
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
        child._pad_intent1 = 0.0;
        child._pad_intent2 = 0.0;
        child._pad_intent3 = 0.0;
        child.id = rng.r#gen::<u32>();
        child.gestation_timer = 0.0;
        child.is_pregnant = 0.0;
        child.heading = rng.r#gen::<f32>() * std::f32::consts::PI * 2.0;
        
        for w in child.w1.iter_mut() {
            if rng.r#gen::<f32>() < mutation_rate { *w = (*w + (rng.r#gen::<f32>() * 2.0 * mutation_strength) - mutation_strength).clamp(-2.0, 2.0); }
        }
        for w in child.w2.iter_mut() {
            if rng.r#gen::<f32>() < mutation_rate { *w = (*w + (rng.r#gen::<f32>() * 2.0 * mutation_strength) - mutation_strength).clamp(-2.0, 2.0); }
        }
        for w in child.w3.iter_mut() {
            if rng.r#gen::<f32>() < mutation_rate { *w = (*w + (rng.r#gen::<f32>() * 2.0 * mutation_strength) - mutation_strength).clamp(-2.0, 2.0); }
        }
        child
    }
}