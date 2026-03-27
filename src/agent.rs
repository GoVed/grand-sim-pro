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
    pub buy_intent: f32,
    pub sell_intent: f32,
    pub ask_price: f32,
    pub bid_price: f32,
    pub drop_water_intent: f32, // New: Intent to drop water into cell
    pub pickup_water_intent: f32, // New: Intent to pick up water from cell
    pub wealth: f32,
    pub pad1: [f32; 2],  // Perfect 96-byte structural header
    pub w1: [f32; 1280], // 40 inputs * 32 hidden1
    pub w2: [f32; 1024], // 32 hidden1 * 32 hidden2
    pub w3: [f32; 704],  // 32 hidden2 * 22 outputs
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
        let hidden_count = 16; // Give the agents a bigger brain so Rayon has real work to do
        let inputs: usize = 40;
        let outputs: usize = 22;

        let mut rng = rand::thread_rng();
        let mut w1 = [0.0; 1280];
        for i in 0..(inputs * hidden_count) {
            w1[i] = (rng.r#gen::<f32>() * 2.0) - 1.0;
        }

        let mut w2 = [0.0; 1024];
        for i in 0..(hidden_count * hidden_count) {
            w2[i] = (rng.r#gen::<f32>() * 2.0) - 1.0;
        }
        let mut w3 = [0.0; 704];
        for i in 0..(hidden_count * outputs) {
            w3[i] = (rng.r#gen::<f32>() * 2.0) - 1.0;
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
            buy_intent: 0.0,
            sell_intent: 0.0,
            ask_price: 1.0,
            bid_price: 1.0,
            drop_water_intent: 0.0,
            pickup_water_intent: 0.0,
            wealth: 500.0,
            pad1: [0.0; 2],
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
        child.buy_intent = 0.0;
        child.sell_intent = 0.0;
        child.ask_price = 1.0;
        child.bid_price = 1.0;
        child.drop_water_intent = 0.0;
        child.pickup_water_intent = 0.0;
        child.id = rng.r#gen::<u32>();
        child.gestation_timer = 0.0;
        child.is_pregnant = 0.0;
        
        // 1. Crossover Genetics and Mutate
        for i in 0..child.w1.len() {
            child.w1[i] = if rng.r#gen::<f32>() > 0.5 { parent1.w1[i] } else { parent2.w1[i] };
            if rng.r#gen::<f32>() < 0.1 { child.w1[i] += (rng.r#gen::<f32>() * 0.5) - 0.25; }
        }
        
        for i in 0..child.w2.len() {
            child.w2[i] = if rng.r#gen::<f32>() > 0.5 { parent1.w2[i] } else { parent2.w2[i] };
            if rng.r#gen::<f32>() < 0.1 { child.w2[i] += (rng.r#gen::<f32>() * 0.5) - 0.25; }
        }

        for i in 0..child.w3.len() {
            child.w3[i] = if rng.r#gen::<f32>() > 0.5 { parent1.w3[i] } else { parent2.w3[i] };
            if rng.r#gen::<f32>() < 0.1 { child.w3[i] += (rng.r#gen::<f32>() * 0.5) - 0.25; }
        }

        // 2. Structural mutation
        if rng.r#gen::<f32>() < 0.05 && child.hidden_count < 32 {
            let h = child.hidden_count as usize;
            for i in 0..40 { child.w1[h * 40 + i] = (rng.r#gen::<f32>() * 2.0) - 1.0; }
            for i in 0..32 { child.w2[h * 32 + i] = (rng.r#gen::<f32>() * 2.0) - 1.0; } // H1 out
            for i in 0..32 { child.w2[i * 32 + h] = (rng.r#gen::<f32>() * 2.0) - 1.0; } // H2 in
            for i in 0..22 { child.w3[h * 22 + i] = (rng.r#gen::<f32>() * 2.0) - 1.0; } // Adjusted for 2 new outputs
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
        child.buy_intent = 0.0;
        child.sell_intent = 0.0;
        child.ask_price = 1.0;
        child.bid_price = 1.0;
        child.drop_water_intent = 0.0;
        child.pickup_water_intent = 0.0;
        child.id = rng.r#gen::<u32>();
        child.gestation_timer = 0.0;
        child.is_pregnant = 0.0;
        child.heading = rng.r#gen::<f32>() * std::f32::consts::PI * 2.0;
        
        for w in child.w1.iter_mut() {
            if rng.r#gen::<f32>() < mutation_rate { *w += (rng.r#gen::<f32>() * 2.0 * mutation_strength) - mutation_strength; }
        }
        for w in child.w2.iter_mut() {
            if rng.r#gen::<f32>() < mutation_rate { *w += (rng.r#gen::<f32>() * 2.0 * mutation_strength) - mutation_strength; }
        }
        for w in child.w3.iter_mut() {
            if rng.r#gen::<f32>() < mutation_rate { *w += (rng.r#gen::<f32>() * 2.0 * mutation_strength) - mutation_strength; }
        }
        child
    }
}