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
    pub signal_emit: f32,
    pub attack_intent: f32,
    pub rest_intent: f32,
    pub pad1: [f32; 2],
    pub w1: [f32; 768], // Fixed size (24 inputs * 32 hidden nodes max)
    pub w2: [f32; 256], // Fixed size (32 hidden nodes max * 8 outputs)
    pub food: f32,      // Replaces simple inventory
    pub water: f32,
    pub stamina: f32,
    pub health: f32,
    pub age: f32,
    pub id: u32,             // Unique ID to track pregnancies
    pub gestation_timer: f32,
    pub is_pregnant: f32,    // Replaces padding, maintaining exactly 4176 bytes
}

impl Person {
    pub fn new(x: f32, y: f32, config: &crate::config::SimConfig) -> Self {
        let hidden_count = 16; // Give the agents a bigger brain so Rayon has real work to do
        let inputs = 24;      
        let outputs = 8;      

        let mut rng = rand::thread_rng();
        let mut w1 = [0.0; 768];
        for i in 0..(inputs * hidden_count) as usize {
            w1[i] = (rng.r#gen::<f32>() * 2.0) - 1.0;
        }

        let mut w2 = [0.0; 256];
        for i in 0..(hidden_count * outputs) as usize {
            w2[i] = (rng.r#gen::<f32>() * 2.0) - 1.0;
        }

        Self {
            x,
            y,
            heading: rng.r#gen::<f32>() * PI * 2.0,
            speed: 1.4,
            hidden_count,
            gender: if rng.r#gen::<f32>() > 0.5 { 1.0 } else { 0.0 }, // 50/50 chance
            reproduce_desire: 0.0,
            signal_emit: 0.0,
            attack_intent: 0.0,
            rest_intent: 0.0,
            pad1: [0.0; 2],
            w1,
            w2,
            food: 100.0, 
            water: 100.0,
            stamina: 100.0,
            health: 100.0,
            age: rng.r#gen::<f32>() * config.max_age * 0.5, // Start between 0 and 50% of max life expectancy
            id: rng.r#gen::<u32>(),
            gestation_timer: 0.0,
            is_pregnant: 0.0,
        }
    }

    pub fn reproduce_sexual(parent1: &mut Self, parent2: &mut Self, cost: f32) -> Self {
        parent1.food -= cost / 2.0; 
        parent2.food -= cost / 2.0; 
        
        let mut child = *parent1;
        child.age = 0.0;
        child.health = 100.0;
        child.food = cost / 2.0; 
        child.water = 100.0;
        child.stamina = 100.0;
        
        let mut rng = rand::thread_rng();
        child.gender = if rng.r#gen::<f32>() > 0.5 { 1.0 } else { 0.0 };
        child.reproduce_desire = 0.0;
        child.signal_emit = 0.0;
        child.attack_intent = 0.0;
        child.rest_intent = 0.0;
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

        // 2. Structural mutation
        if rng.r#gen::<f32>() < 0.05 && child.hidden_count < 32 {
            let h = child.hidden_count as usize;
            for i in 0..24 { child.w1[h * 24 + i] = (rng.r#gen::<f32>() * 2.0) - 1.0; }
            for i in 0..8 { child.w2[h * 8 + i] = (rng.r#gen::<f32>() * 2.0) - 1.0; }
            child.hidden_count += 1;
        }
        
        child
    }

    pub fn clone_as_descendant(&self, map_w: f32, map_h: f32) -> Self {
        let mut child = *self;
        child.age = 0.0;
        child.health = 100.0;
        child.food = 100.0; 
        child.water = 100.0;
        child.stamina = 100.0;
        child.x = map_w / 2.0;
        child.y = map_h / 2.0;
        
        let mut rng = rand::thread_rng();
        child.gender = if rng.r#gen::<f32>() > 0.5 { 1.0 } else { 0.0 };
        child.reproduce_desire = 0.0;
        child.signal_emit = 0.0;
        child.attack_intent = 0.0;
        child.rest_intent = 0.0;
        child.id = rng.r#gen::<u32>();
        child.gestation_timer = 0.0;
        child.is_pregnant = 0.0;
        child.heading = rng.r#gen::<f32>() * std::f32::consts::PI * 2.0;
        
        for w in child.w1.iter_mut() {
            if rng.r#gen::<f32>() < 0.1 { *w += (rng.r#gen::<f32>() * 0.5) - 0.25; }
        }
        for w in child.w2.iter_mut() {
            if rng.r#gen::<f32>() < 0.1 { *w += (rng.r#gen::<f32>() * 0.5) - 0.25; }
        }
        child
    }
}