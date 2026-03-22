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
    pub pad: f32,
    pub w1: [f32; 384], // Fixed size (12 inputs * 32 hidden nodes max)
    pub w2: [f32; 128], // Fixed size (32 hidden nodes max * 4 outputs)
    pub inventory: f32, // Accumulated resources
    pub health: f32,
    pub age: f32,
    pub pad2: f32, // Struct perfectly padded to 816 bytes
}

impl Person {
    pub fn new(x: f32, y: f32) -> Self {
        let hidden_count = 16; // Give the agents a bigger brain so Rayon has real work to do
        let inputs = 12;      // Expanded: +Health, Inventory, Age, Gender
        let outputs = 4;      // Turn, Speed, Share, Reproduce

        let mut rng = rand::thread_rng();
        let mut w1 = [0.0; 384];
        for i in 0..(inputs * hidden_count) as usize {
            w1[i] = (rng.r#gen::<f32>() * 2.0) - 1.0;
        }

        let mut w2 = [0.0; 128];
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
            pad: 0.0,
            w1,
            w2,
            inventory: 100.0, // Start life with $100 baseline
            health: 100.0,
            age: 0.0,
            pad2: 0.0,
        }
    }

    pub fn reproduce_sexual(parent1: &mut Self, parent2: &mut Self, cost: f32) -> Self {
        parent1.inventory -= cost / 2.0; // Both parents split the economic burden
        parent2.inventory -= cost / 2.0; 
        
        let mut child = *parent1;
        child.age = 0.0;
        child.health = 100.0;
        child.inventory = cost / 2.0; // Child starts with half the cost as inheritance
        
        let mut rng = rand::thread_rng();
        child.gender = if rng.r#gen::<f32>() > 0.5 { 1.0 } else { 0.0 };
        child.reproduce_desire = 0.0;
        
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
            for i in 0..12 { child.w1[h * 12 + i] = (rng.r#gen::<f32>() * 2.0) - 1.0; }
            for i in 0..4 { child.w2[h * 4 + i] = (rng.r#gen::<f32>() * 2.0) - 1.0; }
            child.hidden_count += 1;
        }
        
        child
    }

    pub fn clone_as_descendant(&self) -> Self {
        let mut child = *self;
        child.age = 0.0;
        child.health = 100.0;
        child.inventory = 100.0; // Start fresh
        child.x = 400.0;
        child.y = 300.0;
        
        let mut rng = rand::thread_rng();
        child.gender = if rng.r#gen::<f32>() > 0.5 { 1.0 } else { 0.0 };
        child.reproduce_desire = 0.0;
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