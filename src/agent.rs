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
    pub w1: [f32; 64], // Fixed size for GPU Buffer (4 inputs * 16 hidden nodes)
    pub w2: [f32; 48], // Fixed size for GPU Buffer (16 hidden nodes * 3 outputs)
    pub inventory: f32, // Accumulated resources
    pub health: f32,
    pub age: f32,
}

impl Person {
    pub fn new(x: f32, y: f32) -> Self {
        let hidden_count = 16; // Give the agents a bigger brain so Rayon has real work to do
        let inputs = 4;       // Constant Bias, Normalized X, Normalized Y, Local Resource
        let outputs = 3;      // Delta Heading, Speed, Share Resource

        let mut rng = rand::thread_rng();
        let mut w1 = [0.0; 64];
        for i in 0..(inputs * hidden_count) as usize {
            w1[i] = (rng.r#gen::<f32>() * 2.0) - 1.0;
        }

        let mut w2 = [0.0; 48];
        for i in 0..(hidden_count * outputs) as usize {
            w2[i] = (rng.r#gen::<f32>() * 2.0) - 1.0;
        }

        Self {
            x,
            y,
            heading: rng.r#gen::<f32>() * PI * 2.0,
            speed: 1.4,
            hidden_count,
            w1,
            w2,
            inventory: 100.0, // Start life with $100 baseline
            health: 100.0,
            age: 0.0,
        }
    }

    pub fn reproduce(&mut self, cost: f32) -> Self {
        self.inventory -= cost; // Parent pays the biological/economic cost
        
        let mut child = *self;
        child.age = 0.0;
        child.health = 100.0;
        child.inventory = cost / 2.0; // Child starts with half the cost as inheritance
        
        let mut rng = rand::thread_rng();
        
        // 1. Mutate weights
        for w in child.w1.iter_mut().chain(child.w2.iter_mut()) {
            if rng.r#gen::<f32>() < 0.1 { *w += (rng.r#gen::<f32>() * 0.5) - 0.25; }
        }
        
        // 2. Structural mutation (Brain Growth)
        if rng.r#gen::<f32>() < 0.05 && child.hidden_count < 16 {
            let h = child.hidden_count as usize;
            for i in 0..4 { child.w1[h * 4 + i] = (rng.r#gen::<f32>() * 2.0) - 1.0; }
            for i in 0..3 { child.w2[h * 3 + i] = (rng.r#gen::<f32>() * 2.0) - 1.0; }
            child.hidden_count += 1;
        }
        
        child
    }
}