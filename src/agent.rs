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
    pub w2: [f32; 32], // Fixed size for GPU Buffer (16 hidden nodes * 2 outputs)
    pub inventory: f32, // Accumulated resources
    pub pad: [f32; 2], // 8 bytes of padding ensures exact 416-byte struct size
}

impl Person {
    pub fn new(x: f32, y: f32) -> Self {
        let hidden_count = 16; // Give the agents a bigger brain so Rayon has real work to do
        let inputs = 4;       // Constant Bias, Normalized X, Normalized Y, Local Resource
        let outputs = 2;      // e.g., Delta Heading, Speed

        let mut rng = rand::thread_rng();
        let mut w1 = [0.0; 64];
        for i in 0..(inputs * hidden_count) as usize {
            w1[i] = (rng.r#gen::<f32>() * 2.0) - 1.0;
        }

        let mut w2 = [0.0; 32];
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
            inventory: 0.0,
            pad: [0.0; 2],
        }
    }
}