/*
 * Grand Sim Pro: A high-performance GPGPU evolutionary agent simulation.
 * Part of an independent research project into emergent biological complexity.
 *
 * Copyright (C) 2026 Ved Hirenkumar Suthar
 * Licensed under the GNU General Public License v3.0 or later.
 * * This software is provided "as is", without warranty of any kind.
 * See the LICENSE file in the project root for full license details.
 */

use noise::{NoiseFn, Perlin, Fbm};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CellState {
    pub res_value: f32,
    pub population: f32,     // Leaves a trace when agents step here
    pub avg_speed: f32,      // Running average of agent speeds
    pub avg_share: f32,      // Running average of sharing output
    pub avg_reproduce: f32,  // Running average of reproduce desire
    pub avg_aggression: f32, // Hostile intent traces
    pub avg_pregnancy: f32,  // Signals males to protect rather than attack
    pub avg_turn: f32,       // Abstract herd direction
    pub avg_rest: f32,       // Abstract encampment state
    pub comm1: f32,          // Abstract communication channel 1
    pub comm2: f32,          // Abstract communication channel 2
    pub comm3: f32,          // Abstract communication channel 3
    pub comm4: f32,          // Abstract communication channel 4
    pub avg_ask: f32,        // Average price asked to sell food
    pub avg_bid: f32,        // Average price bid to buy food
    pub market_food: f32,    // Physical food in cell liquidity pool
    pub market_wealth: f32,  // Wealth available to buy food from agents
    pub pad1: [f32; 1],      // Exactly 80 bytes total for GPU strict alignment
}

pub struct Environment {
    pub map_data: Vec<u8>,
    pub height_map: Vec<f32>,
    pub map_cells: Vec<CellState>,
}

impl Environment {
    pub fn new(width: u32, height: u32, seed: u32, config: &crate::config::SimConfig) -> Self {
        let fbm = Fbm::<Perlin>::new(seed);
        
        let mut map_data = Vec::with_capacity((width * height * 4) as usize);
        let mut height_map = Vec::with_capacity((width * height) as usize);
        let mut map_cells = Vec::with_capacity((width * height) as usize);

        // Calculate the radius for our 4D mapping to match the original noise scale
        let radius_x = width as f64 / (150.0 * 2.0 * std::f64::consts::PI);
        let radius_y = height as f64 / (150.0 * 2.0 * std::f64::consts::PI);

        for y in 0..height {
            for x in 0..width {
                // Convert X and Y to angles
                let angle_x = (x as f64 / width as f64) * 2.0 * std::f64::consts::PI;
                let angle_y = (y as f64 / height as f64) * 2.0 * std::f64::consts::PI;
                
                // Sample 4D noise for seamless wrapping
                let val = fbm.get([
                    angle_x.cos() * radius_x, angle_x.sin() * radius_x,
                    angle_y.cos() * radius_y, angle_y.sin() * radius_y
                ]);

                let mut color = match val {
                    v if v < -0.2 => [10, 50, 150, 255],
                    v if v < 0.0  => [30, 100, 200, 255],
                    v if v < 0.1  => [240, 230, 140, 255],
                    v if v < 0.4  => [34, 139, 34, 255],
                    _             => [100, 100, 100, 255],
                };

                // Add topological contour lines for height visualization
                if val >= 0.0 && (val % 0.1).abs() < 0.015 {
                    color = [0, 0, 0, 180]; // Dark contour line
                }

                map_data.extend_from_slice(&color);
                height_map.push(val as f32);
                
                // Initialize land based on max configured economic scale
                let base_res = if val >= 0.0 { config.max_tile_resource } else { 0.0 };
                map_cells.push(CellState {
                    res_value: base_res,
                    population: 0.0,
                    avg_speed: 0.0,
                    avg_share: 0.0,
                    avg_reproduce: 0.0,
                    avg_aggression: 0.0,
                    avg_pregnancy: 0.0,
                    avg_turn: 0.0,
                    avg_rest: 0.0,
                    comm1: 0.0,
                    comm2: 0.0,
                    comm3: 0.0,
                    comm4: 0.0,
                    avg_ask: 1.0,
                    avg_bid: 1.0,
                    market_food: 50.0,
                    market_wealth: base_res, // Cells start with money to buy initial farmed crops
                    pad1: [0.0; 1],
                });
            }
        }
        Self { map_data, height_map, map_cells }
    }
}