/*
 * Grand Sim Pro: A high-performance GPGPU evolutionary agent simulation.
 * Part of an independent research project into emergent biological complexity.
 *
 * Copyright (C) 2026 Ved Hirenkumar Suthar
 * Licensed under the GNU General Public License v3.0 or later.
 * * This software is provided "as is", without warranty of any kind.
 * See the LICENSE file in the project root for full license details.
 */

use rand::Rng;
use noise::{NoiseFn, Perlin, Fbm};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CellState {
    pub res_value: i32,      // Fixed-point (val * 1000) for GPU atomics
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
    pub market_food: i32,    // Fixed-point (val * 1000) for GPU atomics
    pub market_wealth: i32,  // Fixed-point (val * 1000) for GPU atomics
    pub market_water: i32,   // Fixed-point (val * 1000) for GPU atomics
    pub shelter_level: f32,  // Physical structures built by agents (replaces padding to maintain 80 bytes)
    pub pheno_r: f32,        // Local dominant phenotype marker
    pub pheno_g: f32,
    pub pheno_b: f32,
    pub _pad_pheno: f32,     // Pad to maintain 16-byte align (96 bytes total)
}

pub struct Environment {
    pub height_map: Vec<f32>,
    pub map_cells: Vec<CellState>,
}

impl Environment {
    pub fn new(width: u32, height: u32, seed: u32, config: &crate::config::SimConfig) -> Self {
        let fbm = Fbm::<Perlin>::new(seed);
        
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

                height_map.push(val as f32);
                
                // Initialize land biomes based on elevation
                // Lower elevations start with more resources (lush), higher elevations are barren
                let elevation_mult = (1.0 - (val as f32 * 2.0)).clamp(0.05, 1.0);
                let base_res = if val >= 0.0 { config.max_tile_resource * elevation_mult } else { 0.0 };
                
                map_cells.push(CellState {
                    res_value: (base_res * 1000.0) as i32,
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
                    market_food: 50_000_000,
                    market_wealth: (base_res * 1000.0) as i32, // Cells start with money to buy initial farmed crops
                    market_water: if val <= 0.05 { (config.max_tile_water * 1000.0) as i32 } else { 0 }, // Shorelines & Oceans start with water
                shelter_level: 0.0,
                pheno_r: 0.0,
                pheno_g: 0.0,
                pheno_b: 0.0,
                _pad_pheno: 0.0,
                });
            }
        }

        // --- Procedural River Generation ---
        let num_rivers = (width * height) / 15000; // Map size scales the number of rivers
        let mut rng = rand::thread_rng();

        for _ in 0..num_rivers {
            let mut sx = rng.gen_range(0..width);
            let mut sy = rng.gen_range(0..height);
            let mut attempts = 0;
            
            // Find a high elevation mountain spring to start the river
            while height_map[(sy * width + sx) as usize] < 0.3 && attempts < 100 {
                sx = rng.gen_range(0..width);
                sy = rng.gen_range(0..height);
                attempts += 1;
            }
            if attempts == 100 { continue; } // Failed to find a mountain

            let mut cx = sx;
            let mut cy = sy;
            let mut river_path = Vec::new();
            
            // Trace downhill until we hit a pit, the ocean, or the max length
            while river_path.len() < 1500 {
                let idx = (cy * width + cx) as usize;
                river_path.push(idx);
                
                let mut min_h = height_map[idx];
                let mut nx = cx;
                let mut ny = cy;
                
                // Check all 8 neighbors (accounting for seamless 4D wrapping)
                let dirs = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)];
                for &(dx, dy) in dirs.iter() {
                    let tx = ((cx as i32 + dx + width as i32) % width as i32) as u32;
                    let ty = ((cy as i32 + dy + height as i32) % height as i32) as u32;
                    let tidx = (ty * width + tx) as usize;
                    if height_map[tidx] < min_h {
                        min_h = height_map[tidx];
                        nx = tx;
                        ny = ty;
                    }
                }
                
                if nx == cx && ny == cy { break; } // Hit a local pit/lake
                cx = nx;
                cy = ny;
                if min_h <= 0.05 { break; } // Reached the ocean/shoreline
            }

            // Carve the river into the map arrays
            for idx in river_path {
                map_cells[idx].market_water = (config.max_tile_water * 1000.0) as i32;
                
                // Erode terrain to 0.01 (Shoreline level) so it naturally replenishes water forever
                // and allows crossing on foot without a boat (boat requires < 0.0)
                if height_map[idx] > 0.01 {
                    height_map[idx] = 0.01;
                }
            }
        }

        Self { height_map, map_cells }
    }
}