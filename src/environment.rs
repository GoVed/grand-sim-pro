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
    pub infra_roads: i32,    // Fixed-point (val * 1000) for GPU atomics
    pub infra_housing: i32,  // Fixed-point (val * 1000) for GPU atomics
    pub infra_farms: i32,    // Fixed-point (val * 1000) for GPU atomics
    pub infra_storage: i32,  // Fixed-point (val * 1000) for GPU atomics
    pub pheno_r: f32,        // Local dominant phenotype marker
    pub pheno_g: f32,
    pub pheno_b: f32,
    pub base_moisture: f32,  // Replaces pad: Defines the local biome humidity
    pub _pad_infra2: f32,
    pub _pad_infra3: f32,
}

pub struct Environment {
    pub height_map: Vec<f32>,
    pub map_cells: Vec<CellState>,
}

impl Environment {
    pub fn new(width: u32, height: u32, seed: u32, config: &crate::config::SimConfig) -> Self {
        let fbm_elevation = Fbm::<Perlin>::new(seed);
        let fbm_moisture = Fbm::<Perlin>::new(seed.wrapping_add(12345)); // Offset seed for moisture
        
        let mut height_map = Vec::with_capacity((width * height) as usize);
        let mut map_cells = Vec::with_capacity((width * height) as usize);

        // Using a massive scalar for true global continents
        let r_elev = width as f64 / 450.0;
        let r_moist = width as f64 / 250.0;

        for y in 0..height {
            for x in 0..width {
                // Spherical Mapping: X maps to Longitude (0 to 2PI), Y maps to Latitude (0 to PI)
                let theta = (x as f64 / width as f64) * 2.0 * std::f64::consts::PI;
                let phi = (y as f64 / height as f64) * std::f64::consts::PI;
                
                let nx = phi.sin() * theta.cos();
                let ny = phi.sin() * theta.sin();
                let nz = phi.cos();
                
                // Sample mathematically perfect 3D spherical noise
                let val = (fbm_elevation.get([nx * r_elev, ny * r_elev, nz * r_elev]) as f32) - 0.05;
                let moisture = fbm_moisture.get([nx * r_moist, ny * r_moist, nz * r_moist]) as f32;

                height_map.push(val as f32);
                
                // Initialize land biomes based on elevation
                // Lower elevations start with more resources (lush), higher elevations are barren
                let elevation_mult = (1.0 - (val as f32 * 2.0)).clamp(0.05, 1.0);
                
                let dist_from_equator = (y as f32 - height as f32 / 2.0).abs() / (height as f32 / 2.0);
                let base_temp = (1.0 - dist_from_equator * 2.0) - val.max(0.0) * 2.0;
                
                let mut biome_mult = 1.0;
                if base_temp < -0.4 { biome_mult = 0.01; } // Snow
                else if base_temp < -0.2 { biome_mult = 0.05; } // Tundra
                else if moisture < -0.3 { biome_mult = 0.02; } // Desert
                else if moisture < 0.0 { biome_mult = 0.4; } // Savanna
                else if moisture > 0.3 { biome_mult = 1.5; } // Jungle
                
                let base_res = if val >= 0.0 { config.max_tile_resource * elevation_mult * biome_mult } else { 0.0 };
                
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
                    infra_roads: 0,
                    infra_housing: 0,
                    infra_farms: 0,
                    infra_storage: 0,
                    pheno_r: 0.0,
                    pheno_g: 0.0,
                    pheno_b: 0.0,
                    base_moisture: moisture,
                    _pad_infra2: 0.0,
                    _pad_infra3: 0.0,
                });
            }
        }

        // --- Procedural River Generation ---
        let num_rivers = (width * height) / 15000; // Map size scales the number of rivers
        let mut rng = rand::thread_rng();
        
        // Helper closure to calculate spherical coordinate wrapping for rivers
        let wrap_coords = |cx: i32, cy: i32, dx: i32, dy: i32, w: i32, h: i32| -> (u32, u32) {
            let mut nx = cx + dx;
            let mut ny = cy + dy;
            if ny < 0 { ny = -ny; nx += w / 2; } // North Pole cross
            else if ny >= h { ny = 2 * h - 1 - ny; nx += w / 2; } // South Pole cross
            
            nx = nx.rem_euclid(w);
            (nx as u32, ny as u32)
        };

        for _ in 0..num_rivers {
            let mut sx = rng.gen_range(0..width);
            let mut sy = rng.gen_range(0..height);
            let mut attempts = 0;
            
            // Find a high elevation mountain spring to start the river
            while height_map[(sy * width + sx) as usize] < 0.2 && attempts < 100 {
                sx = rng.gen_range(0..width);
                sy = rng.gen_range(0..height);
                attempts += 1;
            }
            if attempts == 100 { continue; } // Failed to find a mountain

            let mut cx = sx;
            let mut cy = sy;
            let mut river_path = Vec::new();
            let mut is_lake = false;
            
            // Trace downhill until we hit a pit, the ocean, or the max length
            while river_path.len() < 1500 {
                let idx = (cy * width + cx) as usize;
                river_path.push(idx);
                
                let mut min_h = height_map[idx];
                let mut nx = cx;
                let mut ny = cy;
                
                // Check all 8 neighbors (accounting for spherical wrapping)
                let dirs = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)];
                for &(dx, dy) in dirs.iter() {
                    let (tx, ty) = wrap_coords(cx as i32, cy as i32, dx, dy, width as i32, height as i32);
                    let tidx = (ty * width + tx) as usize;
                    if height_map[tidx] < min_h {
                        min_h = height_map[tidx];
                        nx = tx;
                        ny = ty;
                    }
                }
                
                if nx == cx && ny == cy {
                    is_lake = true;
                    break;
                } // Hit a local pit/lake
                cx = nx;
                cy = ny;
                if min_h <= 0.05 { break; } // Reached the ocean/shoreline
            }

            // Carve the river into the map arrays
            for &idx in &river_path {
                map_cells[idx].market_water = (config.max_tile_water * 1000.0) as i32;
                
                // Erode terrain to -0.01 (Shallow Water) so it naturally replenishes water forever
                // Requires a boat to cross, making rivers and lakes actual water obstacles
                if height_map[idx] > -0.01 {
                    height_map[idx] = -0.01;
                }
                
                // Carve walkable riverbanks (0.01) so agents can reach the water without a boat
                let cx = (idx as u32) % width;
                let cy = (idx as u32) / width;
                let dirs = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)];
                for &(dx, dy) in dirs.iter() {
                    let (tx, ty) = wrap_coords(cx as i32, cy as i32, dx, dy, width as i32, height as i32);
                    let tidx = (ty * width + tx) as usize;
                    
                    map_cells[tidx].market_water = (config.max_tile_water * 1000.0) as i32;
                    if height_map[tidx] > 0.01 {
                        height_map[tidx] = 0.01;
                    }
                }
            }

            // If the river ends in a landlocked pit, flood it to create a lake
            if is_lake {
                if let Some(&last_idx) = river_path.last() {
                    let lx = (last_idx as u32) % width;
                    let ly = (last_idx as u32) / width;
                    let lake_radius = rng.gen_range(2..6) as i32;
                    
                    for dy in -(lake_radius + 1)..=(lake_radius + 1) {
                        for dx in -(lake_radius + 1)..=(lake_radius + 1) {
                            let dist_sq = dx * dx + dy * dy;
                            if dist_sq <= lake_radius * lake_radius {
                                let (tx, ty) = wrap_coords(lx as i32, ly as i32, dx, dy, width as i32, height as i32);
                                let tidx = (ty * width + tx) as usize;
                                
                                map_cells[tidx].market_water = (config.max_tile_water * 1000.0) as i32;
                                if height_map[tidx] > -0.01 { height_map[tidx] = -0.01; }
                            } else if dist_sq <= (lake_radius + 1) * (lake_radius + 1) {
                                let (tx, ty) = wrap_coords(lx as i32, ly as i32, dx, dy, width as i32, height as i32);
                                let tidx = (ty * width + tx) as usize;
                                
                                map_cells[tidx].market_water = (config.max_tile_water * 1000.0) as i32;
                                if height_map[tidx] > 0.01 { height_map[tidx] = 0.01; }
                            }
                        }
                    }
                }
            }
        }

        Self { height_map, map_cells }
    }
}