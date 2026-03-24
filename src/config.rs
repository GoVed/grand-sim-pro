/*
 * Grand Sim Pro: A high-performance GPGPU evolutionary agent simulation.
 * Part of an independent research project into emergent biological complexity.
 *
 * Copyright (C) 2026 Ved Hirenkumar Suthar
 * Licensed under the GNU General Public License v3.0 or later.
 * * This software is provided "as is", without warranty of any kind.
 * See the LICENSE file in the project root for full license details.
 */

use serde::{Serialize, Deserialize};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Serialize, Deserialize)]
pub struct SimConfig {
    pub base_speed: f32,          // Max pixels to move per minute
    pub baseline_cost: f32,       // USD burned per minute just existing (food/water)
    pub move_cost_per_unit: f32,  // USD burned per pixel traveled (calories)
    pub climb_penalty: f32,       // Exertion multiplier for walking uphill
    pub base_gather_rate: f32,    // USD gathered per minute with bare hands
    pub max_gather_rate: f32,     // Max USD gathered per minute with heavy tools
    pub max_tile_resource: f32,   // Maximum USD value a 10x10m plot of land can hold
    pub boat_cost: f32,           // USD required to build a boat and cross water
    pub drop_amount: f32,         // USD gifted/dropped into the environment at once
    pub regen_rate: f32,          // USD regenerated per tile per minute
    pub max_age: f32,             // Max ticks before death by old age
    pub max_health: f32,          // Baseline health points
    pub starvation_rate: f32,     // Health lost per tick when starving
    pub reproduction_cost: f32,   // USD burned to spawn an offspring
    pub map_width: u32,           // Number of procedural tiles on X axis
    pub map_height: u32,          // Number of procedural tiles on Y axis
    pub display_width: u32,       // Resolution width of the Macroquad window
    pub display_height: u32,      // Resolution height of the Macroquad window
    pub agent_count: u32,         // Total fixed target population to simulate
    pub current_tick: u32,        // Tracks global seasons
    pub max_stamina: f32,
    pub max_water: f32,
    pub puberty_age: f32,         // Minimum age to mate
    pub menopause_age: f32,       // Maximum age to mate
    pub gestation_period: f32,    // Time female carries the child
    pub tick_to_mins: f32,        // Conversion rate
    pub pad1: [u32; 2],           // Strict 16-byte uniform alignment pad
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            base_speed: 8.0,          // 8 units = 80 meters per minute (~5 km/h walking speed)
            baseline_cost: 0.05,      // $0.05 / min = $72 / day (basic cost of living)
            move_cost_per_unit: 0.01, // 80m walk costs $0.08 / min ($4.80 / hr energy cost)
            climb_penalty: 5.0,       // Steep slopes massively multiply energy cost
            base_gather_rate: 0.30,   // $0.30 / min = $18 / hr (baseline wage)
            max_gather_rate: 2.00,    // $2.00 / min = $120 / hr (high-skilled/machinery)
            max_tile_resource: 10000.0, // $10,000 worth of harvestable value per tile
            boat_cost: 5000.0,        // $5,000 to construct a viable watercraft
            drop_amount: 1.0,         // Donate $1 chunks per tick
            regen_rate: 0.01,         // Nature regrows $0.01 per minute ($14/day)
            max_age: 4204800.0,       // Dies of old age after ~80 years
            max_health: 100.0,        // Max HP
            starvation_rate: 0.1,     // Loses 0.1 HP per tick when broke
            reproduction_cost: 500.0, // Takes $500 to raise a child
            map_width: 800,           // Default 800 tiles wide
            map_height: 600,          // Default 600 tiles tall
            display_width: 1280,      // Scale up default window size
            display_height: 720,
            agent_count: 4000,        // 4k initial default population
            current_tick: 0,
            max_stamina: 100.0,
            max_water: 100.0,
            puberty_age: 630720.0,    // 12 years
            menopause_age: 2628000.0, // 50 years
            gestation_period: 38880.0,// 9 months
            tick_to_mins: 10.0,       // 1 tick = 10 minutes
            pad1: [0; 2],
        }
    }
}