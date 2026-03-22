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
    pub pad: [f32; 2],            // WGSL Uniform buffers require 16-byte alignment
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
            max_age: 100000.0,        // Dies of old age after ~100k ticks
            max_health: 100.0,        // Max HP
            starvation_rate: 0.1,     // Loses 0.1 HP per tick when broke
            reproduction_cost: 500.0, // Takes $500 to raise a child
            pad: [0.0; 2],
        }
    }
}