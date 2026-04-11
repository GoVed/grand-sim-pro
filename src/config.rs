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
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Serialize, Deserialize, PartialEq, Debug)]
pub struct WorldConfig {
    pub map_width: u32,
    pub map_height: u32,
    pub display_width: u32,
    pub display_height: u32,
    pub regen_rate: f32,
    pub max_tile_resource: f32,
    pub max_tile_water: f32,
    pub tick_to_mins: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Serialize, Deserialize, PartialEq, Debug)]
pub struct SimulationConfig {
    pub agent_count: u32,
    pub spawn_group_size: u32,
    pub founder_count: u32,
    pub load_saved_agents_on_start: u32,
    pub current_tick: u32,
    pub visual_mode: u32,
    pub pad1: u32,
    pub pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Serialize, Deserialize, PartialEq, Debug)]
pub struct BiologyConfig {
    pub base_speed: f32,
    pub max_age: f32,
    pub max_health: f32,
    pub max_stamina: f32,
    pub max_water: f32,
    pub puberty_age: f32,
    pub menopause_age: f32,
    pub gestation_period: f32,
    pub starvation_rate: f32,
    pub max_carry_weight: f32,
    pub infant_speed_mult: f32,
    pub infant_stamina_mult: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Serialize, Deserialize, PartialEq, Debug)]
pub struct EconomyConfig {
    pub baseline_cost: f32,
    pub move_cost_per_unit: f32,
    pub climb_penalty: f32,
    pub reproduction_cost: f32,
    pub boat_cost: f32,
    pub water_transfer_amount: f32,
    pub drop_amount: f32,
    pub base_gather_rate: f32,
    pub max_gather_rate: f32,
    pub base_spoilage_rate: f32,
    pub pad1: f32,
    pub pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Serialize, Deserialize, PartialEq, Debug)]
pub struct GeneticsConfig {
    pub mutation_rate: f32,
    pub mutation_strength: f32,
    pub random_spawn_percentage: f32,
    pub crossover_rate: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Serialize, Deserialize, PartialEq, Debug)]
pub struct InfrastructureConfig {
    pub infra_cost: f32,
    pub max_infra: f32,
    pub decay_rate_roads: f32,
    pub decay_rate_housing: f32,
    pub decay_rate_farms: f32,
    pub decay_rate_storage: f32,
    pub road_speed_bonus: f32,
    pub housing_rest_bonus: f32,
    pub storage_rot_reduction: f32,
    pub build_ticks: f32,
    pub pad1: f32,
    pub pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Serialize, Deserialize, PartialEq, Debug)]
pub struct CombatConfig {
    pub crowding_threshold: f32,
    pub pregnancy_speed_mult: f32,
    pub pregnancy_cost_mult: f32,
    pub defend_cost_mult: f32,
    pub bystander_damage: f32,
    pub attacker_damage: f32,
    pub steal_amount: f32,
    pub pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Serialize, Deserialize, PartialEq, Debug)]
pub struct SimConfig {
    pub world: WorldConfig,
    pub sim: SimulationConfig,
    pub bio: BiologyConfig,
    pub eco: EconomyConfig,
    pub genetics: GeneticsConfig,
    pub infra: InfrastructureConfig,
    pub combat: CombatConfig,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            world: WorldConfig {
                map_width: 800,
                map_height: 600,
                display_width: 1280,
                display_height: 720,
                regen_rate: 0.01,
                max_tile_resource: 10000.0,
                max_tile_water: 1000.0,
                tick_to_mins: 10.0,
            },
            sim: SimulationConfig {
                agent_count: 1000,
                spawn_group_size: 100,
                founder_count: 64,
                load_saved_agents_on_start: 2,
                current_tick: 0,
                visual_mode: 0,
                pad1: 0,
                pad2: 0,
            },
            bio: BiologyConfig {
                base_speed: 8.0,
                max_age: 4204800.0,
                max_health: 100.0,
                max_stamina: 100.0,
                max_water: 25.0,
                puberty_age: 630720.0,
                menopause_age: 2628000.0,
                gestation_period: 38880.0,
                starvation_rate: 0.1,
                max_carry_weight: 100.0,
                infant_speed_mult: 0.1,
                infant_stamina_mult: 0.2,
            },
            eco: EconomyConfig {
                baseline_cost: 0.015,
                move_cost_per_unit: 0.002,
                climb_penalty: 5.0,
                reproduction_cost: 500.0,
                boat_cost: 5000.0,
                water_transfer_amount: 5.0,
                drop_amount: 1.0,
                base_gather_rate: 0.30,
                max_gather_rate: 2.00,
                base_spoilage_rate: 0.0001,
                pad1: 0.0,
                pad2: 0.0,
            },
            genetics: GeneticsConfig {
                mutation_rate: 0.05,
                mutation_strength: 0.05,
                random_spawn_percentage: 0.1,
                crossover_rate: 0.5,
            },
            infra: InfrastructureConfig {
                infra_cost: 100.0,
                max_infra: 1000.0,
                decay_rate_roads: 0.004,
                decay_rate_housing: 0.001,
                decay_rate_farms: 0.02,
                decay_rate_storage: 0.002,
                road_speed_bonus: 2.0,
                housing_rest_bonus: 0.3,
                storage_rot_reduction: 0.9,
                build_ticks: 6.0,
                pad1: 0.0,
                pad2: 0.0,
            },
            combat: CombatConfig {
                crowding_threshold: 15.0,
                pregnancy_speed_mult: 0.7,
                pregnancy_cost_mult: 1.5,
                defend_cost_mult: 1.5,
                bystander_damage: 0.5,
                attacker_damage: 2.0,
                steal_amount: 5.0,
                pad1: 0.0,
            },
        }
    }
}
