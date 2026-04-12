/*
 * Grand Sim Pro: A high-performance GPGPU evolutionary agent simulation.
 * Part of an independent research project into emergent biological complexity.
 *
 * Copyright (C) 2026 Ved Hirenkumar Suthar
 * Licensed under the GNU General Public License v3.0 or later.
 * * This software is provided "as is", without warranty of any kind.
 * See the LICENSE file in the project root for full license details.
 */

struct AgentState {
    x: f32,
    y: f32,
    heading: f32,
    speed: f32,
    hidden_count: u32,
    genetics_index: u32,
    gender: f32,
    reproduce_desire: f32,
    attack_intent: f32,
    rest_intent: f32,
    comms: array<f32, 12>,
    mems: array<f32, 24>,
    buy_intent: f32,
    sell_intent: f32,
    ask_price: f32,
    bid_price: f32,
    wealth: f32,
    drop_water_intent: f32,
    pickup_water_intent: f32,
    defend_intent: f32,
    build_road_intent: f32,
    build_house_intent: f32,
    build_farm_intent: f32,
    build_storage_intent: f32,
    destroy_infra_intent: f32,
    pheno_r: f32,
    pheno_g: f32,
    pheno_b: f32,
    emergency_intent: f32,
    _pad_agent2: f32,
    _pad_agent3: f32,
    food: f32,
    water: f32,
    stamina: f32,
    health: f32,
    age: f32,
    id: u32,
    gestation_timer: f32,
    is_pregnant: f32,
}

struct Genetics {
    w1_weights: array<f32, 1024>, // 128 * 8 Fixed-K Sparse weights
    w1_indices: array<u32, 1024>, // 128 * 8 Fixed-K Sparse indices
    w2: array<f32, 16384>, // 128 * 128
    w3: array<f32, 7168>, // 128 * 56
}

struct WorldConfig {
    map_width: u32,
    map_height: u32,
    display_width: u32,
    display_height: u32,
    regen_rate: f32,
    max_tile_resource: f32,
    max_tile_water: f32,
    tick_to_mins: f32,
}

struct SimulationConfig {
    agent_count: u32,
    spawn_group_size: u32,
    founder_count: u32,
    load_saved_agents_on_start: u32,
    current_tick: u32,
    visual_mode: u32,
    pad1: u32,
    pad2: u32,
}

struct BiologyConfig {
    base_speed: f32,
    max_age: f32,
    max_health: f32,
    max_stamina: f32,
    max_water: f32,
    puberty_age: f32,
    menopause_age: f32,
    gestation_period: f32,
    starvation_rate: f32,
    max_carry_weight: f32,
    infant_speed_mult: f32,
    infant_stamina_mult: f32,
}

struct EconomyConfig {
    baseline_cost: f32,
    move_cost_per_unit: f32,
    climb_penalty: f32,
    reproduction_cost: f32,
    boat_cost: f32,
    water_transfer_amount: f32,
    drop_amount: f32,
    base_gather_rate: f32,
    max_gather_rate: f32,
    base_spoilage_rate: f32,
    pad1: f32,
    pad2: f32,
}

struct GeneticsConfig {
    mutation_rate: f32,
    mutation_strength: f32,
    random_spawn_percentage: f32,
    crossover_rate: f32,
}

struct InfrastructureConfig {
    infra_cost: f32,
    max_infra: f32,
    decay_rate_roads: f32,
    decay_rate_housing: f32,
    decay_rate_farms: f32,
    decay_rate_storage: f32,
    road_speed_bonus: f32,
    housing_rest_bonus: f32,
    storage_rot_reduction: f32,
    build_ticks: f32,
    pad1: f32,
    pad2: f32,
}

struct CombatConfig {
    crowding_threshold: f32,
    pregnancy_speed_mult: f32,
    pregnancy_cost_mult: f32,
    defend_cost_mult: f32,
    bystander_damage: f32,
    attacker_damage: f32,
    steal_amount: f32,
    pad1: f32,
}

struct SimConfig {
    world: WorldConfig,
    sim: SimulationConfig,
    bio: BiologyConfig,
    eco: EconomyConfig,
    genetics: GeneticsConfig,
    infra: InfrastructureConfig,
    combat: CombatConfig,
}

struct CellState {
    res_value: atomic<i32>,
    population: f32,
    avg_speed: f32,
    avg_share: f32,
    avg_reproduce: f32,
    avg_aggression: f32,
    avg_pregnancy: f32,
    avg_turn: f32,
    avg_rest: f32,
    comm1: f32,
    comm2: f32,
    comm3: f32,
    comm4: f32,
    comm5: f32,
    comm6: f32,
    comm7: f32,
    comm8: f32,
    comm9: f32,
    comm10: f32,
    comm11: f32,
    comm12: f32,
    avg_ask: f32,
    avg_bid: f32,
    market_food: atomic<i32>,
    market_wealth: atomic<i32>,
    market_water: atomic<i32>,
    infra_roads: atomic<i32>,
    infra_housing: atomic<i32>,
    infra_farms: atomic<i32>,
    infra_storage: atomic<i32>,
    pheno_r: f32,
    pheno_g: f32,
    pheno_b: f32,
    base_moisture: f32,
    adult_count: atomic<i32>,
    _pad_infra3: f32,
}

