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

@group(0) @binding(0) var<storage, read_write> agents: array<AgentState>;
@group(0) @binding(1) var<storage, read> map_heights: array<f32>;
@group(0) @binding(2) var<storage, read_write> map_cells: array<CellState>;
@group(0) @binding(3) var<uniform> cfg: SimConfig;
@group(0) @binding(4) var<storage, read_write> render_buffer: array<u32>;
@group(0) @binding(5) var<storage, read_write> genetics: array<Genetics>;

// Fast Local Memory Cache for workgroup map tiles (e.g., 10x10 patch + padding)
// This significantly reduces global memory latency for vision sampling.
var<workgroup> lds_res: array<f32, 256>; // 16x16 patch
var<workgroup> lds_height: array<f32, 256>;
var<workgroup> lds_pop: array<f32, 256>;
var<workgroup> lds_base_x: u32;
var<workgroup> lds_base_y: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_index) local_idx: u32) {
    let idx = global_id.x;
    if (idx >= arrayLength(&agents)) { return; }

    // CRITICAL PERFORMANCE FIX: DO NOT load the massive 28KB Agent struct into registers!
    // We access members directly from the storage buffer instead.
    if (agents[idx].health <= 0.0) { return; }
    let g_idx = agents[idx].genetics_index;

    // Store pre-update state for dopaminergic modulation
    let prev_health = agents[idx].health;
    let prev_food = agents[idx].food;
    let prev_wealth = agents[idx].wealth;

    agents[idx].age = agents[idx].age + 1.0;

    let map_w = cfg.world.map_width;
    let map_h = cfg.world.map_height;
    let map_w_f32 = f32(map_w);
    let map_h_f32 = f32(map_h);
    let max_idx = map_w * map_h - 1u;

    // --- Cooperative LDS Map Loading ---
    // All 64 threads in the workgroup are spatially close due to sorting.
    // Thread 0 calculates the bounding box and all threads load a tile patch.
    if (local_idx == 0u) {
        lds_base_x = u32(agents[idx].x) / 8u * 8u;
        lds_base_y = u32(agents[idx].y) / 8u * 8u;
    }
    workgroupBarrier();

    // Each thread in workgroup (0-63) loads 4 tiles (4 * 64 = 256 tiles total in 16x16 patch)
    for (var i = 0u; i < 4u; i = i + 1u) {
        let l_idx = local_idx * 4u + i;
        let lx = l_idx % 16u;
        let ly = l_idx / 16u;
        let gx = (lds_base_x + lx) % map_w;
        let gy = clamp(lds_base_y + ly, 0u, map_h - 1u);
        let g_idx = gy * map_w + gx;

        lds_res[l_idx] = f32(atomicLoad(&map_cells[g_idx].res_value)) / 1000.0;
        lds_height[l_idx] = map_heights[g_idx];
        lds_pop[l_idx] = map_cells[g_idx].population;
    }
    workgroupBarrier(); // Sync before sampling from LDS

    // Pre-calculate common vision parameters
    let ax = agents[idx].x;
    let ay = agents[idx].y;
    let ah = agents[idx].heading;
    let current_idx = u32(ay) * map_w + u32(ax);
    let safe_current_idx = clamp(current_idx, 0u, max_idx);
    let current_height = map_heights[safe_current_idx];

    // Read attributes directly to avoid register pressure
    let local_population = map_cells[safe_current_idx].population;
    let local_avg_speed = map_cells[safe_current_idx].avg_speed;
    let local_avg_share = map_cells[safe_current_idx].avg_share;
    let local_avg_reproduce = map_cells[safe_current_idx].avg_reproduce;
    let local_avg_aggression = map_cells[safe_current_idx].avg_aggression;
    let local_avg_pregnancy = map_cells[safe_current_idx].avg_pregnancy;
    let local_avg_turn = map_cells[safe_current_idx].avg_turn;
    let local_avg_rest = map_cells[safe_current_idx].avg_rest;
    let local_comm1 = map_cells[safe_current_idx].comm1;
    let local_comm2 = map_cells[safe_current_idx].comm2;
    let local_comm3 = map_cells[safe_current_idx].comm3;
    let local_comm4 = map_cells[safe_current_idx].comm4;
    let local_avg_ask = map_cells[safe_current_idx].avg_ask;
    let local_avg_bid = map_cells[safe_current_idx].avg_bid;
    let local_pheno_r = map_cells[safe_current_idx].pheno_r;
    let local_pheno_g = map_cells[safe_current_idx].pheno_g;
    let local_pheno_b = map_cells[safe_current_idx].pheno_b;

    let local_res_value = f32(atomicLoad(&map_cells[safe_current_idx].res_value)) / 1000.0;
    let local_market_water = f32(atomicLoad(&map_cells[safe_current_idx].market_water)) / 1000.0;
    let local_infra_roads = f32(atomicLoad(&map_cells[safe_current_idx].infra_roads)) / 1000.0;
    let local_infra_housing = f32(atomicLoad(&map_cells[safe_current_idx].infra_housing)) / 1000.0;
    let local_infra_farms = f32(atomicLoad(&map_cells[safe_current_idx].infra_farms)) / 1000.0;
    let local_infra_storage = f32(atomicLoad(&map_cells[safe_current_idx].infra_storage)) / 1000.0;
    
    // Generate chaos/noise using the agent's spatial position
    let pseudo_rand = fract(sin(dot(vec2<f32>(ax, ay), vec2<f32>(12.9898, 78.233))) * 43758.5453);

    // --- Day/Night Cycle ---
    let ticks_per_day = 24.0 * 60.0 / cfg.world.tick_to_mins;
    let day_cycle_progress = (f32(cfg.sim.current_tick) % ticks_per_day) / ticks_per_day; // 0.0 to 1.0
    let day_intensity = sin(day_cycle_progress * 6.28318 - 1.5708) * 0.5 + 0.5; // Sine wave, 0.0 at midnight, 1.0 at noon

    // Environment & Vision Pre-Calculation
    let season_time = f32(cfg.sim.current_tick) / 100000.0 * 6.28318;
    let season_sine = sin(season_time);
    let dist_from_equator = abs(ay - map_h_f32 / 2.0) / (map_h_f32 / 2.0);
    let base_temp = (1.0 - dist_from_equator * 2.0) - max(0.0, current_height * 2.0);
    let local_temp = base_temp + season_sine * 0.5 - (1.0 - day_intensity) * 0.3; 
    let effective_temp = local_temp + (local_infra_housing / cfg.infra.max_infra) * 2.0; // Houses provide massive insulation

    // Longitude Convergence: Simulates the narrowing of the sphere at the poles
    let agent_lat = (abs(ay - map_h_f32 / 2.0) / (map_h_f32 / 2.0)) * 1.570796; // 0 to PI/2
    let lon_scale = 1.0 / max(cos(agent_lat), 0.15); // Up to ~6.6x horizontal stretch at the extreme poles

    // Flattened 3x3 LiDAR Vision Grid
    var vis_mult = 1.0;
    if (agents[idx].rest_intent > 0.5 || agents[idx].stamina <= 0.0) { vis_mult = 0.1; } // Eyes are closed while sleeping
    let vision_multiplier = (0.3 + day_intensity * 0.7) * vis_mult; 

    // 1. Neural Net Processing
    var inputs = array<f32, 184>();
    inputs[0] = 1.0;
    inputs[1] = local_res_value / 1000.0;
    inputs[2] = local_population;
    inputs[3] = local_avg_speed + (pseudo_rand * 0.1 - 0.05);
    inputs[4] = local_avg_share + (pseudo_rand * 0.1 - 0.05);
    inputs[5] = local_avg_reproduce + (pseudo_rand * 0.1 - 0.05);
    inputs[6] = local_avg_aggression;
    inputs[7] = local_avg_pregnancy;
    inputs[8] = local_avg_turn;
    inputs[9] = local_avg_rest;
    
    inputs[10] = local_comm1;
    inputs[11] = local_comm2;
    inputs[12] = local_comm3;
    inputs[13] = local_comm4;
    for (var c = 4u; c < 12u; c = c + 1u) { inputs[10 + c] = 0.0; } // comm5 to 12

    inputs[22] = agents[idx].health / cfg.bio.max_health;
    inputs[23] = clamp((agents[idx].food / 1000.0) / cfg.eco.boat_cost, 0.0, 2.0); 
    inputs[24] = agents[idx].water / cfg.bio.max_water;
    inputs[25] = agents[idx].stamina / cfg.bio.max_stamina;
    inputs[26] = agents[idx].age / cfg.bio.max_age;
    inputs[27] = agents[idx].gender;
    inputs[28] = local_temp;
    inputs[29] = season_sine;
    inputs[30] = agents[idx].is_pregnant; 
    inputs[31] = clamp(((agents[idx].food / 1000.0) + agents[idx].water) / cfg.bio.max_carry_weight, 0.0, 2.0); // Food limit weight
    inputs[32] = local_population / cfg.combat.crowding_threshold;
    
    for (var m = 0u; m < 24u; m = m + 1u) {
        inputs[33 + m] = agents[idx].mems[m];
    }
    
    inputs[57] = agents[idx].wealth / cfg.eco.boat_cost;
    inputs[58] = local_avg_ask / 10.0;
    inputs[59] = local_avg_bid / 10.0;
    inputs[60] = day_intensity;
    inputs[61] = agents[idx].pheno_r;
    inputs[62] = agents[idx].pheno_g;
    inputs[63] = agents[idx].pheno_b;
    inputs[64] = local_pheno_r;
    inputs[65] = local_pheno_g;
    inputs[66] = local_pheno_b;
    
    inputs[179] = local_infra_roads / cfg.infra.max_infra;
    inputs[180] = local_infra_housing / cfg.infra.max_infra;
    inputs[181] = local_infra_farms / cfg.infra.max_infra;
    inputs[182] = local_infra_storage / cfg.infra.max_infra;
    inputs[183] = 0.0;

    var input_idx = 67u;
    let cos_h = cos(ah);
    let sin_h = sin(ah);
    let view_spacing = 8.0; // Sample tiles 8 pixels apart for good spread

    // Read the 3x3 grid (skipping the center since we already have local data)
    for (var ly_i = 1; ly_i >= -1; ly_i -= 1) { // Front to Back
        for (var lx_i = -1; lx_i <= 1; lx_i += 1) { // Left to Right
            if (lx_i == 0 && ly_i == 0) { continue; } // Skip the agent's current center cell
            
            let ly = f32(ly_i);
            let lx = f32(lx_i);

            let fwd = ly * view_spacing;
            let lat = lx * view_spacing;

            // Rotate relative to agent heading
            var rot_x = fwd * cos_h - lat * sin_h;
            var rot_y = fwd * sin_h + lat * cos_h;
            
            // Apply longitude convergence strictly to the global X axis
            rot_x = rot_x * lon_scale;

            let sample_x = ax + rot_x;
            let sample_y = ay + rot_y;

            // Spherical Map Wrap for LiDAR
            var wrap_x = sample_x;
            var wrap_y = sample_y;
            if (wrap_y < 0.0) { wrap_y = -wrap_y; wrap_x = wrap_x + map_w_f32 / 2.0; }
            else if (wrap_y >= map_h_f32) { wrap_y = 2.0 * map_h_f32 - 1.0 - wrap_y; wrap_x = wrap_x + map_w_f32 / 2.0; }
            
            wrap_x = wrap_x % map_w_f32; 
            if (wrap_x < 0.0) { wrap_x = wrap_x + map_w_f32; }

            // --- LDS Sampling Logic ---
            let ux = u32(wrap_x);
            let uy = u32(wrap_y);
            
            var s_res = 0.0;
            var s_height = 0.0;
            var s_pop = 0.0;
            
            // If the sample falls within our LDS patch, use fast memory
            if (ux >= lds_base_x && ux < lds_base_x + 16u && uy >= lds_base_y && uy < lds_base_y + 16u) {
                let l_idx = (uy - lds_base_y) * 16u + (ux - lds_base_x);
                s_res = lds_res[l_idx];
                s_height = lds_height[l_idx];
                s_pop = lds_pop[l_idx];
            } else {
                // Fallback to Global VRAM for agents at the edges of the workgroup cluster
                let sample_idx = clamp(uy * map_w + ux, 0u, max_idx);
                s_res = f32(atomicLoad(&map_cells[sample_idx].res_value)) / 1000.0;
                s_height = map_heights[sample_idx];
                s_pop = map_cells[sample_idx].population;
            }

            // Populate the LiDAR input layer array
            inputs[input_idx] = s_res * vision_multiplier;
            inputs[input_idx + 1u] = s_height;
            inputs[input_idx + 2u] = s_pop * vision_multiplier;
            
            // For secondary attributes (comm/pheno/infra), we stick to global for now to keep LDS simple
            // and because they are less critical for base survival behavior
            let sample_idx_global = clamp(uy * map_w + ux, 0u, max_idx);
            inputs[input_idx + 3u] = map_cells[sample_idx_global].comm1;
            inputs[input_idx + 4u] = map_cells[sample_idx_global].comm2;
            inputs[input_idx + 5u] = map_cells[sample_idx_global].comm3;
            inputs[input_idx + 6u] = map_cells[sample_idx_global].comm4;
            inputs[input_idx + 7u] = map_cells[sample_idx_global].pheno_r;
            inputs[input_idx + 8u] = map_cells[sample_idx_global].pheno_g;
            inputs[input_idx + 9u] = map_cells[sample_idx_global].pheno_b;
            inputs[input_idx + 10u] = (f32(atomicLoad(&map_cells[sample_idx_global].infra_roads)) / 1000.0) / cfg.infra.max_infra * vision_multiplier;
            inputs[input_idx + 11u] = (f32(atomicLoad(&map_cells[sample_idx_global].infra_housing)) / 1000.0) / cfg.infra.max_infra * vision_multiplier;
            inputs[input_idx + 12u] = (f32(atomicLoad(&map_cells[sample_idx_global].infra_farms)) / 1000.0) / cfg.infra.max_infra * vision_multiplier;
            inputs[input_idx + 13u] = (f32(atomicLoad(&map_cells[sample_idx_global].infra_storage)) / 1000.0) / cfg.infra.max_infra * vision_multiplier;
            
            input_idx += 14u;
        }
    }

    var hidden1 = array<f32, 128>();
    for (var h1 = 0u; h1 < 128u; h1 = h1 + 1u) {
        var sum = 0.0;
        let hidden_count = agents[idx].hidden_count;
        if (h1 < hidden_count) {
            for (var k = 0u; k < 8u; k = k + 1u) {
                let in_idx = min(genetics[g_idx].w1_indices[h1 * 8u + k], 183u);
                sum = sum + inputs[in_idx] * genetics[g_idx].w1_weights[h1 * 8u + k];
            }
            hidden1[h1] = tanh(sum);
        } else {
            hidden1[h1] = 0.0;
        }
    }

    var hidden2 = array<f32, 128>();
    for (var h2 = 0u; h2 < 128u; h2 = h2 + 1u) {
        var sum = 0.0;
        let hidden_count = agents[idx].hidden_count;
        if (h2 < hidden_count) {
            for (var h1 = 0u; h1 < 128u; h1 = h1 + 1u) {
                if (h1 < hidden_count) {
                    sum = sum + hidden1[h1] * genetics[g_idx].w2[h1 * 128u + h2];
                }
            }
            hidden2[h2] = tanh(sum);
        } else {
            hidden2[h2] = 0.0;
        }
    }

    var outputs = array<f32, 56>(); 
    for (var o = 0u; o < 56u; o = o + 1u) {
        var sum = 0.0;
        let hidden_count = agents[idx].hidden_count;
        for (var h2 = 0u; h2 < hidden_count; h2 = h2 + 1u) {
            if (h2 < hidden_count) {
                sum = sum + hidden2[h2] * genetics[g_idx].w3[h2 * 56u + o];
            }
        }
        outputs[o] = tanh(sum);
    }

    // --- Maturity Scaling (Newborns to Puberty) ---
    let maturity = clamp(agents[idx].age / cfg.bio.puberty_age, 0.0, 1.0);
    let age_speed_mult = mix(cfg.bio.infant_speed_mult, 1.0, maturity);
    let age_stamina_mult = mix(cfg.bio.infant_stamina_mult, 1.0, maturity);

    let rest_intent = clamp(outputs[5] * 0.5 + 0.5, 0.0, 1.0);
    var resting = rest_intent > 0.5 || agents[idx].stamina <= 0.0;

    var turn_intent = outputs[0];
    if (resting) { turn_intent = 0.0; }
    agents[idx].heading = agents[idx].heading + turn_intent * 0.5;

    var speed_intent = clamp(outputs[1] * 0.5 + 0.5, 0.0, 1.0) * age_speed_mult; 
    
    if (maturity < 1.0) {
        var adult_nearby = false;
        let cx = u32(ax);
        let cy = u32(ay);
        for (var dy = -1i; dy <= 1; dy = dy + 1) {
            for (var dx = -1i; dx <= 1; dx = dx + 1) {
                let sx = u32(i32(cx) + dx) % map_w;
                let sy = u32(i32(cy) + dy) % map_h;
                let s_idx = sy * map_w + sx;
                if (atomicLoad(&map_cells[s_idx].adult_count) > 0) {
                    adult_nearby = true;
                    break;
                }
            }
            if (adult_nearby) { break; }
        }
        if (!adult_nearby) { speed_intent = 0.0; }
    }

    if (resting) { speed_intent = 0.0; }
    
    agents[idx].reproduce_desire = clamp(outputs[3] * 0.5 + 0.5, 0.0, 1.0);
    if (resting) { agents[idx].reproduce_desire = 0.0; }
    
    agents[idx].attack_intent = clamp(outputs[4] * 0.5 + 0.5, 0.0, 1.0);
    if (resting) { agents[idx].attack_intent = 0.0; }
    
    agents[idx].rest_intent = rest_intent;
    
    for (var c = 0u; c < 12u; c = c + 1u) {
        agents[idx].comms[c] = select(clamp(outputs[6 + c], -1.0, 1.0), 0.0, resting);
    }

    let learn_intent = clamp(outputs[18] * 0.5 + 0.5, 0.0, 1.0); 
    for (var m = 0u; m < 24u; m = m + 1u) {
        agents[idx].mems[m] = outputs[19 + m];
    }
    
    agents[idx].buy_intent = clamp(outputs[43] * 0.5 + 0.5, 0.0, 1.0);
    if (resting) { agents[idx].buy_intent = 0.0; }
    
    agents[idx].sell_intent = clamp(outputs[44] * 0.5 + 0.5, 0.0, 1.0);
    if (resting) { agents[idx].sell_intent = 0.0; }
    
    agents[idx].ask_price = abs(outputs[45]) * 10.0;
    agents[idx].bid_price = abs(outputs[46]) * 10.0;
    
    agents[idx].drop_water_intent = clamp(outputs[47] * 0.5 + 0.5, 0.0, 1.0); 
    if (resting) { agents[idx].drop_water_intent = 0.0; }
    
    agents[idx].pickup_water_intent = clamp(outputs[48] * 0.5 + 0.5, 0.0, 1.0); 
    if (resting) { agents[idx].pickup_water_intent = 0.0; }
    
    agents[idx].defend_intent = clamp(outputs[49] * 0.5 + 0.5, 0.0, 1.0);
    if (resting) { agents[idx].defend_intent = 0.0; }

    let e_intent = clamp(outputs[55] * 0.5 + 0.5, 0.0, 1.0);
    agents[idx].emergency_intent = e_intent;
    let is_emergency = e_intent > 0.5;

    let b_road = clamp(outputs[50] * 0.5 + 0.5, 0.0, 1.0);
    let b_house = clamp(outputs[51] * 0.5 + 0.5, 0.0, 1.0);
    let b_farm = clamp(outputs[52] * 0.5 + 0.5, 0.0, 1.0);
    let b_storage = clamp(outputs[53] * 0.5 + 0.5, 0.0, 1.0);

    let max_build_intent = max(b_road, max(b_house, max(b_farm, b_storage)));
    let is_building = max_build_intent > 0.5 && max_build_intent > speed_intent && !is_emergency;

    if (resting || is_emergency || !is_building) {
        agents[idx].build_road_intent = 0.0;
        agents[idx].build_house_intent = 0.0;
        agents[idx].build_farm_intent = 0.0;
        agents[idx].build_storage_intent = 0.0;
    } else {
        agents[idx].build_road_intent = b_road;
        agents[idx].build_house_intent = b_house;
        agents[idx].build_farm_intent = b_farm;
        agents[idx].build_storage_intent = b_storage;
    }
    
    agents[idx].destroy_infra_intent = clamp(outputs[54] * 0.5 + 0.5, 0.0, 1.0);
    if (resting || is_emergency) { agents[idx].destroy_infra_intent = 0.0; }

    var base_speed = speed_intent * cfg.bio.base_speed;
    if (resting || is_building) {
        base_speed = 0.0;
        if (resting) {
            agents[idx].stamina = min(agents[idx].stamina + 2.0, cfg.bio.max_stamina);
        }
    }

    let next_x = ax + cos(agents[idx].heading) * base_speed * lon_scale;
    let next_y = ay + sin(agents[idx].heading) * base_speed;

    var wrap_nx = next_x; 
    var wrap_ny = next_y;
    if (wrap_ny < 0.0) { wrap_ny = -wrap_ny; wrap_nx = wrap_nx + map_w_f32 / 2.0; }
    else if (wrap_ny >= map_h_f32) { wrap_ny = 2.0 * map_h_f32 - 1.0 - wrap_ny; wrap_nx = wrap_nx + map_w_f32 / 2.0; }
    wrap_nx = wrap_nx % map_w_f32; if (wrap_nx < 0.0) { wrap_nx = wrap_nx + map_w_f32; }
    let next_idx = clamp(u32(wrap_ny) * map_w + u32(wrap_nx), 0u, max_idx);
    let next_height = map_heights[next_idx];

    let slope = next_height - current_height;
    let height_multiplier = 1.0 - clamp(slope * cfg.eco.climb_penalty, -0.5, 0.9);
    
    let total_weight = (agents[idx].food / 1000.0) + agents[idx].water;
    let encumbrance_mult = clamp(1.0 - (total_weight / cfg.bio.max_carry_weight), 0.05, 1.0);
    let crowding_mult = clamp(1.0 - (local_population / cfg.combat.crowding_threshold), 0.1, 1.0);
    let road_mult = 1.0 + (local_infra_roads / cfg.infra.max_infra) * cfg.infra.road_speed_bonus; 
    
    var actual_speed = base_speed * height_multiplier * encumbrance_mult * crowding_mult * road_mult;

    if (agents[idx].gestation_timer > 0.0) {
        agents[idx].gestation_timer = agents[idx].gestation_timer - 1.0;
        if (agents[idx].gestation_timer <= 0.0) {
            agents[idx].is_pregnant = 0.0;
        }
    }

    var preg_mult = 1.0;
    if (agents[idx].is_pregnant > 0.5) {
        actual_speed = actual_speed * cfg.combat.pregnancy_speed_mult; 
        preg_mult = cfg.combat.pregnancy_cost_mult; 
    }
    
    if (!resting) {
        agents[idx].stamina = agents[idx].stamina - (actual_speed * 0.05 * (2.0 - age_stamina_mult));
        if (agents[idx].stamina <= 0.0) {
            agents[idx].stamina = 0.0;
            resting = true;
        }
    }

    let act_spd = select(actual_speed, 0.0, resting);
    let final_next_x = ax + cos(agents[idx].heading) * act_spd * lon_scale;
    let final_next_y = ay + sin(agents[idx].heading) * act_spd;
    
    var final_wrap_x = final_next_x; 
    var final_wrap_y = final_next_y;
    var final_heading = agents[idx].heading;
    
    if (final_wrap_y < 0.0) { final_wrap_y = -final_wrap_y; final_wrap_x = final_wrap_x + map_w_f32 / 2.0; final_heading = final_heading + 3.14159265; }
    else if (final_wrap_y >= map_h_f32) { final_wrap_y = 2.0 * map_h_f32 - 1.0 - final_wrap_y; final_wrap_x = final_wrap_x + map_w_f32 / 2.0; final_heading = final_heading + 3.14159265; }
    final_wrap_x = final_wrap_x % map_w_f32; if (final_wrap_x < 0.0) { final_wrap_x = final_wrap_x + map_w_f32; }

    let final_idx = clamp(u32(final_wrap_y) * map_w + u32(final_wrap_x), 0u, max_idx);
    let final_height = map_heights[final_idx];

    let caloric_maturity = clamp(agents[idx].age / cfg.bio.puberty_age, 0.2, 1.0);
    var rest_mult = 1.0; 
    if (resting) { rest_mult = 0.5 - ((local_infra_housing / cfg.infra.max_infra) * cfg.infra.housing_rest_bonus); } 
    var defend_mult = 1.0; if (agents[idx].defend_intent > 0.5 && !resting) { defend_mult = cfg.combat.defend_cost_mult; }
    let carrying_effort = 1.0 + (total_weight / 50.0);
    var intent_exertion = 0.0;
    if (!resting) {
        if (agents[idx].build_road_intent > 0.5) { intent_exertion = intent_exertion + 0.15; }
        if (agents[idx].build_house_intent > 0.5) { intent_exertion = intent_exertion + 0.15; }
        if (agents[idx].build_farm_intent > 0.5) { intent_exertion = intent_exertion + 0.15; }
        if (agents[idx].build_storage_intent > 0.5) { intent_exertion = intent_exertion + 0.15; }
        if (agents[idx].destroy_infra_intent > 0.5) { intent_exertion = intent_exertion + 0.25; }
    }
    
    let metabolic_rate = cfg.eco.baseline_cost * caloric_maturity * rest_mult * preg_mult * defend_mult * carrying_effort * (1.0 + intent_exertion);

    agents[idx].water = agents[idx].water - metabolic_rate;
    let cold_penalty = max(0.0, -effective_temp) * 0.1;
    agents[idx].food = agents[idx].food - (metabolic_rate * 1000.0) - (cold_penalty * 1000.0);
    
    let storage_mult = 1.0 - clamp((local_infra_storage / cfg.infra.max_infra) * cfg.infra.storage_rot_reduction, 0.0, cfg.infra.storage_rot_reduction); 
    let spoilage_multiplier = max(0.1, 1.0 + local_temp);
    let spoilage_rate = cfg.eco.base_spoilage_rate * spoilage_multiplier * storage_mult;
    agents[idx].food = agents[idx].food - (agents[idx].food * spoilage_rate);
    
    if (agents[idx].food < 0.0) { agents[idx].food = 0.0; agents[idx].health = agents[idx].health - cfg.bio.starvation_rate; }
    if (agents[idx].water < 0.0) { agents[idx].water = 0.0; agents[idx].health = agents[idx].health - cfg.bio.starvation_rate; }
    if (agents[idx].food > (metabolic_rate * 1000.0) && agents[idx].water > metabolic_rate && agents[idx].health < cfg.bio.max_health) {
        agents[idx].health = min(agents[idx].health + (cfg.bio.starvation_rate * 2.0), cfg.bio.max_health);
        agents[idx].food = agents[idx].food - (metabolic_rate * 1000.0); 
        agents[idx].water = agents[idx].water - metabolic_rate; 
    }

    if (current_height >= 0.0) {
        var drop_desire = outputs[2];
        if (resting) { drop_desire = 0.0; }
        if (local_avg_aggression > 0.5 && agents[idx].defend_intent < 0.5) {
            agents[idx].health = agents[idx].health - cfg.combat.bystander_damage;
        }
        
        if (agents[idx].attack_intent > 0.5 && local_population > 1.0 && !resting && agents[idx].age > cfg.bio.puberty_age) {
            let steal_amount = min((local_population - 1.0) * 0.5, cfg.combat.steal_amount);
            if (agents[idx].defend_intent < 0.5) {
                agents[idx].wealth = min(agents[idx].wealth + steal_amount * 5.0, cfg.eco.boat_cost); 
                if (local_avg_aggression > 0.5) { agents[idx].health = agents[idx].health - cfg.combat.attacker_damage; }
            }
        } else if (agents[idx].drop_water_intent > 0.5 && agents[idx].water > cfg.eco.water_transfer_amount) {
            let transfer_amount = min(agents[idx].water, cfg.eco.water_transfer_amount);
            agents[idx].water = agents[idx].water - transfer_amount;
            atomicAdd(&map_cells[safe_current_idx].market_water, i32(transfer_amount * 1000.0));
            atomicMin(&map_cells[safe_current_idx].market_water, i32(cfg.world.max_tile_water * 1000.0));
        } else if (agents[idx].pickup_water_intent > 0.5 && local_market_water > 0.0 && agents[idx].water < cfg.bio.max_water) {
            let transfer_amount = min(cfg.eco.water_transfer_amount, local_market_water);
            let actual_transfer = min(transfer_amount, cfg.bio.max_water - agents[idx].water);
            agents[idx].water = agents[idx].water + actual_transfer;
            atomicSub(&map_cells[safe_current_idx].market_water, i32(actual_transfer * 1000.0));
            atomicMax(&map_cells[safe_current_idx].market_water, 0);
        } else if (current_height > 0.0 && current_height <= 0.05 && agents[idx].water < cfg.bio.max_water) {
            agents[idx].water = min(agents[idx].water + cfg.eco.water_transfer_amount, cfg.bio.max_water);
        } else if (drop_desire > 0.5 && agents[idx].food > (cfg.eco.drop_amount * 1000.0 + 100000.0)) {
            agents[idx].food = agents[idx].food - (cfg.eco.drop_amount * 1000.0);
            atomicAdd(&map_cells[safe_current_idx].res_value, i32(cfg.eco.drop_amount * 1000.0));
            atomicMin(&map_cells[safe_current_idx].res_value, i32(cfg.world.max_tile_resource * 1000.0));
        } else if (!resting) {
            if (local_res_value > 0.1) {
                let max_mult = (cfg.eco.max_gather_rate / cfg.eco.base_gather_rate) - 1.0;
                let tool_multiplier = 1.0 + min(sqrt(agents[idx].food / 1000.0) * 0.1, max_mult);
                let gathered = min(cfg.eco.base_gather_rate * tool_multiplier * maturity, local_res_value);
                agents[idx].food = agents[idx].food + (gathered * 1000.0);
                atomicSub(&map_cells[safe_current_idx].res_value, i32(gathered * 1000.0));
                atomicMax(&map_cells[safe_current_idx].res_value, 0);
            }
        }
        
        let build_ticks = max(1.0, cfg.infra.build_ticks);
        let build_amount = i32(1000.0 / build_ticks);
        let cost_amount = cfg.infra.infra_cost / build_ticks;

        if (!resting && agents[idx].wealth >= cost_amount) {
            let has_other_than_road = local_infra_housing > 0.0 || local_infra_farms > 0.0 || local_infra_storage > 0.0;
            let has_other_than_house = local_infra_roads > 0.0 || local_infra_farms > 0.0 || local_infra_storage > 0.0;
            let has_other_than_farm = local_infra_roads > 0.0 || local_infra_housing > 0.0 || local_infra_storage > 0.0;
            let has_other_than_storage = local_infra_roads > 0.0 || local_infra_housing > 0.0 || local_infra_farms > 0.0;

            if (agents[idx].build_road_intent > 0.5 && local_infra_roads < cfg.infra.max_infra && !has_other_than_road) {
                agents[idx].wealth = agents[idx].wealth - cost_amount; 
                atomicAdd(&map_cells[safe_current_idx].infra_roads, build_amount);
                atomicMin(&map_cells[safe_current_idx].infra_roads, i32(cfg.infra.max_infra * 1000.0));
            } else if (agents[idx].build_house_intent > 0.5 && local_infra_housing < cfg.infra.max_infra && !has_other_than_house) {
                agents[idx].wealth = agents[idx].wealth - cost_amount; 
                atomicAdd(&map_cells[safe_current_idx].infra_housing, build_amount);
                atomicMin(&map_cells[safe_current_idx].infra_housing, i32(cfg.infra.max_infra * 1000.0));
            } else if (agents[idx].build_farm_intent > 0.5 && local_infra_farms < cfg.infra.max_infra && !has_other_than_farm) {
                agents[idx].wealth = agents[idx].wealth - cost_amount; 
                atomicAdd(&map_cells[safe_current_idx].infra_farms, build_amount);
                atomicMin(&map_cells[safe_current_idx].infra_farms, i32(cfg.infra.max_infra * 1000.0));
            } else if (agents[idx].build_storage_intent > 0.5 && local_infra_storage < cfg.infra.max_infra && !has_other_than_storage) {
                agents[idx].wealth = agents[idx].wealth - cost_amount; 
                atomicAdd(&map_cells[safe_current_idx].infra_storage, build_amount);
                atomicMin(&map_cells[safe_current_idx].infra_storage, i32(cfg.infra.max_infra * 1000.0));
            }
        }
        
        if (!resting && agents[idx].destroy_infra_intent > 0.5) {
            let destroy_cost = (cfg.infra.infra_cost * 0.5) / build_ticks;
            if (agents[idx].wealth >= destroy_cost) {
                var destroyed = false;
                if (local_infra_roads > 0.0) { atomicSub(&map_cells[safe_current_idx].infra_roads, build_amount); atomicMax(&map_cells[safe_current_idx].infra_roads, 0); destroyed = true; }
                else if (local_infra_housing > 0.0) { atomicSub(&map_cells[safe_current_idx].infra_housing, build_amount); atomicMax(&map_cells[safe_current_idx].infra_housing, 0); destroyed = true; }
                else if (local_infra_farms > 0.0) { atomicSub(&map_cells[safe_current_idx].infra_farms, build_amount); atomicMax(&map_cells[safe_current_idx].infra_farms, 0); destroyed = true; }
                else if (local_infra_storage > 0.0) { atomicSub(&map_cells[safe_current_idx].infra_storage, build_amount); atomicMax(&map_cells[safe_current_idx].infra_storage, 0); destroyed = true; }
                if (destroyed) { agents[idx].wealth = agents[idx].wealth - destroy_cost; }
            }
        }
        
        map_cells[safe_current_idx].avg_ask = mix(local_avg_ask, agents[idx].ask_price, 0.1);
        map_cells[safe_current_idx].avg_bid = mix(local_avg_bid, agents[idx].bid_price, 0.1);
        
        let local_market_food = f32(atomicLoad(&map_cells[safe_current_idx].market_food)) / 1000.0;
        let local_market_wealth = f32(atomicLoad(&map_cells[safe_current_idx].market_wealth)) / 1000.0;
        
        if (agents[idx].buy_intent > 0.5 && agents[idx].wealth >= local_avg_ask && local_market_food >= 1000.0 && local_avg_ask > 0.0) {
            agents[idx].wealth = agents[idx].wealth - local_avg_ask;
            agents[idx].food = agents[idx].food + 1000.0;
            atomicAdd(&map_cells[safe_current_idx].market_wealth, i32(local_avg_ask * 1000.0));
            atomicSub(&map_cells[safe_current_idx].market_food, 1000 * 1000);
        } else if (agents[idx].sell_intent > 0.5 && agents[idx].food >= 1000.0 && local_market_wealth >= local_avg_bid && local_avg_bid > 0.0) {
            agents[idx].food = agents[idx].food - 1000.0;
            agents[idx].wealth = agents[idx].wealth + local_avg_bid;
            atomicAdd(&map_cells[safe_current_idx].market_food, 1000 * 1000);
            atomicSub(&map_cells[safe_current_idx].market_wealth, i32(local_avg_bid * 1000.0));
        }
        
        let elevation_mult = clamp(1.0 - (current_height * 2.0), 0.05, 1.0);
        let moisture_mult = clamp(0.05 + (local_market_water / 100.0), 0.05, 5.0);
        var biome_mult = 1.0;
        let m = map_cells[safe_current_idx].base_moisture;
        if (base_temp < -0.4) { biome_mult = 0.01; } // Snow
        else if (base_temp < -0.2) { biome_mult = 0.05; } // Tundra
        else if (m < -0.3) { biome_mult = 0.02; } // Desert
        else if (m < 0.0) { biome_mult = 0.4; } // Savanna
        else if (m > 0.3) { biome_mult = 1.5; } // Jungle
        let farm_mult = 1.0 + (local_infra_farms / cfg.infra.max_infra) * 5.0; 
        let biome_regen_rate = cfg.world.regen_rate * elevation_mult * moisture_mult * farm_mult * biome_mult;
        atomicAdd(&map_cells[safe_current_idx].res_value, i32(biome_regen_rate * 1000.0));
        atomicMin(&map_cells[safe_current_idx].res_value, i32(cfg.world.max_tile_resource * 1000.0));
        
        let rand_r = fract(pseudo_rand * 1.234);
        if (local_infra_roads > 0.0 && rand_r < cfg.infra.decay_rate_roads) { atomicSub(&map_cells[safe_current_idx].infra_roads, 1000); }
        let rand_h = fract(pseudo_rand * 2.345);
        if (local_infra_housing > 0.0 && rand_h < cfg.infra.decay_rate_housing) { atomicSub(&map_cells[safe_current_idx].infra_housing, 1000); }
        let rand_f = fract(pseudo_rand * 3.456);
        if (local_infra_farms > 0.0 && rand_f < cfg.infra.decay_rate_farms) { atomicSub(&map_cells[safe_current_idx].infra_farms, 1000); }
        let rand_s = fract(pseudo_rand * 4.567);
        if (local_infra_storage > 0.0 && rand_s < cfg.infra.decay_rate_storage) { atomicSub(&map_cells[safe_current_idx].infra_storage, 1000); }
        if (pseudo_rand < 0.05) {
            let cur_m_f = f32(atomicLoad(&map_cells[safe_current_idx].market_food));
            if (cur_m_f > 0.0) { atomicSub(&map_cells[safe_current_idx].market_food, i32(cur_m_f * 0.02 * storage_mult)); }
        }
        if (current_height <= 0.05) { atomicMax(&map_cells[safe_current_idx].market_water, i32(cfg.world.max_tile_water * 1000.0)); }
        if (agents[idx].age >= cfg.bio.puberty_age) { atomicAdd(&map_cells[safe_current_idx].adult_count, 1); }

        map_cells[safe_current_idx].population = local_population * 0.99 + 1.0;
        map_cells[safe_current_idx].avg_speed = mix(local_avg_speed, speed_intent, 0.1);
        map_cells[safe_current_idx].avg_share = mix(local_avg_share, outputs[2], 0.1);
        map_cells[safe_current_idx].avg_reproduce = mix(local_avg_reproduce, agents[idx].reproduce_desire, 0.1);
        map_cells[safe_current_idx].avg_aggression = mix(local_avg_aggression, agents[idx].attack_intent, 0.1);
        map_cells[safe_current_idx].avg_pregnancy = mix(local_avg_pregnancy, agents[idx].is_pregnant, 0.1);
        map_cells[safe_current_idx].avg_turn = mix(local_avg_turn, turn_intent, 0.1);
        map_cells[safe_current_idx].avg_rest = mix(local_avg_rest, rest_intent, 0.1);
        map_cells[safe_current_idx].comm1 = mix(local_comm1, agents[idx].comms[0], 0.1);
        map_cells[safe_current_idx].comm2 = mix(local_comm2, agents[idx].comms[1], 0.1);
        map_cells[safe_current_idx].comm3 = mix(local_comm3, agents[idx].comms[2], 0.1);
        map_cells[safe_current_idx].comm4 = mix(local_comm4, agents[idx].comms[3], 0.1);
        map_cells[safe_current_idx].pheno_r = mix(local_pheno_r, agents[idx].pheno_r, 0.1);
        map_cells[safe_current_idx].pheno_g = mix(local_pheno_g, agents[idx].pheno_g, 0.1);
        map_cells[safe_current_idx].pheno_b = mix(local_pheno_b, agents[idx].pheno_b, 0.1);
    }

    let climb_penalty_mult = max(0.0, slope) * cfg.eco.climb_penalty; 
    let weight_move_penalty = total_weight / 20.0;
    let move_cost = ((act_spd * cfg.eco.move_cost_per_unit) * (1.0 + climb_penalty_mult + weight_move_penalty) * 1000.0) / road_mult;
    let is_water = final_height < 0.0;
    let can_enter_water = agents[idx].wealth >= cfg.eco.boat_cost || (current_height < 0.0 && agents[idx].wealth > 0.0);

    if (is_water && !can_enter_water) {
        agents[idx].heading = agents[idx].heading + 3.14159; 
        agents[idx].speed = 0.0;
    } else {
        if (agents[idx].food >= move_cost) {
            agents[idx].x = final_wrap_x;
            agents[idx].y = final_wrap_y;
            agents[idx].heading = final_heading;
            agents[idx].food = agents[idx].food - move_cost;
            agents[idx].speed = act_spd;
        } else {
            agents[idx].speed = 0.0; 
        }
    }
    
    if (agents[idx].age >= cfg.bio.max_age) {
        let ticks_per_year = 525600.0 / cfg.world.tick_to_mins;
        let years_past_max = (agents[idx].age - cfg.bio.max_age) / ticks_per_year;
        let biological_decay = cfg.bio.starvation_rate * (2.0 + pow(years_past_max, 1.5));
        var applied_decay = biological_decay;
        let healthcare_cost = biological_decay * 50.0;
        if (agents[idx].wealth >= healthcare_cost) {
            agents[idx].wealth = agents[idx].wealth - healthcare_cost;
            applied_decay = biological_decay * 0.5;
        }
        agents[idx].health = agents[idx].health - applied_decay;
    }

    let health_delta = agents[idx].health - prev_health;
    let food_delta = agents[idx].food - prev_food;
    let wealth_delta = agents[idx].wealth - prev_wealth;
    let dopamine_signal = (health_delta * 20.0) + (food_delta / 10000.0) + (wealth_delta * 0.5);

    if (abs(dopamine_signal) > 0.01 && learn_intent > 0.1) {
        let lr = dopamine_signal * learn_intent * 0.0005; 
        let hidden_count = agents[idx].hidden_count;
        for (var h1 = 0u; h1 < hidden_count; h1 = h1 + 1u) {
            for (var k = 0u; k < 8u; k = k + 1u) {
                let in_idx = min(genetics[g_idx].w1_indices[h1 * 8u + k], 183u);
                genetics[g_idx].w1_weights[h1 * 8u + k] = clamp(genetics[g_idx].w1_weights[h1 * 8u + k] + lr * inputs[in_idx] * hidden1[h1], -2.0, 2.0);
            }
        }
        for (var h2 = 0u; h2 < hidden_count; h2 = h2 + 1u) {
            for (var h1 = 0u; h1 < hidden_count; h1 = h1 + 1u) {
                genetics[g_idx].w2[h1 * 128u + h2] = clamp(genetics[g_idx].w2[h1 * 128u + h2] + lr * hidden1[h1] * hidden2[h2], -2.0, 2.0);
            }
        }
        for (var o = 0u; o < 56u; o = o + 1u) {
            for (var h2 = 0u; h2 < hidden_count; h2 = h2 + 1u) {
                genetics[g_idx].w3[h2 * 56u + o] = clamp(genetics[g_idx].w3[h2 * 56u + o] + lr * hidden2[h2] * outputs[o], -2.0, 2.0);
            }
        }
    }
}

@compute @workgroup_size(16, 16)
fn render_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    if (x >= cfg.world.map_width || y >= cfg.world.map_height) { return; }
    
    let idx = y * cfg.world.map_width + x;
    
    // Clear adult count every frame before tracing
    atomicStore(&map_cells[idx].adult_count, 0);

    let height = map_heights[idx];
    var r = 0u; var g = 0u; var b = 0u;
    let mode = cfg.sim.visual_mode;
    
    if (mode == 1u) { // Resources
        let val = f32(atomicLoad(&map_cells[idx].res_value)) / 1000.0;
        let max_ln = log(cfg.world.max_tile_resource + 1.0);
        if (val > 0.0) {
            let ratio = clamp(log(val + 1.0) / max_ln, 0.0, 1.0);
            r = u32((1.0 - ratio) * 255.0); g = u32(ratio * 255.0);
        } else { 
            if (height <= 0.0) { r = 10u; g = 20u; b = 50u; } else { r = 25u; g = 25u; b = 25u; }
        }
    } else if (mode >= 5u && mode <= 9u) {
        if (mode == 9u) { 
            let ir = f32(atomicLoad(&map_cells[idx].infra_roads)) / 1000.0;
            let ih = f32(atomicLoad(&map_cells[idx].infra_housing)) / 1000.0;
            let i_f = f32(atomicLoad(&map_cells[idx].infra_farms)) / 1000.0;
            let i_s = f32(atomicLoad(&map_cells[idx].infra_storage)) / 1000.0;
            let max_val = max(max(ir, ih), max(i_f, i_s));
            if (max_val > 0.0) {
                let max_ln = log(cfg.infra.max_infra + 1.0);
                let ratio = clamp(log(max_val + 1.0) / max_ln, 0.0, 1.0);
                if (ir == max_val) { r = u32(ratio * 200.0); g = u32(ratio * 200.0); b = u32(ratio * 200.0); } // Gray Road
                else if (ih == max_val) { r = u32(ratio * 255.0); g = u32(ratio * 140.0); b = 0u; } // Orange House
                else if (i_f == max_val) { r = u32(ratio * 50.0); g = u32(ratio * 200.0); b = u32(ratio * 50.0); } // Green Farm
                else { r = u32(ratio * 200.0); g = u32(ratio * 50.0); b = u32(ratio * 200.0); } // Purple Granary
            } else { 
                if (height <= 0.0) { r = 10u; g = 20u; b = 50u; } else { r = 25u; g = 25u; b = 25u; }
            }
        } else {
            var val = 0.0;
            var max_ln = 1.0;
            if (mode == 5u) { val = f32(atomicLoad(&map_cells[idx].market_wealth)) / 1000.0; max_ln = log(cfg.world.max_tile_resource + 1.0); }
            else if (mode == 6u) { val = f32(atomicLoad(&map_cells[idx].market_food)) / 1000000.0; max_ln = log(cfg.world.max_tile_resource + 1.0); }
            else if (mode == 7u) { val = map_cells[idx].avg_ask; max_ln = log(11.0); }
            else if (mode == 8u) { val = map_cells[idx].avg_bid; max_ln = log(11.0); }
            
            if (val > 0.0) {
                let ratio = clamp(log(val + 1.0) / max_ln, 0.0, 1.0);
                r = u32((1.0 - ratio) * 255.0); g = u32(ratio * 255.0);
            } else { 
                if (height <= 0.0) { r = 10u; g = 20u; b = 50u; } else { r = 25u; g = 25u; b = 25u; }
            }
        }
    } else if (mode == 10u) { // Temperature
        let dist_from_equator = abs(f32(y) - f32(cfg.world.map_height) / 2.0) / (f32(cfg.world.map_height) / 2.0);
        let base_temp = (1.0 - dist_from_equator * 2.0) - max(0.0, height * 2.0);
        let ticks_per_day = 24.0 * 60.0 / cfg.world.tick_to_mins;
        let day_cycle = (f32(cfg.sim.current_tick) % ticks_per_day) / ticks_per_day;
        let day_intensity = sin(day_cycle * 6.28318 - 1.5708) * 0.5 + 0.5;
        let season_time = f32(cfg.sim.current_tick) / 100000.0 * 6.28318;
        let local_temp = base_temp + sin(season_time) * 0.5 - (1.0 - day_intensity) * 0.3;
        let norm = clamp((local_temp + 1.0) / 2.0, 0.0, 1.0);
        r = u32(norm * 255.0); g = 50u; b = u32((1.0 - norm) * 255.0);
    } else if (mode == 12u) { // Tribes / Phenotypes
        let pr = clamp(map_cells[idx].pheno_r * 0.5 + 0.5, 0.0, 1.0);
        let pg = clamp(map_cells[idx].pheno_g * 0.5 + 0.5, 0.0, 1.0);
        let pb = clamp(map_cells[idx].pheno_b * 0.5 + 0.5, 0.0, 1.0);
        let height = map_heights[idx];
        var mult = 1.0; if (height < 0.0) { mult = 0.3; } // Dim ocean water slightly
        r = u32(pr * mult * 255.0);
        g = u32(pg * mult * 255.0);
        b = u32(pb * mult * 255.0);
    } else if (mode == 13u) { // Drinkable Water
        if (height <= 0.0) {
            r = 10u; g = 20u; b = 50u; // Saltwater ocean
        } else {
            let val = f32(atomicLoad(&map_cells[idx].market_water)) / 1000.0;
            if (val > 0.0) {
                let max_ln = log(cfg.world.max_tile_water + 1.0);
                let ratio = clamp(log(val + 1.0) / max_ln, 0.0, 1.0);
                r = 0u; g = u32(ratio * 200.0); b = u32(55.0 + ratio * 200.0);
            } else { r = 40u; g = 40u; b = 40u; } // Dry Land
        }
    } else { // Default / Age / Gender / DayNight
        var base_r = 0u; var base_g = 0u; var base_b = 0u;
        
        // Continental Biome Logic
        if (height < -0.2) { base_r = 10u; base_g = 40u; base_b = 120u; } // Deep Ocean
        else if (height < 0.0) { base_r = 30u; base_g = 80u; base_b = 180u; } // Shallow Ocean
        else {
            let dist_from_equator = abs(f32(y) - f32(cfg.world.map_height) / 2.0) / (f32(cfg.world.map_height) / 2.0);
            let base_temp = (1.0 - dist_from_equator * 2.0) - max(0.0, height * 2.0);
            let m = map_cells[idx].base_moisture;
            
            if (base_temp < -0.4) { base_r = 240u; base_g = 240u; base_b = 255u; } // Snow
            else if (base_temp < -0.2) { base_r = 120u; base_g = 150u; base_b = 140u; } // Tundra / Taiga
            else if (m < -0.3) { base_r = 210u; base_g = 190u; base_b = 130u; } // Desert / Sand
            else if (m < 0.0) { base_r = 160u; base_g = 160u; base_b = 90u; } // Savanna / Scrub
            else if (m > 0.3) { base_r = 20u; base_g = 100u; base_b = 30u; } // Jungle
            else { base_r = 40u; base_g = 140u; base_b = 50u; } // Forest / Grassland
        }
        
        // 2.5D Directional Topography Shading
        var lx = x - 1u; if (x == 0u) { lx = cfg.world.map_width - 1u; }
        var ty = y; if (y > 0u) { ty = y - 1u; } // Don't shadow off the edge of the pole
        let hx = map_heights[y * cfg.world.map_width + lx];
        let hy = map_heights[ty * cfg.world.map_width + x];
        
        let dx = height - hx;
        let dy = height - hy;
        
        // Calculate directional sunlight multiplier
        var shade = clamp(1.0 + (dx + dy) * 15.0, 0.3, 1.8);
        if (height < 0.0) { shade = clamp(1.0 + (dx + dy) * 2.0, 0.8, 1.2); } // Water reflects softer
        
        r = u32(clamp(f32(base_r) * shade, 0.0, 255.0));
        g = u32(clamp(f32(base_g) * shade, 0.0, 255.0));
        b = u32(clamp(f32(base_b) * shade, 0.0, 255.0));
        
        // Add subtle topological contour lines
        if (height >= 0.0 && abs(height % 0.1) < 0.01) { r = u32(f32(r) * 0.6); g = u32(f32(g) * 0.6); b = u32(f32(b) * 0.6); }
        
        if (mode == 11u) {
            let ticks_per_day = 24.0 * 60.0 / cfg.world.tick_to_mins;
            let day_intensity = sin(((f32(cfg.sim.current_tick) % ticks_per_day) / ticks_per_day) * 6.28318 - 1.5708) * 0.5 + 0.5;
            let intensity = max(0.15, pow(day_intensity, 0.7));
            r = u32(f32(r) * intensity); g = u32(f32(g) * intensity); b = u32(f32(b) * intensity);
        }
    }
    
    // Encode Little Endian packed RGBA bytes for fast zero-copy memory translation on CPU
    render_buffer[idx] = r | (g << 8u) | (b << 16u) | (255u << 24u);
}