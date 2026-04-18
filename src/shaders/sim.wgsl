// Fast Local Memory Cache for workgroup map tiles (e.g., 10x10 patch + padding)
var<workgroup> lds_res: array<f32, 256>; // 16x16 patch
var<workgroup> lds_height: array<f32, 256>;
var<workgroup> lds_pop: array<f32, 256>;
var<workgroup> lds_base_x: u32;
var<workgroup> lds_base_y: u32;

fn fast_tanh(x: f32) -> f32 {
    let x2 = x * x;
    return x * (27.0 + x2) / (27.0 + 9.0 * x2);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_index) local_idx: u32) {
    let idx = global_id.x;
    if (idx >= arrayLength(&agents)) { return; }

    if (agents[idx].health <= 0.0) { return; }
    let g_idx = agents[idx].genetics_index;

    // Cache position and heading in registers early
    let ax = agents[idx].x;
    let ay = agents[idx].y;
    let ah = agents[idx].heading;

    // Store pre-update state
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
    if (local_idx == 0u) {
        lds_base_x = u32(ax) / 8u * 8u;
        lds_base_y = u32(ay) / 8u * 8u;
    }
    workgroupBarrier();

    // Loop unrolling for LDS loading
    let l_idx_base = local_idx * 4u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let l_idx = l_idx_base + i;
        let lx = l_idx % 16u;
        let ly = l_idx / 16u;
        let gx = (lds_base_x + lx) % map_w;
        let gy = clamp(lds_base_y + ly, 0u, map_h - 1u);
        let cell_idx = gy * map_w + gx;

        lds_res[l_idx] = f32(atomicLoad(&map_cells[cell_idx].res_value)) / 1000.0;
        lds_height[l_idx] = map_heights[cell_idx];
        lds_pop[l_idx] = map_cells[cell_idx].population;
    }
    workgroupBarrier(); 

    let current_idx = u32(ay) * map_w + u32(ax);
    let safe_current_idx = clamp(current_idx, 0u, max_idx);
    
    // Load local attributes once
    let cell_ptr = &map_cells[safe_current_idx];
    let local_population = (*cell_ptr).population;
    let local_avg_speed = (*cell_ptr).avg_speed;
    let local_avg_share = (*cell_ptr).avg_share;
    let local_avg_reproduce = (*cell_ptr).avg_reproduce;
    let local_avg_aggression = (*cell_ptr).avg_aggression;
    let local_avg_pregnancy = (*cell_ptr).avg_pregnancy;
    let local_avg_turn = (*cell_ptr).avg_turn;
    let local_avg_rest = (*cell_ptr).avg_rest;
    let local_avg_ask = (*cell_ptr).avg_ask;
    let local_avg_bid = (*cell_ptr).avg_bid;

    let local_res_value = f32(atomicLoad(&(*cell_ptr).res_value)) / 1000.0;
    let local_infra_roads = f32(atomicLoad(&(*cell_ptr).infra_roads)) / 1000.0;
    let local_infra_housing = f32(atomicLoad(&(*cell_ptr).infra_housing)) / 1000.0;
    
    let pseudo_rand = fract(sin(dot(vec2<f32>(ax, ay), vec2<f32>(12.9898, 78.233))) * 43758.5453);

    // --- Day/Night Cycle ---
    let ticks_per_day = 24.0 * 60.0 / cfg.world.tick_to_mins;
    let day_cycle_progress = (f32(cfg.sim.current_tick) % ticks_per_day) / ticks_per_day; 
    let day_intensity = sin(day_cycle_progress * 6.28318 - 1.5708) * 0.5 + 0.5; 

    let season_time = f32(cfg.sim.current_tick) / 100000.0 * 6.28318;
    let season_sine = sin(season_time);
    let dist_from_equator = abs(ay - map_h_f32 / 2.0) / (map_h_f32 / 2.0);
    let base_temp = (1.0 - dist_from_equator * 2.0) - max(0.0, map_heights[safe_current_idx] * 2.0);
    let local_temp = base_temp + season_sine * 0.5 - (1.0 - day_intensity) * 0.3; 
    let effective_temp = local_temp + (local_infra_housing / cfg.infra.max_infra) * 2.0;

    let agent_lat = (abs(ay - map_h_f32 / 2.0) / (map_h_f32 / 2.0)) * 1.570796; 
    let lon_scale = 1.0 / max(cos(agent_lat), 0.15); 

    var vis_mult = 1.0;
    if (agents[idx].rest_intent > 0.5 || agents[idx].stamina <= 0.0) { vis_mult = 0.1; }
    let vision_multiplier = (0.3 + day_intensity * 0.7) * vis_mult; 

    // 1. Neural Net Processing
    var inputs = array<f32, 184>();
    inputs[0] = 1.0;
    inputs[1] = local_res_value / 1000.0;
    inputs[2] = local_population;
    inputs[3] = local_avg_speed;
    inputs[4] = local_avg_share;
    inputs[5] = local_avg_reproduce;
    inputs[6] = local_avg_aggression;
    inputs[7] = local_avg_pregnancy;
    inputs[8] = local_avg_turn;
    inputs[9] = local_avg_rest;
    
    let comms_ptr = &map_cells[safe_current_idx];
    inputs[10] = (*comms_ptr).comm1;
    inputs[11] = (*comms_ptr).comm2;
    inputs[12] = (*comms_ptr).comm3;
    inputs[13] = (*comms_ptr).comm4;
    inputs[14] = (*comms_ptr).comm5;
    inputs[15] = (*comms_ptr).comm6;
    inputs[16] = (*comms_ptr).comm7;
    inputs[17] = (*comms_ptr).comm8;
    inputs[18] = (*comms_ptr).comm9;
    inputs[19] = (*comms_ptr).comm10;
    inputs[20] = (*comms_ptr).comm11;
    inputs[21] = (*comms_ptr).comm12;

    inputs[22] = agents[idx].health / cfg.bio.max_health;
    inputs[23] = clamp((agents[idx].food / 1000.0) / cfg.eco.boat_cost, 0.0, 2.0); 
    inputs[24] = agents[idx].water / cfg.bio.max_water;
    inputs[25] = agents[idx].stamina / cfg.bio.max_stamina;
    inputs[26] = agents[idx].age / cfg.bio.max_age;
    inputs[27] = agents[idx].gender;
    inputs[28] = local_temp;
    inputs[29] = season_sine;
    inputs[30] = agents[idx].is_pregnant; 
    inputs[31] = clamp(((agents[idx].food / 1000.0) + agents[idx].water) / cfg.bio.max_carry_weight, 0.0, 2.0);
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
    inputs[64] = (*cell_ptr).pheno_r;
    inputs[65] = (*cell_ptr).pheno_g;
    inputs[66] = (*cell_ptr).pheno_b;
    
    inputs[179] = local_infra_roads / cfg.infra.max_infra;
    inputs[180] = local_infra_housing / cfg.infra.max_infra;
    inputs[181] = f32(atomicLoad(&(*cell_ptr).infra_farms)) / 1000.0 / cfg.infra.max_infra;
    inputs[182] = f32(atomicLoad(&(*cell_ptr).infra_storage)) / 1000.0 / cfg.infra.max_infra;
    inputs[183] = 0.0;

    var input_idx = 67u;
    let cos_h = cos(ah);
    let sin_h = sin(ah);
    let view_spacing = 8.0; 

    // Unroll vision loop 3x3
    for (var ly_i = 1; ly_i >= -1; ly_i -= 1) { 
        for (var lx_i = -1; lx_i <= 1; lx_i += 1) { 
            if (lx_i == 0 && ly_i == 0) { continue; } 
            
            let fwd = f32(ly_i) * view_spacing;
            let lat = f32(lx_i) * view_spacing;

            var rot_x = (fwd * cos_h - lat * sin_h) * lon_scale;
            var rot_y = fwd * sin_h + lat * cos_h;
            
            var wrap_x = (ax + rot_x) % map_w_f32;
            if (wrap_x < 0.0) { wrap_x = wrap_x + map_w_f32; }
            var wrap_y = ay + rot_y;
            if (wrap_y < 0.0) { wrap_y = -wrap_y; wrap_x = (wrap_x + map_w_f32 / 2.0) % map_w_f32; }
            else if (wrap_y >= map_h_f32) { wrap_y = 2.0 * map_h_f32 - 1.0 - wrap_y; wrap_x = (wrap_x + map_w_f32 / 2.0) % map_w_f32; }

            let ux = u32(wrap_x);
            let uy = u32(wrap_y);
            
            var s_res = 0.0;
            var s_height = 0.0;
            var s_pop = 0.0;
            
            if (ux >= lds_base_x && ux < lds_base_x + 16u && uy >= lds_base_y && uy < lds_base_y + 16u) {
                let l_idx = (uy - lds_base_y) * 16u + (ux - lds_base_x);
                s_res = lds_res[l_idx];
                s_height = lds_height[l_idx];
                s_pop = lds_pop[l_idx];
            } else {
                let s_idx = clamp(uy * map_w + ux, 0u, max_idx);
                s_res = f32(atomicLoad(&map_cells[s_idx].res_value)) / 1000.0;
                s_height = map_heights[s_idx];
                s_pop = map_cells[s_idx].population;
            }

            inputs[input_idx] = s_res * vision_multiplier;
            inputs[input_idx + 1u] = s_height;
            inputs[input_idx + 2u] = s_pop * vision_multiplier;
            
            let s_idx_g = clamp(uy * map_w + ux, 0u, max_idx);
            let s_cell = &map_cells[s_idx_g];
            inputs[input_idx + 3u] = (*s_cell).comm1;
            inputs[input_idx + 4u] = (*s_cell).comm2;
            inputs[input_idx + 5u] = (*s_cell).comm3;
            inputs[input_idx + 6u] = (*s_cell).comm4;
            inputs[input_idx + 7u] = (*s_cell).pheno_r;
            inputs[input_idx + 8u] = (*s_cell).pheno_g;
            inputs[input_idx + 9u] = (*s_cell).pheno_b;
            inputs[input_idx + 10u] = f32(atomicLoad(&(*s_cell).infra_roads)) / 1000000.0; 
            inputs[input_idx + 11u] = f32(atomicLoad(&(*s_cell).infra_housing)) / 1000000.0;
            inputs[input_idx + 12u] = f32(atomicLoad(&(*s_cell).infra_farms)) / 1000000.0;
            inputs[input_idx + 13u] = f32(atomicLoad(&(*s_cell).infra_storage)) / 1000000.0;
            
            input_idx += 14u;
        }
    }

    let hidden_count = agents[idx].hidden_count;
    var hidden1 = array<f32, 128>();
    for (var h1 = 0u; h1 < hidden_count; h1 = h1 + 1u) {
        var sum = 0.0;
        let w_base = h1 * 8u;
        for (var k = 0u; k < 8u; k = k + 1u) {
            let in_idx = genetics[g_idx].w1_indices[w_base + k];
            sum = sum + inputs[in_idx] * genetics[g_idx].w1_weights[w_base + k];
        }
        hidden1[h1] = fast_tanh(sum);
    }

    var hidden2 = array<f32, 128>();
    for (var h2 = 0u; h2 < hidden_count; h2 = h2 + 1u) {
        var sum = 0.0;
        let h2_base = h2;
        for (var h1 = 0u; h1 < hidden_count; h1 = h1 + 1u) {
            sum = sum + hidden1[h1] * genetics[g_idx].w2[h1 * 128u + h2_base];
        }
        hidden2[h2] = fast_tanh(sum);
    }

    var outputs = array<f32, 56>(); 
    for (var o = 0u; o < 56u; o = o + 1u) {
        var sum = 0.0;
        for (var h2 = 0u; h2 < hidden_count; h2 = h2 + 1u) {
            sum = sum + hidden2[h2] * genetics[g_idx].w3[h2 * 56u + o];
        }
        outputs[o] = fast_tanh(sum);
    }

    // --- Action Processing ---
    let maturity = clamp(agents[idx].age / cfg.bio.puberty_age, 0.0, 1.0);
    let age_speed_mult = mix(cfg.bio.infant_speed_mult, 1.0, maturity);
    let age_stamina_mult = mix(cfg.bio.infant_stamina_mult, 1.0, maturity);

    let rest_intent = outputs[5] * 0.5 + 0.5;
    var resting = rest_intent > 0.5 || agents[idx].stamina <= 0.0;

    agents[idx].heading = ah + select(outputs[0], 0.0, resting) * 0.5;
    var speed_intent = clamp(outputs[1] * 0.5 + 0.5, 0.0, 1.0) * age_speed_mult; 
    
    if (maturity < 1.0) {
        var adult_nearby = false;
        let cx = u32(ax); let cy = u32(ay);
        for (var dy = -1i; dy <= 1; dy = dy + 1) {
            for (var dx = -1i; dx <= 1; dx = dx + 1) {
                let s_idx = (u32(i32(cy) + dy) % map_h) * map_w + (u32(i32(cx) + dx) % map_w);
                if (atomicLoad(&map_cells[s_idx].adult_count) > 0) { adult_nearby = true; break; }
            }
            if (adult_nearby) { break; }
        }
        if (!adult_nearby) { speed_intent = 0.0; }
    }

    if (resting) { speed_intent = 0.0; }
    
    agents[idx].reproduce_desire = select(outputs[3] * 0.5 + 0.5, 0.0, resting);
    agents[idx].attack_intent = select(outputs[4] * 0.5 + 0.5, 0.0, resting);
    agents[idx].rest_intent = rest_intent;
    
    var comm_exertion = 0.0;
    for (var c = 0u; c < 12u; c = c + 1u) {
        var val = outputs[6 + c];
        if (abs(val) < 0.15) { val = 0.0; } 
        let final_val = select(clamp(val, -1.0, 1.0), 0.0, resting);
        agents[idx].comms[c] = final_val;
        comm_exertion = comm_exertion + abs(final_val);
    }

    let learn_intent = outputs[18] * 0.5 + 0.5; 
    for (var m = 0u; m < 24u; m = m + 1u) { agents[idx].mems[m] = outputs[19 + m]; }
    
    agents[idx].buy_intent = select(outputs[43] * 0.5 + 0.5, 0.0, resting);
    agents[idx].sell_intent = select(outputs[44] * 0.5 + 0.5, 0.0, resting);
    agents[idx].ask_price = abs(outputs[45]) * 10.0;
    agents[idx].bid_price = abs(outputs[46]) * 10.0;
    agents[idx].drop_water_intent = select(outputs[47] * 0.5 + 0.5, 0.0, resting);
    agents[idx].pickup_water_intent = select(outputs[48] * 0.5 + 0.5, 0.0, resting);
    agents[idx].defend_intent = select(outputs[49] * 0.5 + 0.5, 0.0, resting);
    agents[idx].emergency_intent = outputs[55] * 0.5 + 0.5;

    let b_road = outputs[50] * 0.5 + 0.5;
    let b_house = outputs[51] * 0.5 + 0.5;
    let b_farm = outputs[52] * 0.5 + 0.5;
    let b_storage = outputs[53] * 0.5 + 0.5;
    let max_b = max(b_road, max(b_house, max(b_farm, b_storage)));
    let is_building = max_b > 0.5 && max_b > speed_intent && agents[idx].emergency_intent < 0.5 && !resting;

    if (!is_building) {
        agents[idx].build_road_intent = 0.0; agents[idx].build_house_intent = 0.0; agents[idx].build_farm_intent = 0.0; agents[idx].build_storage_intent = 0.0;
    } else {
        agents[idx].build_road_intent = b_road; agents[idx].build_house_intent = b_house; agents[idx].build_farm_intent = b_farm; agents[idx].build_storage_intent = b_storage;
    }
    
    agents[idx].destroy_infra_intent = select(outputs[54] * 0.5 + 0.5, 0.0, resting || agents[idx].emergency_intent > 0.5);

    var base_speed = speed_intent * cfg.bio.base_speed;
    if (resting || is_building) {
        base_speed = 0.0;
        if (resting) { agents[idx].stamina = min(agents[idx].stamina + 2.0, cfg.bio.max_stamina); }
    }

    let next_h = ah + select(outputs[0], 0.0, resting) * 0.5;
    let next_x = ax + cos(next_h) * base_speed * lon_scale;
    let next_y = ay + sin(next_h) * base_speed;

    var wrap_nx = next_x % map_w_f32; if (wrap_nx < 0.0) { wrap_nx = wrap_nx + map_w_f32; }
    var wrap_ny = next_y;
    if (wrap_ny < 0.0) { wrap_ny = -wrap_ny; wrap_nx = (wrap_nx + map_w_f32 / 2.0) % map_w_f32; }
    else if (wrap_ny >= map_h_f32) { wrap_ny = 2.0 * map_h_f32 - 1.0 - wrap_ny; wrap_nx = (wrap_nx + map_w_f32 / 2.0) % map_w_f32; }
    
    let next_idx = clamp(u32(wrap_ny) * map_w + u32(wrap_nx), 0u, max_idx);
    let slope = map_heights[next_idx] - map_heights[safe_current_idx];
    let h_mult = 1.0 - clamp(slope * cfg.eco.climb_penalty, -0.5, 0.9);
    
    let total_w = (agents[idx].food / 1000.0) + agents[idx].water;
    let enc_mult = clamp(1.0 - (total_w / cfg.bio.max_carry_weight), 0.05, 1.0);
    let crowd_mult = clamp(1.0 - (local_population / cfg.combat.crowding_threshold), 0.1, 1.0);
    let road_mult = 1.0 + (local_infra_roads / cfg.infra.max_infra) * cfg.infra.road_speed_bonus; 
    
    var actual_speed = base_speed * h_mult * enc_mult * crowd_mult * road_mult;

    if (agents[idx].gestation_timer > 0.0) {
        agents[idx].gestation_timer = agents[idx].gestation_timer - 1.0;
        if (agents[idx].gestation_timer <= 0.0) { agents[idx].is_pregnant = 0.0; }
    }

    var preg_m = 1.0;
    if (agents[idx].is_pregnant > 0.5) { actual_speed = actual_speed * cfg.combat.pregnancy_speed_mult; preg_m = cfg.combat.pregnancy_cost_mult; }
    
    var int_ex = 0.0;
    if (!resting) {
        int_ex = select(0.0, 0.15, agents[idx].build_road_intent > 0.5) + select(0.0, 0.15, agents[idx].build_house_intent > 0.5) + select(0.0, 0.15, agents[idx].build_farm_intent > 0.5) + select(0.0, 0.15, agents[idx].build_storage_intent > 0.5);
        int_ex = int_ex + select(0.0, 0.25, agents[idx].destroy_infra_intent > 0.5) + (comm_exertion * 0.01);
    }

    if (!resting) {
        agents[idx].stamina = agents[idx].stamina - ((actual_speed * 0.05 + 0.05 + int_ex * 0.2) * (2.0 - age_stamina_mult));
        if (agents[idx].stamina <= 0.0) {
            agents[idx].stamina = 0.0; agents[idx].health = agents[idx].health - (cfg.bio.starvation_rate * 2.0); resting = true;
        }
    }

    let final_spd = select(actual_speed, 0.0, resting);
    let metabolic_rate = cfg.eco.baseline_cost * clamp(agents[idx].age / cfg.bio.puberty_age, 0.2, 1.0) * select(1.0, 0.5 - ((local_infra_housing / cfg.infra.max_infra) * cfg.infra.housing_rest_bonus), resting) * preg_m * select(1.0, cfg.combat.defend_cost_mult, agents[idx].defend_intent > 0.5 && !resting) * (1.0 + (total_w / 50.0)) * (1.0 + int_ex);

    agents[idx].water = agents[idx].water - metabolic_rate;
    agents[idx].food = agents[idx].food - (metabolic_rate * 1000.0) - (max(0.0, -effective_temp) * 100.0);
    
    let storage_m = 1.0 - clamp((f32(atomicLoad(&(*cell_ptr).infra_storage)) / 1000.0 / cfg.infra.max_infra) * cfg.infra.storage_rot_reduction, 0.0, cfg.infra.storage_rot_reduction); 
    agents[idx].food = agents[idx].food - (agents[idx].food * cfg.eco.base_spoilage_rate * max(0.1, 1.0 + local_temp) * storage_m);
    
    if (agents[idx].food < 0.0) { agents[idx].food = 0.0; agents[idx].health = agents[idx].health - cfg.bio.starvation_rate; }
    if (agents[idx].water < 0.0) { agents[idx].water = 0.0; agents[idx].health = agents[idx].health - cfg.bio.starvation_rate; }
    if (agents[idx].food > (metabolic_rate * 1000.0) && agents[idx].water > metabolic_rate && agents[idx].health < cfg.bio.max_health) {
        agents[idx].health = min(agents[idx].health + (cfg.bio.starvation_rate * 2.0), cfg.bio.max_health);
        agents[idx].food = agents[idx].food - (metabolic_rate * 1000.0); agents[idx].water = agents[idx].water - metabolic_rate; 
    }

    if (map_heights[safe_current_idx] >= 0.0) {
        if (local_avg_aggression > 0.5 && agents[idx].defend_intent < 0.5) { agents[idx].health = agents[idx].health - cfg.combat.bystander_damage; }
        
        if (agents[idx].attack_intent > 0.5 && local_population > 1.0 && !resting && agents[idx].age > cfg.bio.puberty_age) {
            if (agents[idx].defend_intent < 0.5) {
                agents[idx].wealth = agents[idx].wealth + min((local_population - 1.0) * 2.5, cfg.eco.boat_cost); 
                if (local_avg_aggression > 0.5) { agents[idx].health = agents[idx].health - cfg.combat.attacker_damage; }
            }
        } else if (agents[idx].drop_water_intent > 0.5 && agents[idx].water > cfg.eco.water_transfer_amount) {
            agents[idx].water = agents[idx].water - cfg.eco.water_transfer_amount;
            atomicAdd(&(*cell_ptr).market_water, i32(cfg.eco.water_transfer_amount * 1000.0));
        } else if (agents[idx].pickup_water_intent > 0.5 && f32(atomicLoad(&(*cell_ptr).market_water)) > 0.0 && agents[idx].water < cfg.bio.max_water) {
            agents[idx].water = min(agents[idx].water + cfg.eco.water_transfer_amount, cfg.bio.max_water);
            atomicSub(&(*cell_ptr).market_water, i32(cfg.eco.water_transfer_amount * 1000.0));
        } else if (map_heights[safe_current_idx] <= 0.05 && agents[idx].water < cfg.bio.max_water) {
            agents[idx].water = min(agents[idx].water + cfg.eco.water_transfer_amount, cfg.bio.max_water);
        } else if (outputs[2] > 0.5 && !resting && agents[idx].food > 100000.0) {
            agents[idx].food = agents[idx].food - (cfg.eco.drop_amount * 1000.0);
            atomicAdd(&(*cell_ptr).res_value, i32(cfg.eco.drop_amount * 1000.0));
        } else if (!resting && local_res_value > 0.1) {
            let gathered = min(cfg.eco.base_gather_rate * (1.0 + min(sqrt(agents[idx].food / 1000.0) * 0.1, (cfg.eco.max_gather_rate / cfg.eco.base_gather_rate) - 1.0)) * maturity, local_res_value);
            agents[idx].food = agents[idx].food + (gathered * 1000.0);
            atomicSub(&(*cell_ptr).res_value, i32(gathered * 1000.0));
        }
        
        let b_amt = i32(1000.0 / max(1.0, cfg.infra.build_ticks));
        let c_amt = cfg.infra.infra_cost / max(1.0, cfg.infra.build_ticks);

        if (!resting && agents[idx].wealth >= c_amt) {
            if (agents[idx].build_road_intent > 0.5) { agents[idx].wealth -= c_amt; atomicAdd(&(*cell_ptr).infra_roads, b_amt); }
            else if (agents[idx].build_house_intent > 0.5) { agents[idx].wealth -= c_amt; atomicAdd(&(*cell_ptr).infra_housing, b_amt); }
            else if (agents[idx].build_farm_intent > 0.5) { agents[idx].wealth -= c_amt; atomicAdd(&(*cell_ptr).infra_farms, b_amt); }
            else if (agents[idx].build_storage_intent > 0.5) { agents[idx].wealth -= c_amt; atomicAdd(&(*cell_ptr).infra_storage, b_amt); }
        }
        
        (*cell_ptr).avg_ask = mix(local_avg_ask, agents[idx].ask_price, 0.1);
        (*cell_ptr).avg_bid = mix(local_avg_bid, agents[idx].bid_price, 0.1);
        
        if (agents[idx].buy_intent > 0.5 && agents[idx].wealth >= local_avg_ask && f32(atomicLoad(&(*cell_ptr).market_food)) >= 1000000.0) {
            agents[idx].wealth -= local_avg_ask; agents[idx].food += 1000.0;
            atomicAdd(&(*cell_ptr).market_wealth, i32(local_avg_ask * 1000.0)); atomicSub(&(*cell_ptr).market_food, 1000000);
        } else if (agents[idx].sell_intent > 0.5 && agents[idx].food >= 1000.0 && f32(atomicLoad(&(*cell_ptr).market_wealth)) >= local_avg_bid * 1000.0) {
            agents[idx].food -= 1000.0; agents[idx].wealth += local_avg_bid;
            atomicAdd(&(*cell_ptr).market_food, 1000000); atomicSub(&(*cell_ptr).market_wealth, i32(local_avg_bid * 1000.0));
        }
        
        let regen = cfg.world.regen_rate * clamp(1.0 - (map_heights[safe_current_idx] * 2.0), 0.05, 1.0) * clamp(0.05 + (f32(atomicLoad(&(*cell_ptr).market_water)) / 100000.0), 0.05, 5.0) * (1.0 + (f32(atomicLoad(&(*cell_ptr).infra_farms)) / 1000.0 / cfg.infra.max_infra) * 5.0);
        atomicAdd(&(*cell_ptr).res_value, i32(regen * 1000.0));
        atomicMin(&(*cell_ptr).res_value, i32(cfg.world.max_tile_resource * 1000.0));
        
        if (agents[idx].age >= cfg.bio.puberty_age) { atomicAdd(&(*cell_ptr).adult_count, 1); }
        (*cell_ptr).population = local_population * 0.99 + 1.0;
        (*cell_ptr).avg_speed = mix(local_avg_speed, speed_intent, 0.1);
        (*cell_ptr).pheno_r = mix((*cell_ptr).pheno_r, agents[idx].pheno_r, 0.1);
    }

    if (agents[idx].food >= final_spd * cfg.eco.move_cost_per_unit * 1000.0) {
        agents[idx].x = wrap_nx; agents[idx].y = wrap_ny; agents[idx].heading = ah + select(outputs[0], 0.0, resting) * 0.5;
        agents[idx].food = agents[idx].food - (final_spd * cfg.eco.move_cost_per_unit * 1000.0);
    }
    
    if (agents[idx].age >= cfg.bio.max_age) {
        let biological_decay = cfg.bio.starvation_rate * (2.0 + pow((agents[idx].age - cfg.bio.max_age) / (525600.0 / cfg.world.tick_to_mins), 1.5));
        agents[idx].health = agents[idx].health - biological_decay * select(1.0, 0.5, agents[idx].wealth >= biological_decay * 50.0);
    }

    let dopamine = ((agents[idx].health - prev_health) * 20.0) + ((agents[idx].food - prev_food) / 10000.0) + ((agents[idx].wealth - prev_wealth) * 0.5);
    if (abs(dopamine) > 0.01 && learn_intent > 0.1) {
        let lr = dopamine * learn_intent * 0.0005; 
        for (var h1 = 0u; h1 < hidden_count; h1 = h1 + 1u) {
            for (var k = 0u; k < 8u; k = k + 1u) {
                let i_idx = genetics[g_idx].w1_indices[h1 * 8u + k];
                genetics[g_idx].w1_weights[h1 * 8u + k] = clamp(genetics[g_idx].w1_weights[h1 * 8u + k] + lr * inputs[i_idx] * hidden1[h1], -2.0, 2.0);
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
