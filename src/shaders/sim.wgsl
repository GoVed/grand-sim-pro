// Fast Local Memory Cache for workgroup map tiles (e.g., 16x16 patch)
var<workgroup> lds_res: array<f32, 256>; 
var<workgroup> lds_height: array<f32, 256>;
var<workgroup> lds_pop: array<f32, 256>;
var<workgroup> lds_base_x: u32;
var<workgroup> lds_base_y: u32;

fn is_nan(x: f32) -> bool {
    return x != x;
}

fn fast_tanh(x: f32) -> f32 {
    let x2 = x * x;
    return x * (27.0 + x2) / (27.0 + 9.0 * x2);
}

fn get_id_feature(id: u32, slot: u32) -> f32 {
    let h = (id ^ (id >> 16u) ^ (slot * 0x85ebca6bu)) * 0x45d9f3bu;
    let h2 = (h ^ (h >> 16u)) * 0x45d9f3bu;
    let res = h2 ^ (h2 >> 16u);
    return (f32(res % 2000u) / 1000.0) - 1.0; // Range -1 to 1
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

    // Store pre-update state for dopamine/learning
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
    let cell_ptr = &map_cells[safe_current_idx];
    
    // Day/Night and Temp
    let ticks_per_day = 24.0 * 60.0 / cfg.world.tick_to_mins;
    let day_cycle_progress = (f32(cfg.sim.current_tick) % ticks_per_day) / ticks_per_day; 
    let day_intensity = sin(day_cycle_progress * 6.28318 - 1.5708) * 0.5 + 0.5; 
    let lon_scale = 1.0 / max(cos((abs(ay - map_h_f32 / 2.0) / (map_h_f32 / 2.0)) * 1.570796), 0.15); 
    
    let season_time = f32(cfg.sim.current_tick) / 100000.0 * 6.28318;
    let season_sine = sin(season_time);
    let local_temp = (1.0 - (abs(ay - map_h_f32 / 2.0) / (map_h_f32 / 2.0)) * 2.0) - max(0.0, map_heights[safe_current_idx] * 2.0) + season_sine * 0.5 - (1.0 - day_intensity) * 0.3; 
    let effective_temp = local_temp + (f32(atomicLoad(&(*cell_ptr).infra_housing)) / 1000.0 / cfg.infra.max_infra) * 2.0;

    var vis_mult = 1.0;
    if (agents[idx].rest_intent > 0.5 || agents[idx].stamina <= 0.0) { vis_mult = 0.1; }
    let vision_multiplier = (0.3 + day_intensity * 0.7) * vis_mult; 

    // --- Identity Features (Self) ---
    agents[idx].id_f1 = get_id_feature(agents[idx].id, 1u);
    agents[idx].id_f2 = get_id_feature(agents[idx].id, 2u);
    agents[idx].id_f3 = get_id_feature(agents[idx].id, 3u);
    agents[idx].id_f4 = get_id_feature(agents[idx].id, 4u);

    // --- Identity Sensing (Nearest Neighbor) ---
    var min_dist = 10.0;
    var nf1 = 0.0; var nf2 = 0.0; var nf3 = 0.0; var nf4 = 0.0;
    let scan_range = 16u;
    let start_scan = select(0u, idx - scan_range, idx > scan_range);
    let end_scan = min(arrayLength(&agents), idx + scan_range);
    for (var i = start_scan; i < end_scan; i = i + 1u) {
        if (i == idx || agents[i].health <= 0.0) { continue; }
        let d = distance(vec2<f32>(agents[i].x, agents[i].y), vec2<f32>(ax, ay));
        if (d < min_dist) {
            min_dist = d;
            nf1 = get_id_feature(agents[i].id, 1u); nf2 = get_id_feature(agents[i].id, 2u);
            nf3 = get_id_feature(agents[i].id, 3u); nf4 = get_id_feature(agents[i].id, 4u);
        }
    }
    agents[idx].nearest_id_f1 = nf1; agents[idx].nearest_id_f2 = nf2; agents[idx].nearest_id_f3 = nf3; agents[idx].nearest_id_f4 = nf4;

    // --- Neural Net Inputs ---
    var inputs = array<f32, 420>();
    inputs[0] = 1.0; // Bias
    inputs[1] = f32(atomicLoad(&(*cell_ptr).res_value)) / 1000.0 / 1000.0;
    inputs[2] = (*cell_ptr).population;
    inputs[3] = (*cell_ptr).avg_speed;
    inputs[4] = (*cell_ptr).avg_share;
    inputs[5] = (*cell_ptr).avg_reproduce;
    inputs[6] = (*cell_ptr).avg_aggression;
    inputs[7] = (*cell_ptr).avg_pregnancy;
    inputs[8] = (*cell_ptr).avg_turn;
    inputs[9] = (*cell_ptr).avg_rest;
    
    // Comms (standard slots)
    inputs[10] = (*cell_ptr).comm1; inputs[11] = (*cell_ptr).comm2; inputs[12] = (*cell_ptr).comm3; inputs[13] = (*cell_ptr).comm4;
    
    // Biometrics
    inputs[22] = agents[idx].health / cfg.bio.max_health;
    inputs[23] = (agents[idx].food / 1000.0) / cfg.eco.boat_cost;
    inputs[24] = agents[idx].water / cfg.bio.max_water;
    inputs[25] = agents[idx].stamina / cfg.bio.max_stamina;
    inputs[26] = agents[idx].age / cfg.bio.max_age;
    inputs[27] = agents[idx].gender;
    inputs[28] = local_temp;
    inputs[29] = season_sine;
    inputs[30] = agents[idx].is_pregnant;
    inputs[31] = (agents[idx].food / 1000.0 + agents[idx].water) / cfg.bio.max_carry_weight;
    inputs[32] = (*cell_ptr).population / cfg.combat.crowding_threshold;
    for (var m = 0u; m < 24u; m = m + 1u) { inputs[33 + m] = agents[idx].mems[m]; }
    inputs[57] = agents[idx].wealth / cfg.eco.boat_cost;
    inputs[58] = (*cell_ptr).avg_ask / 10.0;
    inputs[59] = (*cell_ptr).avg_bid / 10.0;
    inputs[60] = day_intensity;
    inputs[61] = agents[idx].id_f1; inputs[62] = agents[idx].id_f2; inputs[63] = agents[idx].id_f3; inputs[64] = agents[idx].id_f4;
    inputs[65] = (*cell_ptr).pheno_r; inputs[66] = (*cell_ptr).pheno_g; inputs[67] = (*cell_ptr).pheno_b;

    // --- 5x5 Vision Grid ---
    var input_idx = 68u;
    let view_spacing = 8.0;
    var vision_resources = array<f32, 24>();
    var v_count = 0u;

    for (var ly_i = 3; ly_i >= -1; ly_i -= 1) { 
        for (var lx_i = -2; lx_i <= 2; lx_i += 1) { 
            if (lx_i == 0 && ly_i == 0) { continue; } 
            
            let fwd = f32(ly_i) * view_spacing;
            let lat = f32(lx_i) * view_spacing;
            var rot_x = (fwd * cos(ah) - lat * sin(ah)) * lon_scale;
            var rot_y = fwd * sin(ah) + lat * cos(ah);
            var wx = (ax + rot_x) % map_w_f32; if (wx < 0.0) { wx += map_w_f32; }
            var wy = ay + rot_y;
            if (wy < 0.0) { wy = -wy; wx = (wx + map_w_f32 / 2.0) % map_w_f32; }
            else if (wy >= map_h_f32) { wy = 2.0 * map_h_f32 - 1.0 - wy; wx = (wx + map_w_f32 / 2.0) % map_w_f32; }

            let s_idx = clamp(u32(wy) * map_w + u32(wx), 0u, max_idx);
            let s_cell = &map_cells[s_idx];
            
            let s_res = f32(atomicLoad(&(*s_cell).res_value)) / 1000.0;
            inputs[input_idx] = s_res * vision_multiplier;
            inputs[input_idx+1u] = map_heights[s_idx];
            inputs[input_idx+2u] = (*s_cell).population * vision_multiplier;
            inputs[input_idx+3u] = (*s_cell).comm1; inputs[input_idx+4u] = (*s_cell).comm2;
            inputs[input_idx+5u] = (*s_cell).comm3; inputs[input_idx+6u] = (*s_cell).comm4;
            inputs[input_idx+7u] = (*s_cell).pheno_r; inputs[input_idx+8u] = (*s_cell).pheno_g; inputs[input_idx+9u] = (*s_cell).pheno_b;
            inputs[input_idx+10u] = f32(atomicLoad(&(*s_cell).infra_roads)) / 1000.0;
            inputs[input_idx+11u] = f32(atomicLoad(&(*s_cell).infra_housing)) / 1000.0;
            inputs[input_idx+12u] = f32(atomicLoad(&(*s_cell).infra_farms)) / 1000.0;
            inputs[input_idx+13u] = f32(atomicLoad(&(*s_cell).infra_storage)) / 1000.0;
            
            vision_resources[v_count] = s_res;
            v_count++;
            input_idx += 14u;
        }
    }

    // --- Spatial CNN Stage ---
    for (var f = 0u; f < 8u; f = f + 1u) {
        var sum = 0.0;
        let k_base = f * 9u;
        for (var k = 0u; k < 9u; k = k + 1u) {
            let k_val = genetics[g_idx].cnn_kernels[k_base + k]; 
            sum += vision_resources[(f + k) % 24u] * k_val;
        }
        agents[idx].spatial_features[f] = fast_tanh(sum);
    }

    // Local / Meta / Identity / CNN wiring
    inputs[403] = f32(atomicLoad(&(*cell_ptr).infra_roads)) / 1000.0 / cfg.infra.max_infra;
    inputs[404] = f32(atomicLoad(&(*cell_ptr).infra_housing)) / 1000.0 / cfg.infra.max_infra;
    inputs[405] = f32(atomicLoad(&(*cell_ptr).infra_farms)) / 1000.0 / cfg.infra.max_infra;
    inputs[406] = f32(atomicLoad(&(*cell_ptr).infra_storage)) / 1000.0 / cfg.infra.max_infra;
    inputs[407] = agents[idx].nearest_id_f1; inputs[408] = agents[idx].nearest_id_f2;
    inputs[409] = agents[idx].nearest_id_f3; inputs[410] = agents[idx].nearest_id_f4;
    for (var f = 0u; f < 8u; f = f + 1u) { inputs[412 + f] = agents[idx].spatial_features[f]; }

    // --- Forward Pass ---
    let hidden_count = agents[idx].hidden_count;
    var hidden1 = array<f32, 128>();
    for (var h1 = 0u; h1 < hidden_count; h1 = h1 + 1u) {
        var sum = 0.0;
        let w_base = h1 * 8u;
        for (var k = 0u; k < 8u; k = k + 1u) {
            let i_idx = genetics[g_idx].w1_indices[w_base + k];
            sum += inputs[i_idx] * genetics[g_idx].w1_weights[w_base + k];
        }
        // Hebbian Plasticity contribution
        for (var p = 0u; p < 32u; p = p + 1u) {
            if (agents[idx].plastic_indices[p] == h1) {
                sum += agents[idx].plastic_weights[p] * inputs[0];
            }
        }
        hidden1[h1] = fast_tanh(clamp(sum, -10.0, 10.0));
    }

    var hidden2 = array<f32, 128>();
    for (var h2 = 0u; h2 < hidden_count; h2 = h2 + 1u) {
        var sum = 0.0;
        for (var h1 = 0u; h1 < hidden_count; h1 = h1 + 1u) {
            sum += hidden1[h1] * genetics[g_idx].w2[h1 * 128u + h2];
        }
        hidden2[h2] = fast_tanh(clamp(sum, -10.0, 10.0));
    }

    var outputs = array<f32, 56>(); 
    for (var o = 0u; o < 56u; o = o + 1u) {
        var sum = 0.0;
        for (var h2 = 0u; h2 < hidden_count; h2 = h2 + 1u) {
            sum += hidden2[h2] * genetics[g_idx].w3[h2 * 56u + o];
        }
        outputs[o] = fast_tanh(clamp(sum, -10.0, 10.0));
    }

    // --- Action Processing ---
    let maturity = clamp(agents[idx].age / cfg.bio.puberty_age, 0.2, 1.0);
    let age_speed_mult = mix(cfg.bio.infant_speed_mult, 1.0, maturity);
    let rest_intent = outputs[5] * 0.5 + 0.5;
    var resting = rest_intent > 0.5 || agents[idx].stamina <= 0.0;
    
    let turn_intent = outputs[0] * 0.5;
    let speed_intent = clamp(outputs[1] * 0.5 + 0.5, 0.0, 1.0) * age_speed_mult; 

    agents[idx].reproduce_desire = select(outputs[3] * 0.5 + 0.5, 0.0, resting);
    agents[idx].attack_intent = select(outputs[4] * 0.5 + 0.5, 0.0, resting);
    agents[idx].rest_intent = rest_intent;
    
    var comm_exertion = 0.0;
    for (var c = 0u; c < 12u; c = c + 1u) {
        let val = select(clamp(outputs[6+c], -1.0, 1.0), 0.0, resting);
        agents[idx].comms[c] = val;
        comm_exertion += abs(val);
    }
    for (var m = 0u; m < 24u; m = m + 1u) { agents[idx].mems[m] = outputs[19+m]; }
    
    agents[idx].buy_intent = select(outputs[43] * 0.5 + 0.5, 0.0, resting);
    agents[idx].sell_intent = select(outputs[44] * 0.5 + 0.5, 0.0, resting);
    agents[idx].ask_price = exp((outputs[45] * 0.5 + 0.5) * 10.0);
    agents[idx].bid_price = exp((outputs[46] * 0.5 + 0.5) * 10.0);
    agents[idx].drop_water_intent = select(outputs[47] * 0.5 + 0.5, 0.0, resting);
    agents[idx].pickup_water_intent = select(outputs[48] * 0.5 + 0.5, 0.0, resting);
    agents[idx].defend_intent = select(outputs[49] * 0.5 + 0.5, 0.0, resting);
    agents[idx].emergency_intent = outputs[55] * 0.5 + 0.5;
    
    let b_road = outputs[50] * 0.5 + 0.5; let b_house = outputs[51] * 0.5 + 0.5;
    let b_farm = outputs[52] * 0.5 + 0.5; let b_storage = outputs[53] * 0.5 + 0.5;
    let max_b = max(b_road, max(b_house, max(b_farm, b_storage)));
    let is_building = max_b > 0.5 && !resting;
    if (is_building) {
        agents[idx].build_road_intent = b_road; agents[idx].build_house_intent = b_house;
        agents[idx].build_farm_intent = b_farm; agents[idx].build_storage_intent = b_storage;
    } else {
        agents[idx].build_road_intent = 0.0; agents[idx].build_house_intent = 0.0;
        agents[idx].build_farm_intent = 0.0; agents[idx].build_storage_intent = 0.0;
    }

    // --- Simulation Physics & Metabolism ---
    var base_speed = select(speed_intent * cfg.bio.base_speed, 0.0, resting || is_building);
    if (resting) { agents[idx].stamina = min(agents[idx].stamina + 2.0, cfg.bio.max_stamina); }

    let current_h = ah + select(turn_intent, 0.0, resting || is_building);
    
    let total_w = (agents[idx].food / 1000.0) + agents[idx].water;
    let enc_mult = clamp(1.0 - (total_w / cfg.bio.max_carry_weight), 0.05, 1.0);
    let crowd_mult = clamp(1.0 - ((*cell_ptr).population / cfg.combat.crowding_threshold), 0.1, 1.0);
    
    var actual_speed = base_speed * enc_mult * crowd_mult;
    if (agents[idx].is_pregnant > 0.5) { actual_speed *= cfg.combat.pregnancy_speed_mult; }

    // Position Prediction for slope
    let pred_x = ax + cos(current_h) * actual_speed * lon_scale;
    let pred_y = ay + sin(current_h) * actual_speed;
    var wrap_px = pred_x % map_w_f32; if (wrap_px < 0.0) { wrap_px += map_w_f32; }
    var wrap_py = clamp(pred_y, 0.0, map_h_f32 - 1.0);
    let pred_idx = clamp(u32(wrap_py) * map_w + u32(wrap_px), 0u, max_idx);
    let slope = map_heights[pred_idx] - map_heights[safe_current_idx];
    let h_mult = 1.0 - clamp(slope * cfg.eco.climb_penalty, -0.5, 0.9);
    
    actual_speed *= h_mult;
    let next_x = ax + cos(current_h) * actual_speed * lon_scale;
    let next_y = ay + sin(current_h) * actual_speed;
    var wrap_nx = next_x % map_w_f32; if (wrap_nx < 0.0) { wrap_nx += map_w_f32; }
    var wrap_ny = clamp(next_y, 0.0, map_h_f32 - 1.0);

    if (!resting) {
        agents[idx].stamina -= (actual_speed * 0.05 + 0.05);
        if (agents[idx].stamina <= 0.0) { agents[idx].stamina = 0.0; agents[idx].health -= (cfg.bio.starvation_rate * 2.0); }
    }

    let metabolic_rate = cfg.eco.baseline_cost * maturity * select(1.0, 0.5, resting);
    agents[idx].water -= metabolic_rate;
    agents[idx].food -= (metabolic_rate * 1000.0);
    
    if (agents[idx].food < 0.0) { agents[idx].food = 0.0; agents[idx].health -= cfg.bio.starvation_rate; }
    if (agents[idx].water < 0.0) { agents[idx].water = 0.0; agents[idx].health -= cfg.bio.starvation_rate; }

    // Harvesting
    if (!resting && f32(atomicLoad(&(*cell_ptr).res_value)) > 100.0) {
        let gathered = min(cfg.eco.base_gather_rate * maturity, f32(atomicLoad(&(*cell_ptr).res_value)) / 1000.0);
        agents[idx].food += (gathered * 1000.0);
        atomicSub(&(*cell_ptr).res_value, i32(gathered * 1000.0));
    }

    // Infrastructure Building
    let c_amt = cfg.infra.infra_cost;
    if (!resting && is_building && agents[idx].wealth >= c_amt) {
        agents[idx].wealth -= c_amt;
        let b_amt = i32(1000.0 / max(1.0, cfg.infra.build_ticks));
        if (agents[idx].build_road_intent > 0.5) { atomicAdd(&(*cell_ptr).infra_roads, b_amt); }
        else if (agents[idx].build_house_intent > 0.5) { atomicAdd(&(*cell_ptr).infra_housing, b_amt); }
        else if (agents[idx].build_farm_intent > 0.5) { atomicAdd(&(*cell_ptr).infra_farms, b_amt); }
        else if (agents[idx].build_storage_intent > 0.5) { atomicAdd(&(*cell_ptr).infra_storage, b_amt); }
    }

    // Aggression & Combat
    if (!resting && maturity >= 1.0) {
        if ((*cell_ptr).avg_aggression > 0.5 && agents[idx].defend_intent < 0.5) {
            agents[idx].health -= cfg.combat.bystander_damage;
        }
        if (agents[idx].attack_intent > 0.5 && (*cell_ptr).population > 1.0) {
            agents[idx].wealth += 1.0; 
        }
    }

    // Movement Commit
    if (agents[idx].food >= actual_speed * cfg.eco.move_cost_per_unit * 1000.0) {
        agents[idx].x = wrap_nx; agents[idx].y = wrap_ny; agents[idx].heading = current_h;
        agents[idx].food -= actual_speed * cfg.eco.move_cost_per_unit * 1000.0;
    }

    if (agents[idx].age >= cfg.bio.max_age) { agents[idx].health -= cfg.bio.starvation_rate * 2.0; }

    // Tile Resource Regen
    let regen = cfg.world.regen_rate * clamp(1.0 - (map_heights[safe_current_idx] * 2.0), 0.05, 1.0);
    atomicAdd(&(*cell_ptr).res_value, i32(regen * 1000.0));
    atomicMin(&(*cell_ptr).res_value, i32(cfg.world.max_tile_resource * 1000.0));

    // Hebbian Update (Learning)
    let dopamine = (agents[idx].health - prev_health) * 10.0 + (agents[idx].food - prev_food) / 10000.0;
    if (abs(dopamine) > 0.01) {
        let lr = dopamine * 0.001;
        for (var p = 0u; p < 32u; p = p + 1u) {
            let h_idx = agents[idx].plastic_indices[p];
            agents[idx].plastic_weights[p] = clamp(agents[idx].plastic_weights[p] + lr * hidden1[h_idx], -1.0, 1.0);
        }
    }
}
