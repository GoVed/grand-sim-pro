/*
 * Grand Sim Pro: A high-performance GPGPU evolutionary agent simulation.
 * Part of an independent research project into emergent biological complexity.
 *
 * Copyright (C) 2026 Ved Hirenkumar Suthar
 * Licensed under the GNU General Public License v3.0 or later.
 * * This software is provided "as is", without warranty of any kind.
 * See the LICENSE file in the project root for full license details.
 */

struct Agent {
    x: f32,
    y: f32,
    heading: f32,
    speed: f32,
    hidden_count: u32,
    gender: f32,
    reproduce_desire: f32,
    attack_intent: f32,
    rest_intent: f32,
    comm1: f32,
    comm2: f32,
    comm3: f32,
    comm4: f32,
    mem1: f32,
    mem2: f32,
    mem3: f32,
    mem4: f32,
    buy_intent: f32,
    sell_intent: f32,
    ask_price: f32,
    bid_price: f32,
    wealth: f32,
    pad1: array<f32, 2>,
    w1: array<f32, 1280>,
    w2: array<f32, 1024>,
    w3: array<f32, 640>,
    food: f32,
    water: f32,
    stamina: f32,
    health: f32,
    age: f32,
    id: u32,
    gestation_timer: f32,
    is_pregnant: f32,
}

struct SimConfig {
    base_speed: f32,
    baseline_cost: f32,
    move_cost_per_unit: f32,
    climb_penalty: f32,
    base_gather_rate: f32,
    max_gather_rate: f32,
    max_tile_resource: f32,
    boat_cost: f32,
    drop_amount: f32,
    regen_rate: f32,
    max_age: f32,
    max_health: f32,
    starvation_rate: f32,
    reproduction_cost: f32,
    map_width: u32,
    map_height: u32,
    display_width: u32,
    display_height: u32,
    agent_count: u32,
    current_tick: u32,
    max_stamina: f32,
    max_water: f32,
    puberty_age: f32,
    menopause_age: f32,
    gestation_period: f32,
    tick_to_mins: f32,
    pad1: vec2<u32>,
}

struct CellState {
    res_value: f32,
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
    avg_ask: f32,
    avg_bid: f32,
    market_food: f32,
    market_wealth: f32,
    pad1: array<f32, 1>,
}

@group(0) @binding(0) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(1) var<storage, read> map_heights: array<f32>;
@group(0) @binding(2) var<storage, read_write> map_cells: array<CellState>;
@group(0) @binding(3) var<uniform> cfg: SimConfig;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&agents)) { return; }

    var agent = agents[idx];
    
    if (agent.health <= 0.0) {
        return; // Agent is dead, skip processing
    }
    agent.age = agent.age + 1.0;
    
    let map_width = cfg.map_width;
    let map_w_f32 = f32(cfg.map_width);
    let map_h_f32 = f32(cfg.map_height);
    let max_idx = cfg.map_width * cfg.map_height - 1u;
    
    let current_idx = u32(agent.y) * map_width + u32(agent.x);
    let safe_current_idx = clamp(current_idx, 0u, max_idx);
    
    var local_cell = map_cells[safe_current_idx];

    // Generate chaos/noise using the agent's spatial position
    let pseudo_rand = fract(sin(dot(vec2<f32>(agent.x, agent.y), vec2<f32>(12.9898, 78.233))) * 43758.5453);

    // Environment & Vision Pre-Calculation
    let current_height = map_heights[safe_current_idx];
    
    let season_time = f32(cfg.current_tick) / 100000.0 * 6.28318;
    let season_sine = sin(season_time);
    let dist_from_equator = abs(agent.y - map_h_f32 / 2.0) / (map_h_f32 / 2.0);
    let base_temp = (1.0 - dist_from_equator * 2.0) - max(0.0, current_height * 2.0);
    let local_temp = base_temp + season_sine * 0.5;

    let look_dist = 10.0;
    let look_x = agent.x + cos(agent.heading) * look_dist;
    let look_y = agent.y + sin(agent.heading) * look_dist;
    var look_wrap_x = look_x % map_w_f32; if (look_wrap_x < 0.0) { look_wrap_x = look_wrap_x + map_w_f32; }
    var look_wrap_y = look_y % map_h_f32; if (look_wrap_y < 0.0) { look_wrap_y = look_wrap_y + map_h_f32; }
    let look_idx = clamp(u32(look_wrap_y) * map_width + u32(look_wrap_x), 0u, max_idx);
    let look_cell = map_cells[look_idx];
    let look_height = map_heights[look_idx];

    // 1. Neural Net Processing
    var inputs = array<f32, 40>(
        1.0, agent.x / map_w_f32, agent.y / map_h_f32, local_cell.res_value / 1000.0,
        local_cell.population,
        local_cell.avg_speed + (pseudo_rand * 0.1 - 0.05), // Some noise injected
        local_cell.avg_share + (pseudo_rand * 0.1 - 0.05),
        local_cell.avg_reproduce + (pseudo_rand * 0.1 - 0.05),
        local_cell.avg_aggression,
        local_cell.avg_pregnancy,
        local_cell.avg_turn,
        local_cell.avg_rest,
        local_cell.comm1, local_cell.comm2, local_cell.comm3, local_cell.comm4,
        agent.health / cfg.max_health,
        agent.food / cfg.boat_cost, 
        agent.water / cfg.max_water,
        agent.stamina / cfg.max_stamina,
        agent.age / cfg.max_age,
        agent.gender,
        look_cell.res_value / 1000.0,
        look_height,
        look_cell.population,
        local_temp,
        season_sine,
        agent.is_pregnant, 
        (agent.food + agent.water) / 250.0,
        local_cell.population / 15.0,
        agent.mem1, agent.mem2, agent.mem3, agent.mem4,
        agent.wealth / cfg.boat_cost,
        local_cell.avg_ask / 10.0,
        local_cell.avg_bid / 10.0,
        0.0, 0.0, 0.0
    );

    var hidden1 = array<f32, 32>();
    for (var h1 = 0u; h1 < 32u; h1 = h1 + 1u) {
        var sum = 0.0;
        if (h1 < agent.hidden_count) {
            for (var i = 0u; i < 40u; i = i + 1u) {
                sum = sum + inputs[i] * agent.w1[h1 * 40u + i];
            }
            hidden1[h1] = tanh(sum);
        } else {
            hidden1[h1] = 0.0;
        }
    }

    var hidden2 = array<f32, 32>();
    for (var h2 = 0u; h2 < 32u; h2 = h2 + 1u) {
        var sum = 0.0;
        if (h2 < agent.hidden_count) {
            for (var h1 = 0u; h1 < 32u; h1 = h1 + 1u) {
                if (h1 < agent.hidden_count) {
                    sum = sum + hidden1[h1] * agent.w2[h1 * 32u + h2];
                }
            }
            hidden2[h2] = tanh(sum);
        } else {
            hidden2[h2] = 0.0;
        }
    }

    var outputs = array<f32, 20>();
    for (var o = 0u; o < 20u; o = o + 1u) {
        var sum = 0.0;
        for (var h2 = 0u; h2 < 32u; h2 = h2 + 1u) {
            if (h2 < agent.hidden_count) {
                sum = sum + hidden2[h2] * agent.w3[h2 * 20u + o];
            }
        }
        outputs[o] = tanh(sum);
    }

    let turn_intent = outputs[0];
    agent.heading = agent.heading + turn_intent * 0.5;
    let speed_intent = clamp(outputs[1] * 0.5 + 0.5, 0.0, 1.0); // Normalize 0 to 1
    agent.reproduce_desire = clamp(outputs[3] * 0.5 + 0.5, 0.0, 1.0);
    agent.attack_intent = clamp(outputs[4] * 0.5 + 0.5, 0.0, 1.0);
    let rest_intent = clamp(outputs[5] * 0.5 + 0.5, 0.0, 1.0);
    agent.comm1 = clamp(outputs[6], -1.0, 1.0);
    agent.comm2 = clamp(outputs[7], -1.0, 1.0);
    agent.comm3 = clamp(outputs[8], -1.0, 1.0);
    agent.comm4 = clamp(outputs[9], -1.0, 1.0);
    let learn_intent = outputs[10];
    agent.mem1 = outputs[11];
    agent.mem2 = outputs[12];
    agent.mem3 = outputs[13];
    agent.mem4 = outputs[14];
    agent.buy_intent = clamp(outputs[15] * 0.5 + 0.5, 0.0, 1.0);
    agent.sell_intent = clamp(outputs[16] * 0.5 + 0.5, 0.0, 1.0);
    agent.ask_price = abs(outputs[17]) * 10.0;
    agent.bid_price = abs(outputs[18]) * 10.0;

    // Hebbian Learning: dynamically adjust weights during their lifetime!
    if (abs(learn_intent) > 0.01) {
        let lr = learn_intent * 0.001; 
        
        for (var h1 = 0u; h1 < 32u; h1 = h1 + 1u) {
            if (h1 < agent.hidden_count) {
                for (var i = 0u; i < 40u; i = i + 1u) {
                    agent.w1[h1 * 40u + i] = clamp(agent.w1[h1 * 40u + i] + lr * inputs[i] * hidden1[h1], -2.0, 2.0);
                }
            }
        }
        for (var h2 = 0u; h2 < 32u; h2 = h2 + 1u) {
            if (h2 < agent.hidden_count) {
                for (var h1 = 0u; h1 < 32u; h1 = h1 + 1u) {
                    if (h1 < agent.hidden_count) {
                        agent.w2[h1 * 32u + h2] = clamp(agent.w2[h1 * 32u + h2] + lr * hidden1[h1] * hidden2[h2], -2.0, 2.0);
                    }
                }
            }
        }
        for (var o = 0u; o < 20u; o = o + 1u) {
            for (var h2 = 0u; h2 < 32u; h2 = h2 + 1u) {
                if (h2 < agent.hidden_count) {
                    agent.w3[h2 * 20u + o] = clamp(agent.w3[h2 * 20u + o] + lr * hidden2[h2] * outputs[o], -2.0, 2.0);
                }
            }
        }
    }
    
    var base_speed = speed_intent * cfg.base_speed;
    var resting = false;
    if (rest_intent > 0.5) {
        resting = true;
        base_speed = 0.0;
        agent.stamina = min(agent.stamina + 2.0, cfg.max_stamina);
    }

    // Evaluate intended destination to gauge the slope
    let next_x = agent.x + cos(agent.heading) * base_speed;
    let next_y = agent.y + sin(agent.heading) * base_speed;

    var wrap_x = next_x % map_w_f32; if (wrap_x < 0.0) { wrap_x = wrap_x + map_w_f32; }
    var wrap_y = next_y % map_h_f32; if (wrap_y < 0.0) { wrap_y = wrap_y + map_h_f32; }

    let next_idx = u32(wrap_y) * map_width + u32(wrap_x);
    let safe_idx = clamp(next_idx, 0u, max_idx);
    let next_height = map_heights[safe_idx];

    // Height Factor: Uphill slows down, downhill speeds up
    let slope = next_height - current_height;
    let height_multiplier = 1.0 - clamp(slope * cfg.climb_penalty, -0.5, 0.9);
    
    let total_weight = agent.food + agent.water;
    let encumbrance_mult = clamp(1.0 - (total_weight / 250.0), 0.1, 1.0);
    let crowding_mult = clamp(1.0 - (local_cell.population / 15.0), 0.1, 1.0);
    
    var actual_speed = base_speed * height_multiplier * encumbrance_mult * crowding_mult;

    if (agent.gestation_timer > 0.0) {
        agent.gestation_timer = agent.gestation_timer - 1.0;
        if (agent.gestation_timer <= 0.0) {
            agent.is_pregnant = 0.0;
        }
    }

    var preg_mult = 1.0;
    if (agent.is_pregnant > 0.5) {
        actual_speed = actual_speed * 0.7; // 30% slower when carrying child
        preg_mult = 1.5; // Eat/drink 50% more
    }
    
    if (!resting) {
        agent.stamina = agent.stamina - (actual_speed * 0.05);
        if (agent.stamina <= 0.0) {
            agent.stamina = 0.0;
            resting = true;
        }
    }

    // Recalculate true next position with slope-adjusted speed
    let act_spd = select(actual_speed, 0.0, resting);
    let final_next_x = agent.x + cos(agent.heading) * act_spd;
    let final_next_y = agent.y + sin(agent.heading) * act_spd;
    var final_wrap_x = final_next_x % map_w_f32; if (final_wrap_x < 0.0) { final_wrap_x = final_wrap_x + map_w_f32; }
    var final_wrap_y = final_next_y % map_h_f32; if (final_wrap_y < 0.0) { final_wrap_y = final_wrap_y + map_h_f32; }

    let final_idx = u32(final_wrap_y) * map_width + u32(final_wrap_x);
    let safe_final_idx = clamp(final_idx, 0u, max_idx);
    let final_height = map_heights[safe_final_idx];

    // Resource & Terrain Mechanics
    if (current_height <= 0.05) { agent.water = min(agent.water + 5.0, cfg.max_water); }
    
    var rest_mult = 1.0; if (resting) { rest_mult = 0.5; }
    agent.water = agent.water - (cfg.baseline_cost * rest_mult * preg_mult);
    let cold_penalty = max(0.0, -local_temp) * 0.1;
    agent.food = agent.food - (cfg.baseline_cost * rest_mult * preg_mult) - cold_penalty;
    
    if (agent.food < 0.0) { agent.food = 0.0; agent.health = agent.health - cfg.starvation_rate; }
    if (agent.water < 0.0) { agent.water = 0.0; agent.health = agent.health - cfg.starvation_rate; }
    if (agent.food > cfg.baseline_cost && agent.water > cfg.baseline_cost && agent.health < cfg.max_health) {
        agent.health = min(agent.health + (cfg.starvation_rate * 2.0), cfg.max_health);
        agent.food = agent.food - cfg.baseline_cost; 
        agent.water = agent.water - cfg.baseline_cost; 
    }

    if (current_height >= 0.0) {
        let drop_desire = outputs[2];
        
        if (agent.attack_intent > 0.5 && local_cell.population > 1.0 && !resting && agent.age > cfg.puberty_age) {
            // Steal directly from abstract population
            let steal_amount = min((local_cell.population - 1.0) * 0.5, 5.0);
            agent.wealth = min(agent.wealth + steal_amount * 5.0, cfg.boat_cost); // Mugging gives cash
            if (local_cell.avg_aggression > 0.5) { agent.health = agent.health - 2.0; }
        } else if (drop_desire > 0.5 && agent.food > (cfg.drop_amount + 100.0)) {
            // Community Logic: Drop resources to the current tile
            agent.food = agent.food - cfg.drop_amount;
            local_cell.res_value = min(local_cell.res_value + cfg.drop_amount, cfg.max_tile_resource);
        } else if (!resting) {
            if (local_cell.res_value > 0.1) {
                let max_mult = (cfg.max_gather_rate / cfg.base_gather_rate) - 1.0;
                let tool_multiplier = 1.0 + min(sqrt(agent.food) * 0.1, max_mult);
                
                // Newborns are weak and can only gather a fraction of what adults can
                let maturity = clamp(agent.age / cfg.puberty_age, 0.2, 1.0); 
                let gathered = min(cfg.base_gather_rate * tool_multiplier * maturity, local_cell.res_value);
                agent.food = agent.food + gathered;
                local_cell.res_value = local_cell.res_value - gathered;
            }
        }
        
        // --- The Virtual Trading Market ---
        local_cell.avg_ask = mix(local_cell.avg_ask, agent.ask_price, 0.1);
        local_cell.avg_bid = mix(local_cell.avg_bid, agent.bid_price, 0.1);
        
        if (agent.buy_intent > 0.5 && agent.wealth >= local_cell.avg_ask && local_cell.market_food >= 1.0 && local_cell.avg_ask > 0.0) {
            agent.wealth = agent.wealth - local_cell.avg_ask;
            agent.food = agent.food + 1.0;
            local_cell.market_wealth = local_cell.market_wealth + local_cell.avg_ask;
            local_cell.market_food = local_cell.market_food - 1.0;
        } else if (agent.sell_intent > 0.5 && agent.food >= 1.0 && local_cell.market_wealth >= local_cell.avg_bid && local_cell.avg_bid > 0.0) {
            agent.food = agent.food - 1.0;
            agent.wealth = agent.wealth + local_cell.avg_bid;
            local_cell.market_food = local_cell.market_food + 1.0;
            local_cell.market_wealth = local_cell.market_wealth - local_cell.avg_bid;
        }
        
        local_cell.res_value = min(local_cell.res_value + cfg.regen_rate, cfg.max_tile_resource);
        
        // Mix our intent into the cell's pheromone trace
        local_cell.population = local_cell.population * 0.99 + 1.0; // Automatically decays to zero if abandoned
        local_cell.avg_speed = mix(local_cell.avg_speed, speed_intent, 0.1);
        local_cell.avg_share = mix(local_cell.avg_share, drop_desire, 0.1);
        local_cell.avg_reproduce = mix(local_cell.avg_reproduce, agent.reproduce_desire, 0.1);
        local_cell.avg_aggression = mix(local_cell.avg_aggression, agent.attack_intent, 0.1);
        local_cell.avg_pregnancy = mix(local_cell.avg_pregnancy, agent.is_pregnant, 0.1);
        local_cell.avg_turn = mix(local_cell.avg_turn, turn_intent, 0.1);
        local_cell.avg_rest = mix(local_cell.avg_rest, rest_intent, 0.1);
        local_cell.comm1 = mix(local_cell.comm1, agent.comm1, 0.1);
        local_cell.comm2 = mix(local_cell.comm2, agent.comm2, 0.1);
        local_cell.comm3 = mix(local_cell.comm3, agent.comm3, 0.1);
        local_cell.comm4 = mix(local_cell.comm4, agent.comm4, 0.1);
    }
    map_cells[safe_current_idx] = local_cell;

    // Calculate climbing exertion: positive slopes drastically increase the movement cost
    let climb_penalty = max(0.0, slope) * 20.0; 
    let move_cost = (act_spd * 0.1) * (1.0 + climb_penalty);
    let is_water = final_height < 0.0;
    let boat_threshold = 80.0; 

    // Water travel requires passing the boat threshold (or already being in the ocean with savings)
    let can_enter_water = agent.wealth >= boat_threshold || (current_height < 0.0 && agent.wealth > 0.0);

    if (is_water && !can_enter_water) {
        agent.heading = agent.heading + 3.14159; // Turn around, can't afford a boat
        agent.speed = 0.0;
    } else {
        if (agent.food >= move_cost) {
            agent.x = final_wrap_x;
            agent.y = final_wrap_y;
            agent.food = agent.food - move_cost;
            agent.speed = act_spd;
        } else {
            agent.speed = 0.0; // Exhausted (must rest to gather)
        }
    }
    
    if (agent.age > cfg.max_age) {
        agent.health = agent.health - (cfg.starvation_rate * 2.0); // Rapid health decline
    }
    agents[idx] = agent;
}