struct Agent {
    x: f32,
    y: f32,
    heading: f32,
    speed: f32,
    hidden_count: u32,
    gender: f32,
    reproduce_desire: f32,
    pad: f32,
    w1: array<f32, 384>,
    w2: array<f32, 128>,
    inventory: f32,
    health: f32,
    age: f32,
    pad2: f32,
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
    pad: vec2<f32>,
}

struct CellState {
    res_value: f32,
    population: f32,
    avg_speed: f32,
    avg_share: f32,
    avg_reproduce: f32,
    pad: vec3<f32>,
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
    
    let map_width = 800u;
    let current_idx = u32(agent.y) * map_width + u32(agent.x);
    let safe_current_idx = clamp(current_idx, 0u, 479999u);
    
    var local_cell = map_cells[safe_current_idx];

    // Generate chaos/noise using the agent's spatial position
    let pseudo_rand = fract(sin(dot(vec2<f32>(agent.x, agent.y), vec2<f32>(12.9898, 78.233))) * 43758.5453);

    // 1. Neural Net Processing
    var inputs = array<f32, 12>(
        1.0, agent.x / 800.0, agent.y / 600.0, local_cell.res_value / 1000.0,
        local_cell.population,
        local_cell.avg_speed + (pseudo_rand * 0.1 - 0.05), // Some noise injected
        local_cell.avg_share + (pseudo_rand * 0.1 - 0.05),
        local_cell.avg_reproduce + (pseudo_rand * 0.1 - 0.05),
        agent.health / cfg.max_health,
        agent.inventory / cfg.boat_cost, // Normalize by boat cost since it's the biggest target
        agent.age / cfg.max_age,
        agent.gender
    );

    var hidden = array<f32, 32>();
    for (var h = 0u; h < agent.hidden_count; h = h + 1u) {
        var sum = 0.0;
        for (var i = 0u; i < 12u; i = i + 1u) {
            sum = sum + inputs[i] * agent.w1[h * 12u + i];
        }
        hidden[h] = tanh(sum);
    }

    var outputs = array<f32, 4>();
    for (var o = 0u; o < 4u; o = o + 1u) {
        var sum = 0.0;
        for (var h = 0u; h < agent.hidden_count; h = h + 1u) {
            sum = sum + hidden[h] * agent.w2[h * 4u + o];
        }
        outputs[o] = tanh(sum);
    }

    agent.heading = agent.heading + outputs[0] * 0.5;
    let speed_intent = clamp(outputs[1] * 0.5 + 0.5, 0.0, 1.0); // Normalize 0 to 1
    let base_speed = speed_intent * cfg.base_speed;
    agent.reproduce_desire = clamp(outputs[3] * 0.5 + 0.5, 0.0, 1.0);

    // Get current elevation
    let current_height = map_heights[safe_current_idx];

    // Evaluate intended destination to gauge the slope
    let next_x = agent.x + cos(agent.heading) * base_speed;
    let next_y = agent.y + sin(agent.heading) * base_speed;

    var wrap_x = next_x % 800.0; if (wrap_x < 0.0) { wrap_x = wrap_x + 800.0; }
    var wrap_y = next_y % 600.0; if (wrap_y < 0.0) { wrap_y = wrap_y + 600.0; }

    let next_idx = u32(wrap_y) * map_width + u32(wrap_x);
    let safe_idx = clamp(next_idx, 0u, 479999u); // Safe bound (800 * 600 - 1)
    let next_height = map_heights[safe_idx];

    // Height Factor: Uphill slows down, downhill speeds up
    let slope = next_height - current_height;
    let height_multiplier = 1.0 - clamp(slope * cfg.climb_penalty, -0.5, 0.9);
    let actual_speed = base_speed * height_multiplier;

    // Recalculate true next position with slope-adjusted speed
    let final_next_x = agent.x + cos(agent.heading) * actual_speed;
    let final_next_y = agent.y + sin(agent.heading) * actual_speed;
    var final_wrap_x = final_next_x % 800.0; if (final_wrap_x < 0.0) { final_wrap_x = final_wrap_x + 800.0; }
    var final_wrap_y = final_next_y % 600.0; if (final_wrap_y < 0.0) { final_wrap_y = final_wrap_y + 600.0; }

    let final_idx = u32(final_wrap_y) * map_width + u32(final_wrap_x);
    let safe_final_idx = clamp(final_idx, 0u, 479999u);
    let final_height = map_heights[safe_final_idx];

    // Resource & Terrain Mechanics
    agent.inventory = agent.inventory - cfg.baseline_cost;
    if (agent.inventory < 0.0) {
        agent.inventory = 0.0;
        agent.health = agent.health - cfg.starvation_rate;
    } else if (agent.inventory > cfg.baseline_cost && agent.health < cfg.max_health) {
        // Heal if we have enough resources
        agent.health = min(agent.health + cfg.starvation_rate, cfg.max_health);
        agent.inventory = agent.inventory - cfg.baseline_cost; 
    }

    if (current_height >= 0.0) {
        let drop_desire = outputs[2];
        if (drop_desire > 0.5 && agent.inventory > (cfg.drop_amount + 100.0)) {
            // Community Logic: Drop resources to the current tile
            agent.inventory = agent.inventory - cfg.drop_amount;
            local_cell.res_value = min(local_cell.res_value + cfg.drop_amount, cfg.max_tile_resource);
        } else {
            if (local_cell.res_value > 0.1) {
                let max_mult = (cfg.max_gather_rate / cfg.base_gather_rate) - 1.0;
                let tool_multiplier = 1.0 + min(sqrt(agent.inventory) * 0.1, max_mult);
                
                let gathered = min(cfg.base_gather_rate * tool_multiplier, local_cell.res_value);
                agent.inventory = agent.inventory + gathered;
                local_cell.res_value = local_cell.res_value - gathered;
            }
        }
        
        local_cell.res_value = min(local_cell.res_value + cfg.regen_rate, cfg.max_tile_resource);
        
        // Mix our intent into the cell's pheromone trace
        local_cell.population = local_cell.population * 0.99 + 1.0; // Automatically decays to zero if abandoned
        local_cell.avg_speed = mix(local_cell.avg_speed, speed_intent, 0.1);
        local_cell.avg_share = mix(local_cell.avg_share, drop_desire, 0.1);
        local_cell.avg_reproduce = mix(local_cell.avg_reproduce, agent.reproduce_desire, 0.1);
    }
    map_cells[safe_current_idx] = local_cell;

    // Calculate climbing exertion: positive slopes drastically increase the movement cost
    let climb_penalty = max(0.0, slope) * 20.0; 
    let move_cost = (actual_speed * 0.1) * (1.0 + climb_penalty);
    let is_water = final_height < 0.0;
    let boat_threshold = 80.0; 

    // Water travel requires passing the boat threshold (or already being in the ocean with savings)
    let can_enter_water = agent.inventory >= boat_threshold || (current_height < 0.0 && agent.inventory > 0.0);

    if (is_water && !can_enter_water) {
        agent.heading = agent.heading + 3.14159; // Turn around, can't afford a boat
        agent.speed = 0.0;
    } else {
        if (agent.inventory >= move_cost) {
            agent.x = final_wrap_x;
            agent.y = final_wrap_y;
            agent.inventory = agent.inventory - move_cost;
            agent.speed = actual_speed;
        } else {
            agent.speed = 0.0; // Exhausted (must rest to gather)
        }
    }
    
    if (agent.age > cfg.max_age) {
        agent.health = agent.health - (cfg.starvation_rate * 2.0); // Rapid health decline
    }
    agents[idx] = agent;
}