struct Agent {
    x: f32,
    y: f32,
    heading: f32,
    speed: f32,
    hidden_count: u32,
    w1: array<f32, 64>,
    w2: array<f32, 32>,
    inventory: f32,
    pad: array<f32, 2>,
}

@group(0) @binding(0) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(1) var<storage, read> map_heights: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&agents)) { return; }

    var agent = agents[idx];
    let map_width = 800u;

    // 1. Neural Net Processing
    var inputs = array<f32, 4>(1.0, agent.x / 800.0, agent.y / 600.0, 0.5);

    var hidden = array<f32, 16>();
    for (var h = 0u; h < agent.hidden_count; h = h + 1u) {
        var sum = 0.0;
        for (var i = 0u; i < 4u; i = i + 1u) {
            sum = sum + inputs[i] * agent.w1[h * 4u + i];
        }
        hidden[h] = tanh(sum);
    }

    var outputs = array<f32, 2>();
    for (var o = 0u; o < 2u; o = o + 1u) {
        var sum = 0.0;
        for (var h = 0u; h < agent.hidden_count; h = h + 1u) {
            sum = sum + hidden[h] * agent.w2[h * 2u + o];
        }
        outputs[o] = tanh(sum);
    }

    agent.heading = agent.heading + outputs[0] * 0.5;
    let base_speed = clamp(outputs[1] * 1.5 + 1.5, 0.1, 3.0);

    // Get current elevation
    let current_idx = u32(agent.y) * map_width + u32(agent.x);
    let safe_current_idx = clamp(current_idx, 0u, 479999u);
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
    let height_multiplier = 1.0 - clamp(slope * 4.0, -0.5, 0.8);
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
    if (current_height >= 0.0) {
        agent.inventory = min(agent.inventory + 0.1, 150.0); // Passive gathering on land
    }

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
    agents[idx] = agent;
}