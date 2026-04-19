/*
 * Grand Sim Pro: World Environment Update Shader
 */

@compute @workgroup_size(64)
fn world_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let map_w = cfg.world.map_width;
    let map_h = cfg.world.map_height;
    let idx = global_id.x;
    
    if (idx >= map_w * map_h) { return; }
    
    let cell_ptr = &map_cells[idx];
    
    // 1. Resource Regeneration (Correct: once per tile per tick)
    let regen = cfg.world.regen_rate * clamp(1.0 - (map_heights[idx] * 2.0), 0.05, 1.0);
    atomicAdd(&(*cell_ptr).res_value, i32(regen * 1000.0));
    atomicMin(&(*cell_ptr).res_value, i32(cfg.world.max_tile_resource * 1000.0));
    
    // 2. Population Reset (Standard: population is count of agents in current tick)
    atomicStore(&(*cell_ptr).population, 0);
    atomicStore(&(*cell_ptr).adult_count, 0);
}
