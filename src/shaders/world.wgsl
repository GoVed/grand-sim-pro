/*
 * Grand Sim Pro: World Environment Update Shader
 */

@compute @workgroup_size(8, 8)
fn world_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let map_w = cfg.world.map_width;
    let map_h = cfg.world.map_height;
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= map_w || y >= map_h) { return; }
    let idx = y * map_w + x;
    
    let cell_ptr = &map_cells[idx];
    
    // 1. Resource Regeneration (Multiply by 100.0 because it runs every 100 ticks)
    let regen = cfg.world.regen_rate * 100.0 * clamp(1.0 - (map_heights[idx] * 2.0), 0.05, 1.0);
    atomicAdd(&(*cell_ptr).res_value, i32(regen * 1000.0));
    atomicMin(&(*cell_ptr).res_value, i32(cfg.world.max_tile_resource * 1000.0));
}
