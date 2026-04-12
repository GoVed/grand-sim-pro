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