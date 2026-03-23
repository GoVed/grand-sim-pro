mod agent;
mod environment;
mod simulation;
mod gpu_engine;
mod config;

use macroquad::prelude::*;
use simulation::SimulationManager;
use gpu_engine::GpuEngine;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use std::collections::HashMap;

// Translates 10-minute ticks into realistic Years and Months
fn format_time(ticks: u64, tick_to_mins: f32) -> String {
    let total_mins = ticks as f64 * tick_to_mins as f64;
    let total_days = total_mins / (60.0 * 24.0);
    let years = (total_days / 365.0).floor() as u32;
    let months = ((total_days % 365.0) / 30.0).floor() as u32;
    if years > 0 {
        format!("{}y {}m", years, months)
    } else {
        format!("{}m", months)
    }
}

// Standalone config loader so we can establish window dimensions before the async engine boots
fn load_config() -> config::SimConfig {
    if let Ok(data) = std::fs::read_to_string("sim_config.json") {
        if let Ok(cfg) = serde_json::from_str(&data) {
            return cfg;
        }
    }
    let default_cfg = config::SimConfig::default();
    std::fs::write("sim_config.json", serde_json::to_string_pretty(&default_cfg).unwrap()).ok();
    default_cfg
}

fn window_conf() -> Conf {
    let cfg = load_config();
    Conf {
        window_title: "World Sim (Native)".to_owned(),
        window_width: cfg.display_width as i32,
        window_height: cfg.display_height as i32,
        ..Default::default()
    }
}

#[derive(PartialEq, Clone, Copy)]
enum VisualMode { Default, Resources, Age, Gender, Pregnancy }

#[derive(PartialEq, Clone, Copy)]
enum SortCol { Index, Age, Health, Food, Gender, Outputs }

struct AgentRenderData {
    x: f32,
    y: f32,
    food: f32,
    health: f32,
    age: f32,
    gender: f32,
    is_pregnant: f32,
}

struct SharedData {
    sim: SimulationManager,
    config: config::SimConfig,
    is_paused: bool,
    restart_message_active: bool,
    ticks_per_loop: usize,
    total_ticks: u64,
    last_compute_time_ms: u128,
}

#[macroquad::main(window_conf)]
async fn main() {
    let loaded_config = load_config();
    let map_width = loaded_config.map_width;
    let map_height = loaded_config.map_height;
    let agent_count = loaded_config.agent_count;

    let shared_data = Arc::new(Mutex::new(SharedData {
        sim: SimulationManager::new(map_width, map_height, 1234, agent_count, &loaded_config),
        config: loaded_config,
        is_paused: false,
        restart_message_active: false,
        ticks_per_loop: 2000,
        total_ticks: 0,
        last_compute_time_ms: 0,
    }));

    let sim_thread_data = shared_data.clone();
    
    let initial_agents = sim_thread_data.lock().unwrap().sim.agents.clone();
    let height_map = sim_thread_data.lock().unwrap().sim.env.height_map.clone();
    let init_cells = sim_thread_data.lock().unwrap().sim.env.map_cells.clone();
    let gpu = Arc::new(GpuEngine::new(&initial_agents, &height_map, &init_cells, &loaded_config));

    // Spawn background simulation thread
    thread::spawn(move || loop {
        let mut ran_ticks = false;
        {
            let mut data = sim_thread_data.lock().unwrap();
            if !data.is_paused {
                let start = Instant::now();
                
                data.config.current_tick = data.total_ticks as u32;
                gpu.update_config(&data.config);

                // Dispatch heavy work entirely to the AMD GPU
                gpu.compute_ticks(data.ticks_per_loop);
                
                // Download updated positions and map state precisely when needed
                let (agents, cells) = gpu.fetch_state();
                data.sim.agents = agents;
                data.sim.env.map_cells = cells;

                // --- CPU Gestation & Birth Pass ---
                let mut living_ids = std::collections::HashSet::new();
                let mut cell_occupants: HashMap<usize, Vec<usize>> = HashMap::new();
                let mut dead_indices = Vec::new();
                
                let puberty = data.config.puberty_age;
                let menopause = data.config.menopause_age;

                for i in 0..data.sim.agents.len() {
                    let a = &data.sim.agents[i];
                    if a.health <= 0.0 {
                        dead_indices.push(i);
                    } else {
                        living_ids.insert(a.id);
                        let map_w = data.config.map_width as usize;
                        let map_h = data.config.map_height as usize;
                        let idx = (a.y as usize).clamp(0, map_h.saturating_sub(1)) * map_w + (a.x as usize).clamp(0, map_w.saturating_sub(1));
                        cell_occupants.entry(idx).or_default().push(i);
                    }
                }

                // Clean up miscarriages if mother dies
                data.sim.pending_births.retain(|id, _| living_ids.contains(id));

                let mut modifications = false;

                let mut births_to_process = Vec::new();
                for mother in &data.sim.agents {
                    if mother.gestation_timer <= 0.0 && data.sim.pending_births.contains_key(&mother.id) {
                        births_to_process.push((mother.id, mother.x, mother.y));
                    }
                }

                for (mother_id, mx, my) in births_to_process {
                    if let Some(dead_idx) = dead_indices.pop() {
                        if let Some(mut child) = data.sim.pending_births.remove(&mother_id) {
                            child.x = mx;
                            child.y = my;
                            data.sim.agents[dead_idx] = child;
                            modifications = true;
                        }
                    }
                }

                // --- CPU Sexual Genetics Pass ---
                let reproduction_cost = data.config.reproduction_cost;
                
                for (_cell_idx, occupants) in cell_occupants {
                    if occupants.len() < 2 { continue; } // Need at least 2 people to mate

                    let mut males = Vec::new();
                    let mut females = Vec::new();

                    for &idx in &occupants {
                        let a = &data.sim.agents[idx];
                        let is_mature = a.age >= puberty && a.age <= menopause;
                        if is_mature && a.reproduce_desire > 0.5 && a.food >= reproduction_cost / 2.0 && a.health > data.config.max_health * 0.5 {
                            if a.gender > 0.5 { males.push(idx); } else if a.gestation_timer <= 0.0 && !data.sim.pending_births.contains_key(&a.id) { females.push(idx); }
                        }
                    }

                    // Pair them up
                    while let (Some(m_idx), Some(f_idx)) = (males.pop(), females.pop()) {
                        let mut p1 = data.sim.agents[m_idx].clone();
                        let mut p2 = data.sim.agents[f_idx].clone();
                        
                        let child = crate::agent::Person::reproduce_sexual(&mut p1, &mut p2, reproduction_cost);
                        
                        data.sim.agents[m_idx] = p1; // Deduct money from father
                        
                        p2.is_pregnant = 1.0;
                        p2.gestation_timer = data.config.gestation_period;
                        data.sim.agents[f_idx] = p2;
                        
                        data.sim.pending_births.insert(p2.id, child); // Mother physically carries the genetic material
                        modifications = true;
                    }
                }

                // Send mutated population back to VRAM
                if modifications {
                    gpu.update_agents(&data.sim.agents);
                }
                
                data.total_ticks += data.ticks_per_loop as u64;
                data.last_compute_time_ms = start.elapsed().as_millis();
                ran_ticks = true;

                // --- Auto-Restart Logic ---
                let living_count = data.sim.agents.iter().filter(|a| a.health > 0.0).count();
                if living_count == 0 {
                    data.restart_message_active = true;
                    drop(data); // Release lock instantly so the UI can render the message
                    thread::sleep(Duration::from_millis(1000));
                    
                    let mut data = sim_thread_data.lock().unwrap();
                    
                    // Sort agents by age descending to find the 8 longest-living survivors
                    data.sim.agents.sort_by(|a, b| b.age.partial_cmp(&a.age).unwrap_or(std::cmp::Ordering::Equal));
                    
                    let target_pop = data.config.agent_count as usize;
                    let map_w = data.config.map_width as f32;
                    let map_h = data.config.map_height as f32;
                    let mut new_population = Vec::with_capacity(target_pop);
                    let founders_count = 8.min(data.sim.agents.len());
                    let children_per_founder = target_pop / founders_count.max(1);
                    
                    for i in 0..founders_count {
                        let founder = &data.sim.agents[i];
                        for _ in 0..children_per_founder {
                            new_population.push(founder.clone_as_descendant(map_w, map_h));
                        }
                    }
                    while new_population.len() < target_pop { new_population.push(data.sim.agents[0].clone_as_descendant(map_w, map_h)); }
                    data.sim.agents = new_population;
                    
                    let max_res = data.config.max_tile_resource;
                    for i in 0..data.sim.env.height_map.len() {
                        let h = data.sim.env.height_map[i];
                        data.sim.env.map_cells[i].res_value = if h >= 0.0 { max_res } else { 0.0 };
                        data.sim.env.map_cells[i].population = 0.0;
                        data.sim.env.map_cells[i].avg_speed = 0.0;
                        data.sim.env.map_cells[i].avg_share = 0.0;
                        data.sim.env.map_cells[i].avg_reproduce = 0.0;
                        data.sim.env.map_cells[i].avg_aggression = 0.0;
                        data.sim.env.map_cells[i].avg_pregnancy = 0.0;
                        data.sim.env.map_cells[i].avg_turn = 0.0;
                        data.sim.env.map_cells[i].avg_rest = 0.0;
                        data.sim.env.map_cells[i].comm1 = 0.0;
                        data.sim.env.map_cells[i].comm2 = 0.0;
                        data.sim.env.map_cells[i].comm3 = 0.0;
                        data.sim.env.map_cells[i].comm4 = 0.0;
                        data.sim.env.map_cells[i].pad1 = [0.0; 3];
                    }
                    data.total_ticks = 0;
                    data.restart_message_active = false;
                    gpu.update_agents(&data.sim.agents);
                    gpu.update_cells(&data.sim.env.map_cells);
                }
            }
        }
        
        if !ran_ticks {
            thread::sleep(Duration::from_millis(16)); // Prevent CPU hogging when paused
        } else {
            thread::sleep(Duration::from_millis(1)); // Force a context switch so the UI can lock the Mutex
        }
    });
    
    let mut image = Image::gen_image_color(map_width as u16, map_height as u16, BLANK);
    let texture = Texture2D::from_image(&image);
    
    let mut zoom = 1.0f32;
    let mut offset_x = 0.0f32;
    let mut offset_y = 0.0f32;
    let mut last_mouse = mouse_position();

    let mut local_agent_coords: Vec<AgentRenderData> = Vec::with_capacity(agent_count as usize);
    let mut local_map_data = Vec::with_capacity((map_width * map_height * 4) as usize);
    
    let mut paused = false;
    let mut speed = 20;
    let mut ticks = 0;
    let mut compute_time = 0;
    let mut pop_count = 0;
    let mut restart_msg = false;

    let mut pending_pause_toggle = false;
    let mut show_visuals_panel = false;
    let mut current_visual_mode = VisualMode::Default;
    let mut pending_speed_change: i32 = 0;

    let mut frame_times: Vec<f32> = Vec::with_capacity(300);
    
    let mut show_inspector = false;
    let mut sort_col = SortCol::Food;
    let mut sort_desc = true;
    let mut inspector_scroll: usize = 0;
    let mut selected_agent: Option<crate::agent::Person> = None;
    let mut inspector_agents: Vec<(usize, crate::agent::Person)> = Vec::with_capacity(agent_count as usize);

    loop {
        // Register UI inputs instantly
        if is_key_pressed(KeyCode::Space) { pending_pause_toggle = !pending_pause_toggle; }
        if is_key_pressed(KeyCode::R) { show_visuals_panel = !show_visuals_panel; }
        if is_key_pressed(KeyCode::Up) { pending_speed_change += 1; }
        if is_key_pressed(KeyCode::Down) { pending_speed_change -= 1; }
        if is_key_pressed(KeyCode::Tab) { 
            show_inspector = !show_inspector; 
            selected_agent = None; // Reset detail view when closing
        }

        let (_, mouse_wheel_y) = mouse_wheel();

        let (mx, my) = mouse_position();
        let left_clicked = is_mouse_button_pressed(MouseButton::Left);
        
        if !show_inspector {
            if mouse_wheel_y > 0.0 { zoom *= 1.1; }
            if mouse_wheel_y < 0.0 { zoom *= 0.9; }
            
            if is_mouse_button_down(MouseButton::Left) {
                let mut hit_ui = false;
                if show_visuals_panel && mx > 280.0 && mx < 440.0 && my > 10.0 && my < 170.0 { hit_ui = true; }
                
                if !hit_ui {
                    offset_x += (mx - last_mouse.0) / zoom;
                    offset_y -= (my - last_mouse.1) / zoom;
                }
            }
        }
        last_mouse = (mx, my);

        // --- Sync & Read State ---
        // Use try_lock to ensure the 60FPS UI loop never blocks on a heavy simulation step
        if let Ok(mut data) = shared_data.try_lock() {
            if pending_pause_toggle {
                data.is_paused = !data.is_paused;
                pending_pause_toggle = false;
            }
            if pending_speed_change != 0 {
                if pending_speed_change > 0 {
                    data.ticks_per_loop = (data.ticks_per_loop as f32 * 1.5) as usize; // Scale up by 50%
                } else {
                    data.ticks_per_loop = (data.ticks_per_loop as f32 / 1.5).max(5.0) as usize; // Scale down
                }
                pending_speed_change = 0;
            }
            
            paused = data.is_paused;
            speed = data.ticks_per_loop;
            ticks = data.total_ticks;
            compute_time = data.last_compute_time_ms;
            pop_count = data.sim.agents.iter().filter(|a| a.health > 0.0).count();
            restart_msg = data.restart_message_active;
            
            // Dynamic map generation based on toggle state
            local_map_data.clear();
            if current_visual_mode == VisualMode::Resources {
                let max_res_ln = (data.config.max_tile_resource + 1.0).ln();
                for cell in &data.sim.env.map_cells {
                    let res = cell.res_value;
                    let mut r = 10; let mut g = 50; let mut b = 150; // Base Water
                    if res > 0.0 {
                        let ratio = ((res + 1.0).ln() / max_res_ln).clamp(0.0, 1.0);
                        r = ((1.0 - ratio) * 255.0) as u8;
                        g = (ratio * 255.0) as u8;
                        b = 0;
                    }
                    local_map_data.extend_from_slice(&[r, g, b, 255]);
                }
            } else {
                local_map_data.extend_from_slice(&data.sim.env.map_data);
            }

            local_agent_coords.clear();
            local_agent_coords.extend(data.sim.agents.iter().map(|a| AgentRenderData {
                x: a.x, y: a.y, food: a.food, health: a.health, age: a.age, gender: a.gender, is_pregnant: a.is_pregnant
            }));
            
            // Sync data for the inspector UI
            if show_inspector {
                inspector_agents.clear();
                for (i, a) in data.sim.agents.iter().enumerate() {
                    if a.health > 0.0 {
                        inspector_agents.push((i, *a));
                    }
                }
            }
        }

        // --- Rendering ---
        clear_background(BLACK);

        let cam = Camera2D {
            target: vec2(map_width as f32 / 2.0 - offset_x, map_height as f32 / 2.0 - offset_y),
            // -zoom for the Y axis is used here to match a 2D top-left Origin standard
            zoom: vec2(zoom / (screen_width() / 2.0), -zoom / (screen_height() / 2.0)), 
            ..Default::default()
        };
        set_camera(&cam);

        // 1. Efficient Zero-Copy Map Rendering
        if !local_map_data.is_empty() {
            image.bytes.copy_from_slice(&local_map_data);
            texture.update(&image);
        }
        draw_texture(&texture, 0.0, 0.0, WHITE);

        // 2. High-performance Agent Rendering
        let max_inv_ln = (loaded_config.boat_cost * 0.25 + 1.0).ln();
        for a in &local_agent_coords {
            if a.health <= 0.0 {
                draw_circle(a.x, a.y, 1.0, Color::new(0.2, 0.2, 0.2, 0.5)); // Draw corpses faintly
                continue;
            }
            
            // Render size based on maturity
            let radius = 1.0 + (a.age / loaded_config.puberty_age).min(1.0) * 1.0;
            
            let color = match current_visual_mode {
                VisualMode::Resources => {
                    let ratio = ((a.food.max(0.0) + 1.0).ln() / max_inv_ln).clamp(0.0, 1.0);
                    Color::new(1.0 - ratio, ratio, 0.0, 1.0)
                },
                VisualMode::Age => {
                    let ratio = (a.age / loaded_config.max_age).clamp(0.0, 1.0);
                    Color::new(ratio, 1.0 - ratio, 1.0, 1.0)
                },
                VisualMode::Gender => if a.gender > 0.5 { Color::new(0.2, 0.6, 1.0, 1.0) } else { Color::new(1.0, 0.4, 0.7, 1.0) },
                VisualMode::Pregnancy => if a.is_pregnant > 0.5 { Color::new(1.0, 0.8, 0.0, 1.0) } else if a.gender > 0.5 { Color::new(0.2, 0.4, 0.8, 0.5) } else { Color::new(0.8, 0.2, 0.5, 0.5) },
                VisualMode::Default => WHITE,
            };
            draw_circle(a.x, a.y, radius, color);
        }

        // 3. Render UI / Metrics
        set_default_camera(); // Reset camera transformations for static HUD overlays
        
        frame_times.push(get_frame_time());
        if frame_times.len() > 300 {
            frame_times.remove(0);
        }
        
        let mut avg_fps = 0.0;
        let mut low_1_fps = 0.0;
        if !frame_times.is_empty() {
            let sum_time: f32 = frame_times.iter().sum();
            avg_fps = frame_times.len() as f32 / sum_time;
            
            let mut sorted = frame_times.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let count_1_percent = (sorted.len() / 100).max(1);
            let low_1_sum: f32 = sorted.iter().rev().take(count_1_percent).sum(); // Take the longest frame times
            low_1_fps = count_1_percent as f32 / low_1_sum;
        }
        
        draw_rectangle(10.0, 10.0, 260.0, 240.0, Color::new(0.0, 0.04, 0.04, 0.9));
        draw_rectangle_lines(10.0, 10.0, 260.0, 240.0, 1.0, Color::new(0.0, 1.0, 0.8, 1.0));
        
        let fps = get_fps();
        let mut y = 30.0;
        let dy = 20.0;
        
        draw_text("REAL-TIME METRICS", 20.0, y, 16.0, Color::new(0.0, 1.0, 0.8, 1.0));
        y += dy;
        draw_text(&format!("Population: {}", pop_count), 20.0, y, 16.0, WHITE);
        y += dy;
        draw_text(&format!("Compute: {}ms/loop", compute_time), 20.0, y, 16.0, WHITE);
        y += dy;
        draw_text(&format!("Speed: {}x (Up/Down)", speed), 20.0, y, 16.0, WHITE);
        y += dy;
        let time_str = format_time(ticks, loaded_config.tick_to_mins);
        draw_text(&format!("Sim Time: {}", time_str), 20.0, y, 16.0, WHITE);
        y += dy;
        draw_text(&format!("FPS: {}", fps), 20.0, y, 16.0, WHITE);
        y += dy;
        draw_text(&format!("Avg FPS: {:.1}", avg_fps), 20.0, y, 16.0, WHITE);
        y += dy;
        draw_text(&format!("1% Low: {:.1}", low_1_fps), 20.0, y, 16.0, WHITE);
        y += dy;
        let mode_str = match current_visual_mode {
            VisualMode::Default => "Default",
            VisualMode::Resources => "Resources",
            VisualMode::Age => "Age",
            VisualMode::Gender => "Gender",
            VisualMode::Pregnancy => "Pregnancy",
        };
        draw_text(&format!("Visuals [R]: {}", mode_str), 20.0, y, 16.0, WHITE);
        y += dy;
        draw_text(&format!("Inspector [TAB]: {}", if show_inspector { "OPEN" } else { "CLOSED" }), 20.0, y, 16.0, WHITE);

        if paused {
            draw_text("PAUSED", screen_width() / 2.0 - 50.0, 30.0, 30.0, RED);
        }
        
        if restart_msg {
            let text = "ALL DIED, RESTARTING...";
            let text_dims = measure_text(text, None, 40, 1.0);
            draw_text(text, screen_width() / 2.0 - text_dims.width / 2.0, screen_height() / 2.0, 40.0, RED);
        }

        // --- Visuals Toggle Overlay ---
        if show_visuals_panel {
            draw_rectangle(280.0, 10.0, 160.0, 160.0, Color::new(0.0, 0.04, 0.04, 0.9));
            draw_rectangle_lines(280.0, 10.0, 160.0, 160.0, 1.0, Color::new(0.0, 1.0, 0.8, 1.0));
            draw_text("VISUALS", 290.0, 30.0, 16.0, Color::new(0.0, 1.0, 0.8, 1.0));
            
            let modes = [
                (VisualMode::Default, "1. Default"),
                (VisualMode::Resources, "2. Resources"),
                (VisualMode::Age, "3. Age"),
                (VisualMode::Gender, "4. Gender"),
                (VisualMode::Pregnancy, "5. Pregnancy"),
            ];
            
            let mut vy = 55.0;
            for (mode, label) in modes.iter() {
                let is_hover = mx > 290.0 && mx < 430.0 && my > vy - 12.0 && my < vy + 4.0;
                let color = if current_visual_mode == *mode { WHITE } else if is_hover { GRAY } else { DARKGRAY };
                draw_text(label, 290.0, vy, 16.0, color);
                
                if left_clicked && is_hover {
                    current_visual_mode = *mode;
                }
                vy += 22.0;
            }
        }

        // --- UI Inspector Overlay ---
        if show_inspector {
            draw_rectangle(40.0, 40.0, screen_width() - 80.0, screen_height() - 80.0, Color::new(0.05, 0.05, 0.05, 0.95));
            draw_rectangle_lines(40.0, 40.0, screen_width() - 80.0, screen_height() - 80.0, 2.0, Color::new(0.0, 1.0, 0.8, 1.0));
            
            if let Some(a) = selected_agent {
                // Detail View: Neural Network Heatmap
                let is_hover_back = mx > 60.0 && mx < 140.0 && my > 60.0 && my < 90.0;
                draw_rectangle(60.0, 60.0, 80.0, 30.0, if is_hover_back { Color::new(0.4, 0.4, 0.4, 1.0) } else { Color::new(0.2, 0.2, 0.2, 1.0) });
                draw_text("<- BACK", 65.0, 80.0, 20.0, WHITE);
                if left_clicked && is_hover_back { selected_agent = None; }
                
                draw_text(&format!("Stats: Age {} | HP {:.1} | Food {:.1} | H2O {:.1} | Sta {:.0}", format_time(a.age as u64, loaded_config.tick_to_mins), a.health, a.food, a.water, a.stamina), 160.0, 80.0, 20.0, WHITE);

                let start_x = 50.0;
                let start_y = 180.0;
                let cs = 10.0; // Fit the massive grids!

                draw_text("Inputs (32) -> H1", start_x, start_y - 65.0, 16.0, WHITE);
                draw_text("1:Bias 2:X 3:Y 4:Res 5:Pop 6:Spd 7:Shr 8:Rep 9:Atk 10:Prg", start_x, start_y - 50.0, 14.0, GRAY);
                draw_text("11:Trn 12:Rst 13:C1 14:C2 15:C3 16:C4 17:HP 18:Fd 19:H2O 20:Sta", start_x, start_y - 35.0, 14.0, GRAY);
                draw_text("21:Age 22:Gen 23:LRes 24:LElv 25:LPop 26:Tmp 27:Sea 28:Prg 29:Enc 30:Crw 31,32:-", start_x, start_y - 20.0, 14.0, GRAY);

                for h in 0..a.hidden_count as usize {
                    for i in 0..32 {
                        let w = a.w1[h * 32 + i];
                        let color = if w > 0.0 { Color::new(0.0, w.min(1.0), 0.0, 1.0) } else { Color::new((-w).min(1.0), 0.0, 0.0, 1.0) };
                        draw_rectangle(start_x + i as f32 * cs, start_y + h as f32 * cs, cs - 1.0, cs - 1.0, color);
                    }
                }

                let start_x2 = start_x + 34.0 * cs;
                draw_text("H1 -> H2", start_x2, start_y - 65.0, 16.0, WHITE);
                for h2 in 0..a.hidden_count as usize {
                    for h1 in 0..a.hidden_count as usize {
                        let w = a.w2[h1 * 32 + h2];
                        let color = if w > 0.0 { Color::new(0.0, w.min(1.0), 0.0, 1.0) } else { Color::new((-w).min(1.0), 0.0, 0.0, 1.0) };
                        draw_rectangle(start_x2 + h1 as f32 * cs, start_y + h2 as f32 * cs, cs - 1.0, cs - 1.0, color);
                    }
                }
                
                let start_x3 = start_x2 + 34.0 * cs;
                draw_text("H2 -> Outputs (10)", start_x3, start_y - 65.0, 16.0, WHITE);
                draw_text("1:Trn 2:Spd 3:Shr 4:Rep 5:Atk", start_x3, start_y - 50.0, 14.0, GRAY);
                draw_text("6:Rst 7:C1 8:C2 9:C3 10:C4", start_x3, start_y - 35.0, 14.0, GRAY);
                
                for o in 0..10 {
                    for h in 0..a.hidden_count as usize {
                        let w = a.w3[h * 10 + o];
                        let color = if w > 0.0 { Color::new(0.0, w.min(1.0), 0.0, 1.0) } else { Color::new((-w).min(1.0), 0.0, 0.0, 1.0) };
                        draw_rectangle(start_x3 + o as f32 * cs, start_y + h as f32 * cs, cs - 1.0, cs - 1.0, color);
                    }
                }
            } else {
                // Table View
                inspector_agents.sort_by(|a, b| {
                    let cmp = match sort_col {
                        SortCol::Index => a.0.cmp(&b.0),
                        SortCol::Age => a.1.age.partial_cmp(&b.1.age).unwrap_or(std::cmp::Ordering::Equal),
                        SortCol::Health => a.1.health.partial_cmp(&b.1.health).unwrap_or(std::cmp::Ordering::Equal),
                        SortCol::Food => a.1.food.partial_cmp(&b.1.food).unwrap_or(std::cmp::Ordering::Equal),
                        SortCol::Gender => a.1.gender.partial_cmp(&b.1.gender).unwrap_or(std::cmp::Ordering::Equal),
                        SortCol::Outputs => a.1.reproduce_desire.partial_cmp(&b.1.reproduce_desire).unwrap_or(std::cmp::Ordering::Equal),
                    };
                    if sort_desc { cmp.reverse() } else { cmp }
                });

                let headers = [("ID", 60.0, SortCol::Index), ("Age", 140.0, SortCol::Age), ("Health", 210.0, SortCol::Health), ("Food", 270.0, SortCol::Food), ("Gen", 330.0, SortCol::Gender), ("Outputs (Sp,Rp,At,Rs)", 380.0, SortCol::Outputs)];

                for (label, hx, col) in headers.iter() {
                    let is_hover = mx > *hx && mx < *hx + 60.0 && my > 50.0 && my < 80.0;
                    let color = if sort_col == *col { Color::new(0.0, 1.0, 0.8, 1.0) } else if is_hover { GRAY } else { WHITE };
                    draw_text(label, *hx, 70.0, 20.0, color);
                    
                    if left_clicked && is_hover {
                        if sort_col == *col { sort_desc = !sort_desc; } else { sort_col = *col; sort_desc = true; }
                    }
                }

                let row_h = 20.0;
                let visible = 22;
                if mouse_wheel_y < 0.0 { inspector_scroll = inspector_scroll.saturating_add(1); }
                if mouse_wheel_y > 0.0 { inspector_scroll = inspector_scroll.saturating_sub(1); }
                inspector_scroll = inspector_scroll.min(inspector_agents.len().saturating_sub(visible));

                for i in 0..visible {
                    let idx = inspector_scroll + i;
                    if idx >= inspector_agents.len() { break; }
                    let (a_id, a) = &inspector_agents[idx];
                    let y = 100.0 + i as f32 * row_h;
                    
                    if mx > 50.0 && mx < screen_width() - 50.0 && my > y - 15.0 && my < y + 5.0 {
                        draw_rectangle(50.0, y - 15.0, screen_width() - 100.0, row_h, Color::new(0.2, 0.2, 0.2, 0.8));
                        if left_clicked { selected_agent = Some(*a); }
                    }

                    draw_text(&format!("{}", a_id), 60.0, y, 16.0, WHITE);
                    draw_text(&format!("{}", format_time(a.age as u64, loaded_config.tick_to_mins)), 140.0, y, 16.0, WHITE);
                    draw_text(&format!("{:.1}", a.health), 210.0, y, 16.0, WHITE);
                    draw_text(&format!("{:.1}", a.food), 270.0, y, 16.0, WHITE);
                    draw_text(if a.gender > 0.5 { "M" } else { "F" }, 330.0, y, 16.0, WHITE);
                    
                    let spd = (a.speed / loaded_config.base_speed).clamp(0.0, 1.0);
                    let out_str = format!("S:{:.1} R:{:.1} A:{:.1} Z:{:.1} C1..4: {:.1} {:.1} {:.1} {:.1}", spd, a.reproduce_desire, a.attack_intent, a.rest_intent, a.comm1, a.comm2, a.comm3, a.comm4);
                    draw_text(&out_str, 370.0, y, 16.0, WHITE);
                }
                
                draw_text(&format!("Showing {} - {} of {}", inspector_scroll, (inspector_scroll + visible).min(inspector_agents.len()), inspector_agents.len()), 60.0, 550.0, 16.0, GRAY);
                draw_text("Scroll to view more. Click row to inspect Neural Network.", 300.0, 550.0, 16.0, GRAY);
            }
        }

        next_frame().await;
    }
}