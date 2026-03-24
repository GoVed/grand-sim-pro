/*
 * Grand Sim Pro: A high-performance GPGPU evolutionary agent simulation.
 * Part of an independent research project into emergent biological complexity.
 *
 * Copyright (C) 2026 Ved Hirenkumar Suthar
 * Licensed under the GNU General Public License v3.0 or later.
 * * This software is provided "as is", without warranty of any kind.
 * See the LICENSE file in the project root for full license details.
 */

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
use ::rand::Rng;

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
enum VisualMode { Default, Resources, Age, Gender, Pregnancy, MarketWealth, MarketFood, AskPrice, BidPrice }

#[derive(PartialEq, Clone, Copy)]
enum SortCol { Index, Age, Health, Food, Wealth, Gender, Speed, Heading, State, Outputs }

struct AgentRenderData {
    x: f32,
    y: f32,
    health: f32,
    food: f32,
    age: f32,
    wealth: f32,
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

    let random_seed = ::rand::thread_rng().r#gen::<u32>();

    let shared_data = Arc::new(Mutex::new(SharedData {
        sim: SimulationManager::new(map_width, map_height, random_seed, agent_count, &loaded_config),
        config: loaded_config,
        is_paused: false,
        restart_message_active: false,
        ticks_per_loop: 200,
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
                        if is_mature && a.reproduce_desire > 0.5 && a.wealth >= reproduction_cost / 2.0 && a.health > data.config.max_health * 0.5 {
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
                        data.sim.env.map_cells[i].avg_ask = 1.0;
                        data.sim.env.map_cells[i].avg_bid = 1.0;
                        data.sim.env.map_cells[i].market_food = 50.0;
                        data.sim.env.map_cells[i].market_wealth = max_res;
                        data.sim.env.map_cells[i].pad1 = [0.0; 1];
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
    
    let mut followed_agent_id: Option<u32> = None;
    let mut followed_agent: Option<crate::agent::Person> = None;

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
            
            if is_mouse_button_down(MouseButton::Left) && followed_agent_id.is_none() {
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
            match current_visual_mode {
                VisualMode::Resources | VisualMode::MarketWealth | VisualMode::MarketFood | VisualMode::AskPrice | VisualMode::BidPrice => {
                    let max_res_ln = (data.config.max_tile_resource + 1.0).ln();
                    let max_price_ln = 11.0_f32.ln(); // Market prices cap around $10
                    for cell in &data.sim.env.map_cells {
                        let (val, max_ln) = match current_visual_mode {
                            VisualMode::Resources => (cell.res_value, max_res_ln),
                            VisualMode::MarketWealth => (cell.market_wealth, max_res_ln),
                            VisualMode::MarketFood => (cell.market_food, max_res_ln),
                            VisualMode::AskPrice => (cell.avg_ask, max_price_ln),
                            VisualMode::BidPrice => (cell.avg_bid, max_price_ln),
                            _ => (0.0, 1.0),
                        };
                        let mut r = 10; let mut g = 50; let mut b = 150; // Base Water
                        if val > 0.0 {
                            let ratio = ((val + 1.0).ln() / max_ln).clamp(0.0, 1.0);
                            r = ((1.0 - ratio) * 255.0) as u8;
                            g = (ratio * 255.0) as u8;
                            b = 0;
                        }
                        local_map_data.extend_from_slice(&[r, g, b, 255]);
                    }
                },
                _ => local_map_data.extend_from_slice(&data.sim.env.map_data),
            }

            local_agent_coords.clear();
            local_agent_coords.extend(data.sim.agents.iter().map(|a| AgentRenderData {
                x: a.x, y: a.y, health: a.health, food: a.food, age: a.age, wealth: a.wealth, gender: a.gender, is_pregnant: a.is_pregnant
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
            
            if let Some(fid) = followed_agent_id {
                followed_agent = data.sim.agents.iter().find(|a| a.id == fid).cloned();
            } else {
                followed_agent = None;
            }
        }

        if let Some(ref a) = followed_agent {
            offset_x = map_width as f32 / 2.0 - a.x;
            offset_y = map_height as f32 / 2.0 - a.y;
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
                VisualMode::Resources | VisualMode::MarketWealth | VisualMode::MarketFood | VisualMode::AskPrice | VisualMode::BidPrice => {
                    let val = if current_visual_mode == VisualMode::MarketFood { a.food } else { a.wealth };
                    let ratio = ((val.max(0.0) + 1.0).ln() / max_inv_ln).clamp(0.0, 1.0);
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

        if let Some(ref a) = followed_agent {
            draw_circle_lines(a.x, a.y, 8.0, 2.0, YELLOW); // Highlight the tracked agent
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
            VisualMode::MarketWealth => "Market Wealth",
            VisualMode::MarketFood => "Market Food",
            VisualMode::AskPrice => "Ask Price",
            VisualMode::BidPrice => "Bid Price",
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
            draw_rectangle(280.0, 10.0, 160.0, 240.0, Color::new(0.0, 0.04, 0.04, 0.9));
            draw_rectangle_lines(280.0, 10.0, 160.0, 240.0, 1.0, Color::new(0.0, 1.0, 0.8, 1.0));
            draw_text("VISUALS", 290.0, 30.0, 16.0, Color::new(0.0, 1.0, 0.8, 1.0));
            
            let modes = [
                (VisualMode::Default, "1. Default"),
                (VisualMode::Resources, "2. Resources"),
                (VisualMode::Age, "3. Age"),
                (VisualMode::Gender, "4. Gender"),
                (VisualMode::Pregnancy, "5. Pregnancy"),
                (VisualMode::MarketWealth, "6. Market Wealth"),
                (VisualMode::MarketFood, "7. Market Food"),
                (VisualMode::AskPrice, "8. Ask Price"),
                (VisualMode::BidPrice, "9. Bid Price"),
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
                
                draw_text(&format!("Stats: Age {} | HP {:.1} | Food {:.1} | H2O {:.1} | Wealth ${:.1}", format_time(a.age as u64, loaded_config.tick_to_mins), a.health, a.food, a.water, a.wealth), 160.0, 80.0, 20.0, WHITE);

                let start_x = 50.0;
                let start_y = 180.0;
                let cs = 10.0; // Fit the massive grids!

                draw_text("Inputs (40) -> H1", start_x, start_y - 65.0, 16.0, WHITE);
                draw_text("1:Bias 2:X 3:Y 4:Res 5:Pop 6:Spd 7:Shr 8:Rep 9:Atk 10:Prg", start_x, start_y - 50.0, 14.0, GRAY);
                draw_text("11:Trn 12:Rst 13:C1 14:C2 15:C3 16:C4 17:HP 18:Fd 19:H2O 20:Sta", start_x, start_y - 35.0, 14.0, GRAY);
                draw_text("21:Age 22:Gen 23:LRes 24:LElv 25:LPop 26:Tmp 27:Sea 28:Prg 29:Enc 30:Crw 31..34:Mem 35:Wlh 36:Ask 37:Bid", start_x, start_y - 20.0, 14.0, GRAY);

                for h in 0..a.hidden_count as usize {
                    for i in 0..40 {
                        let w = a.w1[h * 40 + i];
                        let color = if w > 0.0 { Color::new(0.0, w.min(1.0), 0.0, 1.0) } else { Color::new((-w).min(1.0), 0.0, 0.0, 1.0) };
                        draw_rectangle(start_x + i as f32 * cs, start_y + h as f32 * cs, cs - 1.0, cs - 1.0, color);
                    }
                }

                let start_x2 = start_x + 42.0 * cs;
                draw_text("H1 -> H2", start_x2, start_y - 65.0, 16.0, WHITE);
                for h2 in 0..a.hidden_count as usize {
                    for h1 in 0..a.hidden_count as usize {
                        let w = a.w2[h1 * 32 + h2];
                        let color = if w > 0.0 { Color::new(0.0, w.min(1.0), 0.0, 1.0) } else { Color::new((-w).min(1.0), 0.0, 0.0, 1.0) };
                        draw_rectangle(start_x2 + h1 as f32 * cs, start_y + h2 as f32 * cs, cs - 1.0, cs - 1.0, color);
                    }
                }
                
                let start_x3 = start_x2 + 34.0 * cs;
                draw_text("H2 -> Outputs (20)", start_x3, start_y - 65.0, 16.0, WHITE);
                draw_text("1:Trn 2:Spd 3:Shr 4:Rep 5:Atk 6:Rst 7:C1 8:C2", start_x3, start_y - 50.0, 14.0, GRAY);
                draw_text("9:C3 10:C4 11:Lrn 12..15:M1..M4 16:Buy 17:Sel 18:Ask 19:Bid", start_x3, start_y - 35.0, 14.0, GRAY);
                
                for o in 0..20 {
                    for h in 0..a.hidden_count as usize {
                        let w = a.w3[h * 20 + o];
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
                        SortCol::Wealth => a.1.wealth.partial_cmp(&b.1.wealth).unwrap_or(std::cmp::Ordering::Equal),
                        SortCol::Gender => a.1.gender.partial_cmp(&b.1.gender).unwrap_or(std::cmp::Ordering::Equal),
                        SortCol::Speed => a.1.speed.partial_cmp(&b.1.speed).unwrap_or(std::cmp::Ordering::Equal),
                        SortCol::Heading => a.1.heading.partial_cmp(&b.1.heading).unwrap_or(std::cmp::Ordering::Equal),
                        SortCol::State => a.1.is_pregnant.partial_cmp(&b.1.is_pregnant).unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| a.1.rest_intent.partial_cmp(&b.1.rest_intent).unwrap_or(std::cmp::Ordering::Equal)),
                        SortCol::Outputs => a.1.reproduce_desire.partial_cmp(&b.1.reproduce_desire).unwrap_or(std::cmp::Ordering::Equal),
                    };
                    if sort_desc { cmp.reverse() } else { cmp }
                });

                let headers = [
                    ("ID", 60.0, SortCol::Index), 
                    ("Age", 120.0, SortCol::Age), 
                    ("HP", 180.0, SortCol::Health), 
                    ("Fd", 230.0, SortCol::Food), 
                    ("Wlth", 280.0, SortCol::Wealth), 
                    ("Gen", 340.0, SortCol::Gender), 
                    ("Spd", 390.0, SortCol::Speed), 
                    ("Dir", 440.0, SortCol::Heading), 
                    ("State", 480.0, SortCol::State), 
                    ("Markets (Buy, Sel, Ask, Bid)", 580.0, SortCol::Outputs)
                ];

                for (label, hx, col) in headers.iter() {
                    let is_hover = mx > *hx && mx < *hx + 40.0 && my > 50.0 && my < 80.0;
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
                    
                    let loc_x = 800.0;
                    let is_hover_locate = mx > loc_x && mx < loc_x + 60.0 && my > y - 12.0 && my < y + 4.0;
                    
                    if mx > 50.0 && mx < screen_width() - 50.0 && my > y - 15.0 && my < y + 5.0 {
                        draw_rectangle(50.0, y - 15.0, screen_width() - 100.0, row_h, Color::new(0.2, 0.2, 0.2, 0.8));
                        if left_clicked {
                            if is_hover_locate {
                                followed_agent_id = Some(a.id);
                                show_inspector = false;
                                selected_agent = None;
                            } else {
                                selected_agent = Some(*a);
                            }
                        }
                    }

                    draw_text(&format!("{}", a_id), 60.0, y, 16.0, WHITE);
                    draw_text(&format!("{}", format_time(a.age as u64, loaded_config.tick_to_mins)), 120.0, y, 16.0, WHITE);
                    draw_text(&format!("{:.0}", a.health), 180.0, y, 16.0, WHITE);
                    draw_text(&format!("{:.0}", a.food), 230.0, y, 16.0, WHITE);
                    draw_text(&format!("${:.0}", a.wealth), 280.0, y, 16.0, WHITE);
                    draw_text(if a.gender > 0.5 { "M" } else { "F" }, 340.0, y, 16.0, WHITE);
                    
                    // Speed Indicator (Text + Bar Graphic)
                    draw_text(&format!("{:.1}", a.speed), 390.0, y, 16.0, WHITE);
                    let spd_ratio = (a.speed / loaded_config.base_speed).clamp(0.0, 1.0);
                    draw_rectangle(390.0, y + 4.0, spd_ratio * 25.0, 2.0, Color::new(0.0, 1.0, 0.5, 1.0));
                    
                    // Heading Indicator (Arrow Graphic)
                    let dir_cx = 450.0;
                    let dir_cy = y - 5.0;
                    let dx = a.heading.cos() * 8.0;
                    let dy = a.heading.sin() * 8.0;
                    draw_line(dir_cx - dx, dir_cy - dy, dir_cx + dx, dir_cy + dy, 1.5, LIGHTGRAY);
                    draw_circle(dir_cx + dx, dir_cy + dy, 2.0, WHITE);
                    
                    // State Indicator Tags
                    let mut st_x = 480.0;
                    if a.is_pregnant > 0.5 { draw_text("[PRG]", st_x, y, 14.0, YELLOW); st_x += 35.0; }
                    if a.rest_intent > 0.5 { draw_text("[Zzz]", st_x, y, 14.0, SKYBLUE); st_x += 35.0; }
                    if a.attack_intent > 0.5 { draw_text("[ATK]", st_x, y, 14.0, RED); }
                    
                    // Market Intents
                    let out_str = format!("B:{:.1} S:{:.1} A:{:.1} B:{:.1}", a.buy_intent, a.sell_intent, a.ask_price, a.bid_price);
                    draw_text(&out_str, 580.0, y, 16.0, WHITE);
                    
                    draw_text("[Locate]", loc_x, y, 16.0, if is_hover_locate { YELLOW } else { LIGHTGRAY });
                }
                
                draw_text(&format!("Showing {} - {} of {}", inspector_scroll, (inspector_scroll + visible).min(inspector_agents.len()), inspector_agents.len()), 60.0, 550.0, 16.0, GRAY);
                draw_text("Scroll to view more. Click row to inspect Neural Network.", 300.0, 550.0, 16.0, GRAY);
            }
        } else if let Some(a) = &followed_agent {
            let panel_w = 260.0;
            let panel_h = 360.0;
            let panel_x = screen_width() - panel_w - 20.0;
            let panel_y = 20.0;

            draw_rectangle(panel_x, panel_y, panel_w, panel_h, Color::new(0.0, 0.04, 0.04, 0.9));
            draw_rectangle_lines(panel_x, panel_y, panel_w, panel_h, 1.0, Color::new(0.0, 1.0, 0.8, 1.0));

            // Close button
            let close_x = panel_x + panel_w - 30.0;
            let close_y = panel_y + 10.0;
            let is_close_hover = mx > close_x && mx < close_x + 20.0 && my > close_y && my < close_y + 20.0;
            draw_text("X", close_x, close_y + 15.0, 20.0, if is_close_hover { RED } else { GRAY });
            if left_clicked && is_close_hover {
                followed_agent_id = None;
                show_inspector = true; // Reopen the inspector when closing tracker
            }

            let mut py = panel_y + 30.0;
            let dy = 22.0;
            draw_text("AGENT TRACKER", panel_x + 20.0, py, 18.0, Color::new(0.0, 1.0, 0.8, 1.0));
            py += dy;
            draw_text(&format!("ID: {}", a.id), panel_x + 20.0, py, 16.0, WHITE);
            py += dy;
            draw_text(&format!("Age: {}", format_time(a.age as u64, loaded_config.tick_to_mins)), panel_x + 20.0, py, 16.0, WHITE);
            py += dy;
            draw_text(&format!("Health: {:.1}", a.health), panel_x + 20.0, py, 16.0, WHITE);
            py += dy;
            draw_text(&format!("Food: {:.1} | H2O: {:.1}", a.food, a.water), panel_x + 20.0, py, 16.0, WHITE);
            py += dy;
            draw_text(&format!("Wealth: ${:.1}", a.wealth), panel_x + 20.0, py, 16.0, WHITE);
            py += dy;
            draw_text(&format!("Gender: {}", if a.gender > 0.5 { "Male" } else { "Female" }), panel_x + 20.0, py, 16.0, WHITE);
            py += dy;
            draw_text(&format!("Speed: {:.2}", a.speed), panel_x + 20.0, py, 16.0, WHITE);
            py += dy;
            
            let mut state_str = String::new();
            if a.health <= 0.0 { state_str.push_str("[DEAD] "); }
            if a.is_pregnant > 0.5 { state_str.push_str("[PRG] "); }
            if a.rest_intent > 0.5 { state_str.push_str("[Zzz] "); }
            if a.attack_intent > 0.5 { state_str.push_str("[ATK] "); }
            if state_str.is_empty() { state_str.push_str("[IDLE]"); }
            
            draw_text(&format!("State: {}", state_str), panel_x + 20.0, py, 16.0, WHITE);
            py += dy + 10.0;
            draw_text("INTENTS:", panel_x + 20.0, py, 16.0, GRAY);
            py += dy;
            draw_text(&format!("Buy: {:.2} | Sell: {:.2}", a.buy_intent, a.sell_intent), panel_x + 20.0, py, 16.0, WHITE);
            py += dy;
            draw_text(&format!("Ask: {:.2} | Bid: {:.2}", a.ask_price, a.bid_price), panel_x + 20.0, py, 16.0, WHITE);
            py += dy;
            draw_text(&format!("Reproduce: {:.2}", a.reproduce_desire), panel_x + 20.0, py, 16.0, WHITE);
        }

        next_frame().await;
    }
}