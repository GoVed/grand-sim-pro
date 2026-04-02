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
mod shared;
mod sim_thread;
mod ui;

use macroquad::prelude::*;
use simulation::SimulationManager;
use gpu_engine::GpuEngine;
use std::sync::{Arc, Mutex};
use ::rand::Rng;
use shared::{SharedData, VisualMode, SortCol, AgentRenderData};

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
        generation_survival_times: Vec::new(),
    }));

    let sim_thread_data = shared_data.clone();
    
    let initial_agents = sim_thread_data.lock().unwrap().sim.agents.clone();
    let height_map = sim_thread_data.lock().unwrap().sim.env.height_map.clone();
    let init_cells = sim_thread_data.lock().unwrap().sim.env.map_cells.clone();
    let gpu = Arc::new(GpuEngine::new(&initial_agents, &height_map, &init_cells, &loaded_config));

    sim_thread::spawn(sim_thread_data.clone(), gpu.clone());

    let mut image = Image::gen_image_color(map_width as u16, map_height as u16, BLANK);
    let texture = Texture2D::from_image(&image);
    
    let mut zoom = 1.0f32;
    let mut offset_x = 0.0f32;
    let mut offset_y = 0.0f32;
    let mut last_mouse = mouse_position();

    let mut local_agent_coords: Vec<AgentRenderData> = Vec::with_capacity(agent_count as usize);
    
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

    let mut pending_save_agents = false;
    let mut frame_times: Vec<f32> = Vec::with_capacity(300);
    
    let mut show_inspector = false;
    let mut show_generation_graph = false;
    let mut sort_col = SortCol::Food;
    let mut sort_desc = true;
    let mut inspector_scroll: usize = 0;
    let mut selected_agent: Option<crate::agent::Person> = None;
    let mut inspector_agents: Vec<(usize, crate::agent::Person)> = Vec::with_capacity(agent_count as usize);
    
    let mut followed_agent_id: Option<u32> = None;
    let mut followed_agent: Option<crate::agent::Person> = None;
    let mut generation_times: Vec<u64> = Vec::new();

    loop {
        // Register UI inputs instantly
        if is_key_pressed(KeyCode::Space) { pending_pause_toggle = !pending_pause_toggle; }
        if is_key_pressed(KeyCode::S) { pending_save_agents = true; }
        if is_key_pressed(KeyCode::R) { show_visuals_panel = !show_visuals_panel; }
        if is_key_pressed(KeyCode::T) { current_visual_mode = VisualMode::Temperature; }
        if is_key_pressed(KeyCode::G) { show_generation_graph = !show_generation_graph; }
        if is_key_pressed(KeyCode::N) { current_visual_mode = VisualMode::DayNight; }
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
                if show_visuals_panel && mx > 280.0 && mx < 440.0 && my > 10.0 && my < 320.0 { hit_ui = true; }
                
                if !hit_ui {
                    offset_x += (mx - last_mouse.0) / zoom;
                    offset_y -= (my - last_mouse.1) / zoom;
                }
            }
        }
        last_mouse = (mx, my);

        let mode_u32 = match current_visual_mode {
            VisualMode::Default => 0,
            VisualMode::Resources => 1,
            VisualMode::Age => 2,
            VisualMode::Gender => 3,
            VisualMode::Pregnancy => 4,
            VisualMode::MarketWealth => 5,
            VisualMode::MarketFood => 6,
            VisualMode::AskPrice => 7,
            VisualMode::BidPrice => 8,
            VisualMode::Shelter => 9,
            VisualMode::Temperature => 10,
            VisualMode::DayNight => 11,
        };

        // --- Sync & Read State ---
        // Use try_lock to ensure the 60FPS UI loop never blocks on a heavy simulation step
        if let Ok(mut data) = shared_data.try_lock() {
            data.config.visual_mode = mode_u32;
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

            if pending_save_agents {
                let _ = std::fs::create_dir_all("saved_agents_weights");
                let mut living: Vec<_> = data.sim.agents.iter().filter(|a| a.health > 0.0).collect();
                living.sort_by(|a, b| {
                    let score_a = a.wealth + a.food;
                    let score_b = b.wealth + b.food;
                    score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
                });
                let save_count = living.len().min(data.config.founder_count as usize);
                for i in 0..save_count {
                    let weights = living[i].extract_weights();
                    if let Ok(json) = serde_json::to_string_pretty(&weights) {
                        let _ = std::fs::write(format!("saved_agents_weights/agent_{}.json", i), json);
                    }
                }
                pending_save_agents = false;
            }
            
            paused = data.is_paused;
            speed = data.ticks_per_loop;
            ticks = data.total_ticks;
            compute_time = data.last_compute_time_ms;
            pop_count = data.sim.agents.iter().filter(|a| a.health > 0.0).count();
            restart_msg = data.restart_message_active;
            
            if show_generation_graph {
                generation_times = data.generation_survival_times.clone();
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

        // --- Day/Night Cycle Visuals ---
        let ticks_per_day = 24.0 * 60.0 / loaded_config.tick_to_mins;
        let day_cycle_progress = (ticks as f32 % ticks_per_day) / ticks_per_day;

        let cam = Camera2D {
            target: vec2(map_width as f32 / 2.0 - offset_x, map_height as f32 / 2.0 - offset_y),
            // -zoom for the Y axis is used here to match a 2D top-left Origin standard
            zoom: vec2(zoom / (screen_width() / 2.0), -zoom / (screen_height() / 2.0)), 
            ..Default::default()
        };
        set_camera(&cam);

        // 1. Efficient GPU-Accelerated Zero-Copy Map Rendering
        let render_data = gpu.fetch_render(map_width, map_height);
        image.bytes.copy_from_slice(&render_data);
        texture.update(&image);
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
                    let val = if current_visual_mode == VisualMode::MarketFood { a.food / 1000.0 } else { a.wealth };
                    let ratio = ((val.max(0.0) + 1.0).ln() / max_inv_ln).clamp(0.0, 1.0);
                    Color::new(1.0 - ratio, ratio, 0.0, 1.0)
                },
                VisualMode::Age => {
                    let ratio = (a.age / loaded_config.max_age).clamp(0.0, 1.0);
                    Color::new(ratio, 1.0 - ratio, 1.0, 1.0)
                },
                VisualMode::Gender => if a.gender > 0.5 { Color::new(0.2, 0.6, 1.0, 1.0) } else { Color::new(1.0, 0.4, 0.7, 1.0) },
                VisualMode::Pregnancy => if a.is_pregnant > 0.5 { Color::new(1.0, 0.8, 0.0, 1.0) } else if a.gender > 0.5 { Color::new(0.2, 0.4, 0.8, 0.5) } else { Color::new(0.8, 0.2, 0.5, 0.5) },
                VisualMode::Default | VisualMode::Shelter | VisualMode::Temperature | VisualMode::DayNight => WHITE,
            };
            
            draw_circle(a.x, a.y, radius, color);
        }

        if let Some(ref a) = followed_agent {
            draw_circle_lines(a.x, a.y, 8.0, 2.0, YELLOW); // Highlight the tracked agent
        }

        // 3. Render UI / Metrics
        set_default_camera(); // Reset camera transformations for static HUD overlays

        // --- Draw Clock ---
        let clock_x = screen_width() - 220.0;
        let hours = (day_cycle_progress * 24.0).floor() as u32;
        let minutes = ((day_cycle_progress * 24.0).fract() * 60.0).floor() as u32;
        let time_str = format!("Time: {:02}:{:02}", hours, minutes);
        draw_rectangle(clock_x, 10.0, 210.0, 40.0, Color::new(0.0, 0.0, 0.0, 0.7));
        draw_text(&time_str, clock_x + 10.0, 38.0, 28.0, WHITE);

        // --- Draw Legend ---
        let legend_x = screen_width() - 220.0;
        let legend_y = screen_height() - 140.0;
        draw_rectangle(legend_x, legend_y, 210.0, 130.0, Color::new(0.0, 0.0, 0.0, 0.7));
        draw_text("Legend", legend_x + 10.0, legend_y + 25.0, 24.0, WHITE);
        
        match current_visual_mode {
            VisualMode::Resources | VisualMode::MarketWealth | VisualMode::MarketFood | VisualMode::AskPrice | VisualMode::BidPrice => {
                draw_text("High Value", legend_x + 10.0, legend_y + 60.0, 20.0, GREEN);
                draw_text("Low Value", legend_x + 10.0, legend_y + 90.0, 20.0, RED);
            },
            VisualMode::Age => {
                draw_text("Young", legend_x + 10.0, legend_y + 60.0, 20.0, Color::new(0.0, 1.0, 1.0, 1.0));
                draw_text("Old", legend_x + 10.0, legend_y + 90.0, 20.0, Color::new(1.0, 0.0, 1.0, 1.0));
            },
            VisualMode::Gender => {
                draw_text("Male", legend_x + 10.0, legend_y + 60.0, 20.0, Color::new(0.2, 0.6, 1.0, 1.0));
                draw_text("Female", legend_x + 10.0, legend_y + 90.0, 20.0, Color::new(1.0, 0.4, 0.7, 1.0));
            },
            VisualMode::Pregnancy => {
                draw_text("Pregnant", legend_x + 10.0, legend_y + 60.0, 20.0, Color::new(1.0, 0.8, 0.0, 1.0));
                draw_text("Not Pregnant", legend_x + 10.0, legend_y + 90.0, 20.0, GRAY);
            },
            VisualMode::Shelter => {
                draw_text("High Shelter", legend_x + 10.0, legend_y + 60.0, 20.0, Color::new(0.54, 0.27, 0.07, 1.0));
                draw_text("No Shelter", legend_x + 10.0, legend_y + 90.0, 20.0, BLACK);
            },
            VisualMode::Temperature => {
                draw_text("Hot", legend_x + 10.0, legend_y + 60.0, 20.0, RED);
                draw_text("Temperate", legend_x + 10.0, legend_y + 90.0, 20.0, Color::new(0.5, 0.2, 0.5, 1.0));
                draw_text("Freezing", legend_x + 10.0, legend_y + 120.0, 20.0, BLUE);
            },
            VisualMode::DayNight => {
                draw_text("Daylight", legend_x + 10.0, legend_y + 60.0, 20.0, WHITE);
                draw_text("Night Shadow", legend_x + 10.0, legend_y + 90.0, 20.0, DARKGRAY);
            },
            VisualMode::Default => {
                draw_text("Biome / Terrain", legend_x + 10.0, legend_y + 60.0, 20.0, WHITE);
                draw_text("Agents (White)", legend_x + 10.0, legend_y + 90.0, 20.0, WHITE);
            }
        }
        
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
        
        ui::draw_metrics(pop_count, compute_time, speed, ticks, loaded_config.tick_to_mins, get_fps(), avg_fps, low_1_fps, current_visual_mode, show_inspector, show_generation_graph, paused, restart_msg);
        if show_visuals_panel { ui::draw_visuals_panel(mx, my, left_clicked, &mut current_visual_mode); }
        if show_generation_graph {
            ui::draw_generation_graph(&generation_times, loaded_config.tick_to_mins);
        }
        if show_inspector {
            ui::draw_inspector(mx, my, left_clicked, mouse_wheel_y, &mut inspector_agents, &mut sort_col, &mut sort_desc, &mut inspector_scroll, &mut selected_agent, &mut followed_agent_id, &mut show_inspector, loaded_config.tick_to_mins, loaded_config.base_speed);
        } else if let Some(a) = &followed_agent {
            ui::draw_tracker(mx, my, left_clicked, a, &mut followed_agent_id, &mut show_inspector, loaded_config.tick_to_mins);
        }

        next_frame().await;
    }
}