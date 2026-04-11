/*
 * Grand Sim Pro: A high-performance GPGPU evolutionary agent simulation.
 * Part of an independent research project into emergent biological complexity.
 *
 * Copyright (C) 2026 Ved Hirenkumar Suthar
 * Licensed under the GNU General Public License v3.0 or later.
 * * This software is provided "as is", without warranty of any kind.
 * See the LICENSE file in the project root for full license details.
 */

use world_sim::agent;
use world_sim::simulation;
use world_sim::gpu_engine;
use world_sim::config;
use world_sim::shared;
use world_sim::sim_thread;
use world_sim::ui;

use macroquad::prelude::*;
use simulation::SimulationManager;
use gpu_engine::GpuEngine;
use std::sync::{Arc, Mutex};
use ::rand::Rng;
use shared::{SharedData, VisualMode, SortCol, AgentRenderData};

// Standalone config loader so we can establish window dimensions before the async engine boots
fn load_config() -> config::SimConfig {
    #[cfg(not(target_os = "android"))]
    {
        if let Ok(data) = std::fs::read_to_string("sim_config.json") {
            if let Ok(cfg) = serde_json::from_str(&data) {
                return cfg;
            }
        }
        let default_cfg = config::SimConfig::default();
        std::fs::write("sim_config.json", serde_json::to_string_pretty(&default_cfg).unwrap()).ok();
        return default_cfg;
    }
    
    #[cfg(target_os = "android")]
    {
        // Android does not support direct working directory access. 
        // Fallback to defaults until native storage paths are configured.
        return config::SimConfig::default();
    }
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
    #[cfg(target_os = "android")]
    {
        android_logger::init_once(
            android_logger::Config::default().with_max_level(log::LevelFilter::Trace),
        );
        std::panic::set_hook(Box::new(|info| {
            log::error!("RUST PANIC: {:?}", info);
        }));
    }

    let loaded_config = load_config();
    let map_width = loaded_config.map_width;
    let map_height = loaded_config.map_height;
    let agent_count = loaded_config.agent_count;

    let random_seed = ::rand::thread_rng().r#gen::<u32>();

    let (tx, rx) = std::sync::mpsc::channel();
    let loaded_config_clone = loaded_config;
    std::thread::spawn(move || {
        let sim = SimulationManager::new(map_width, map_height, random_seed, agent_count, &loaded_config_clone);
        let _ = tx.send(sim);
    });

    let mut dots = 0;
    let mut last_dot_time = get_time();
    let sim_manager = loop {
        clear_background(DARKGRAY);
        let dot_str = ".".repeat(dots);
        draw_text(&format!("Generating Procedural World & Spawning Agents{}", dot_str), 40.0, screen_height() / 2.0, 30.0, WHITE);
        
        if get_time() - last_dot_time > 0.5 {
            dots = (dots + 1) % 4;
            last_dot_time = get_time();
        }
        
        match rx.try_recv() {
            Ok(sim) => break sim,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => panic!("World generation thread panicked"),
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
        }
        
        next_frame().await;
    };

    let shared_data = Arc::new(Mutex::new(SharedData {
        sim: sim_manager,
        config: loaded_config,
        is_paused: false,
        restart_message_active: false,
        ticks_per_loop: 5,
        total_ticks: 0,
        last_compute_time_ms: 0,
        generation_survival_times: Vec::new(),
    }));

    let sim_thread_data = shared_data.clone();
    
    let initial_agents = sim_thread_data.lock().unwrap().sim.agents.clone();
    let height_map = sim_thread_data.lock().unwrap().sim.env.height_map.clone();
    let init_cells = sim_thread_data.lock().unwrap().sim.env.map_cells.clone();

    // Yield a few frames to the Android OS to fully map the surface and flush the UI buffer
    for dots in 0..4 {
        clear_background(DARKGRAY);
        let dot_str = ".".repeat(dots);
        draw_text(&format!("Initializing GPU Compute Engine{}", dot_str), 40.0, screen_height() / 2.0, 30.0, WHITE);
        next_frame().await;
    }

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
    
    let mut show_config_panel = false;
    let mut config_scroll = 0.0;
    let mut pending_save_config = false;
    let mut local_config = loaded_config;
    let mut config_changed_by_ui = false;
    let mut active_config_button: Option<String> = None;
    let mut config_button_hold_time: f32 = 0.0;
    
    let mut followed_agent_id: Option<u32> = None;
    let mut followed_agent: Option<crate::agent::Person> = None;
    let mut generation_times: Vec<u64> = Vec::new();
    let mut last_touch_distance: Option<f32> = None;

    loop {
        // Register UI inputs instantly
        if is_key_pressed(KeyCode::Space) { pending_pause_toggle = !pending_pause_toggle; }
        if is_key_pressed(KeyCode::S) { pending_save_agents = true; }
        if is_key_pressed(KeyCode::C) { show_config_panel = !show_config_panel; }
        if is_key_pressed(KeyCode::R) { show_visuals_panel = !show_visuals_panel; }
        if is_key_pressed(KeyCode::T) { current_visual_mode = VisualMode::Temperature; }
        if is_key_pressed(KeyCode::G) { show_generation_graph = !show_generation_graph; }
        if is_key_pressed(KeyCode::N) { current_visual_mode = VisualMode::DayNight; }
        if is_key_pressed(KeyCode::I) { current_visual_mode = VisualMode::Tribes; }
        if is_key_pressed(KeyCode::W) { current_visual_mode = VisualMode::Water; }
        if is_key_pressed(KeyCode::Up) { pending_speed_change += 1; }
        if is_key_pressed(KeyCode::Down) { pending_speed_change -= 1; }
        if is_key_pressed(KeyCode::Tab) { 
            show_inspector = !show_inspector; 
            selected_agent = None; // Reset detail view when closing
        }

        let (_, mouse_wheel_y) = mouse_wheel();

        let (mx, my) = mouse_position();
        let left_clicked = is_mouse_button_pressed(MouseButton::Left);
        let is_mouse_down = is_mouse_button_down(MouseButton::Left);
        let frame_time = get_frame_time();

        if is_mouse_button_released(MouseButton::Left) {
            active_config_button = None;
            config_button_hold_time = 0.0;
        }
        
        let current_touches = touches();
        let is_multi_touch = current_touches.len() >= 2;
        
        if is_multi_touch {
            let p1 = current_touches[0].position;
            let p2 = current_touches[1].position;
            let current_dist = ((p1.x - p2.x).powi(2) + (p1.y - p2.y).powi(2)).sqrt();
            
            if let Some(last_dist) = last_touch_distance {
                zoom *= current_dist / last_dist;
            }
            last_touch_distance = Some(current_dist);
        } else {
            last_touch_distance = None;
        }

        if !show_inspector && !show_config_panel {
            if mouse_wheel_y > 0.0 { zoom *= 1.1; }
            if mouse_wheel_y < 0.0 { zoom *= 0.9; }
            
            if is_mouse_button_down(MouseButton::Left) && followed_agent_id.is_none() && !is_multi_touch {
                let mut hit_ui = false;
                if show_visuals_panel && mx > 280.0 && mx < 480.0 && my > 10.0 && my < 370.0 { hit_ui = true; }
                
                #[cfg(target_os = "android")]
                {
                    // Prevent map panning when touching the top-right Android UI buttons
                    if my > 60.0 && my < 100.0 && mx > screen_width() - 220.0 && mx < screen_width() - 10.0 {
                        hit_ui = true;
                    }
                }

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
            VisualMode::Infrastructure => 9,
            VisualMode::Temperature => 10,
            VisualMode::DayNight => 11,
            VisualMode::Tribes => 12,
            VisualMode::Water => 13,
        };

        // --- Sync & Read State ---
        // Use try_lock to ensure the 60FPS UI loop never blocks on a heavy simulation step
        if let Ok(mut data) = shared_data.try_lock() {
            data.config.visual_mode = mode_u32;
            if config_changed_by_ui {
                let current_tick = data.config.current_tick;
                data.config = local_config;
                data.config.current_tick = current_tick;
                data.config.visual_mode = mode_u32;
                config_changed_by_ui = false;
            } else {
                local_config = data.config;
            }
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
                #[cfg(not(target_os = "android"))]
                {
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
                }
                pending_save_agents = false;
            }
            
            if pending_save_config {
                #[cfg(not(target_os = "android"))]
                {
                    if let Ok(json) = serde_json::to_string_pretty(&data.config) {
                        let _ = std::fs::write("sim_config.json", json);
                    }
                }
                pending_save_config = false;
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
                x: a.x, y: a.y, health: a.health, food: a.food, age: a.age, wealth: a.wealth, gender: a.gender, is_pregnant: a.is_pregnant,
                pheno_r: a.pheno_r, pheno_g: a.pheno_g, pheno_b: a.pheno_b
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
        } else {
            // If lock failed, we still want to keep the UI smooth
            // We'll just skip the sync this frame and use previous data
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
                if current_visual_mode == VisualMode::Default {
                    draw_circle(a.x, a.y, 1.5, Color::new(0.8, 0.2, 0.2, 0.6)); // Draw corpses visibly but faint
                }
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
                VisualMode::Tribes => Color::new(a.pheno_r * 0.5 + 0.5, a.pheno_g * 0.5 + 0.5, a.pheno_b * 0.5 + 0.5, 1.0),
                VisualMode::Default | VisualMode::Infrastructure | VisualMode::Temperature | VisualMode::DayNight | VisualMode::Water => WHITE,
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

        #[cfg(target_os = "android")]
        {
            let btn_w = 100.0;
            let btn_h = 40.0;
            let spacing = 10.0;
            let right_edge = screen_width() - spacing;
            
            // --- Row 1 (y = 60) ---
            let vis_x = right_edge - btn_w;
            let vis_y = 60.0;
            let vis_hover = mx > vis_x && mx < vis_x + btn_w && my > vis_y && my < vis_y + btn_h;
            draw_rectangle(vis_x, vis_y, btn_w, btn_h, if vis_hover { Color::new(0.4, 0.4, 0.4, 0.9) } else { Color::new(0.2, 0.2, 0.2, 0.9) });
            draw_rectangle_lines(vis_x, vis_y, btn_w, btn_h, 2.0, WHITE);
            draw_text("VISUALS", vis_x + 16.0, vis_y + 26.0, 18.0, WHITE);
            if left_clicked && vis_hover { show_visuals_panel = !show_visuals_panel; }

            let cfg_x = vis_x - btn_w - spacing;
            let cfg_y = 60.0;
            let cfg_hover = mx > cfg_x && mx < cfg_x + btn_w && my > cfg_y && my < cfg_y + btn_h;
            draw_rectangle(cfg_x, cfg_y, btn_w, btn_h, if cfg_hover { Color::new(0.4, 0.4, 0.4, 0.9) } else { Color::new(0.2, 0.2, 0.2, 0.9) });
            draw_rectangle_lines(cfg_x, cfg_y, btn_w, btn_h, 2.0, WHITE);
            draw_text("CONFIG", cfg_x + 22.0, cfg_y + 26.0, 18.0, WHITE);
            if left_clicked && cfg_hover { show_config_panel = !show_config_panel; }

            // --- Row 2 (y = 110) ---
            let r2_y = 110.0;
            let graph_x = right_edge - btn_w;
            let agt_x = graph_x - btn_w - spacing;

            let graph_hover = mx > graph_x && mx < graph_x + btn_w && my > r2_y && my < r2_y + btn_h;
            draw_rectangle(graph_x, r2_y, btn_w, btn_h, if graph_hover { Color::new(0.4, 0.4, 0.4, 0.9) } else { Color::new(0.2, 0.2, 0.2, 0.9) });
            draw_rectangle_lines(graph_x, r2_y, btn_w, btn_h, 2.0, WHITE);
            draw_text("GRAPH", graph_x + 22.0, r2_y + 26.0, 18.0, WHITE);
            if left_clicked && graph_hover { show_generation_graph = !show_generation_graph; }

            let agt_hover = mx > agt_x && mx < agt_x + btn_w && my > r2_y && my < r2_y + btn_h;
            draw_rectangle(agt_x, r2_y, btn_w, btn_h, if agt_hover { Color::new(0.4, 0.4, 0.4, 0.9) } else { Color::new(0.2, 0.2, 0.2, 0.9) });
            draw_rectangle_lines(agt_x, r2_y, btn_w, btn_h, 2.0, WHITE);
            draw_text("AGENTS", agt_x + 18.0, r2_y + 26.0, 18.0, WHITE);
            if left_clicked && agt_hover { show_inspector = !show_inspector; selected_agent = None; }

            // --- Row 3 (y = 160) ---
            let r3_y = 160.0;
            let spd_p_x = right_edge - 45.0;
            let spd_m_x = spd_p_x - 45.0 - spacing;
            let pause_x = spd_m_x - btn_w - spacing;

            let pause_hover = mx > pause_x && mx < pause_x + btn_w && my > r3_y && my < r3_y + btn_h;
            draw_rectangle(pause_x, r3_y, btn_w, btn_h, if pause_hover { Color::new(0.4, 0.4, 0.4, 0.9) } else { Color::new(0.2, 0.2, 0.2, 0.9) });
            draw_rectangle_lines(pause_x, r3_y, btn_w, btn_h, 2.0, WHITE);
            draw_text(if paused { "RESUME" } else { "PAUSE" }, pause_x + 18.0, r3_y + 26.0, 18.0, if paused { GREEN } else { YELLOW });
            if left_clicked && pause_hover { pending_pause_toggle = !pending_pause_toggle; }

            let spd_m_hover = mx > spd_m_x && mx < spd_m_x + 45.0 && my > r3_y && my < r3_y + btn_h;
            draw_rectangle(spd_m_x, r3_y, 45.0, btn_h, if spd_m_hover { Color::new(0.4, 0.4, 0.4, 0.9) } else { Color::new(0.2, 0.2, 0.2, 0.9) });
            draw_rectangle_lines(spd_m_x, r3_y, 45.0, btn_h, 2.0, WHITE);
            draw_text("<<", spd_m_x + 12.0, r3_y + 26.0, 18.0, WHITE);
            if left_clicked && spd_m_hover { pending_speed_change -= 1; }

            let spd_p_hover = mx > spd_p_x && mx < spd_p_x + 45.0 && my > r3_y && my < r3_y + btn_h;
            draw_rectangle(spd_p_x, r3_y, 45.0, btn_h, if spd_p_hover { Color::new(0.4, 0.4, 0.4, 0.9) } else { Color::new(0.2, 0.2, 0.2, 0.9) });
            draw_rectangle_lines(spd_p_x, r3_y, 45.0, btn_h, 2.0, WHITE);
            draw_text(">>", spd_p_x + 12.0, r3_y + 26.0, 18.0, WHITE);
            if left_clicked && spd_p_hover { pending_speed_change += 1; }
        }

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
            VisualMode::Infrastructure => {
                draw_text("Roads (Spd)", legend_x + 10.0, legend_y + 35.0, 16.0, GRAY);
                draw_text("Houses (Rest)", legend_x + 10.0, legend_y + 60.0, 16.0, ORANGE);
                draw_text("Farms (Food)", legend_x + 10.0, legend_y + 85.0, 16.0, GREEN);
                draw_text("Granary (Store)", legend_x + 10.0, legend_y + 110.0, 16.0, MAGENTA);
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
            VisualMode::Tribes => {
                draw_text("Similar Colors =", legend_x + 10.0, legend_y + 60.0, 20.0, WHITE);
                draw_text("Same Tribe / Kin", legend_x + 10.0, legend_y + 90.0, 20.0, Color::new(0.0, 1.0, 0.5, 1.0));
            },
            VisualMode::Water => {
                draw_text("High Water", legend_x + 10.0, legend_y + 60.0, 20.0, Color::new(0.0, 0.8, 1.0, 1.0));
                draw_text("Dry / Saltwater", legend_x + 10.0, legend_y + 90.0, 20.0, DARKGRAY);
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
        
        ui::draw_metrics(pop_count, compute_time, speed, ticks, loaded_config.tick_to_mins, get_fps(), avg_fps, low_1_fps, current_visual_mode, show_inspector, show_generation_graph, show_config_panel, paused, restart_msg);
        if show_visuals_panel { ui::draw_visuals_panel(mx, my, left_clicked, &mut current_visual_mode); }
        if show_generation_graph {
            ui::draw_generation_graph(&generation_times, loaded_config.tick_to_mins);
        }
        if show_inspector {
            ui::draw_inspector(mx, my, left_clicked, mouse_wheel_y, &mut inspector_agents, &mut sort_col, &mut sort_desc, &mut inspector_scroll, &mut selected_agent, &mut followed_agent_id, &mut show_inspector, loaded_config.tick_to_mins, loaded_config.base_speed);
        } else if let Some(a) = &followed_agent {
            ui::draw_tracker(mx, my, left_clicked, a, &mut followed_agent_id, &mut show_inspector, loaded_config.tick_to_mins);
        }
        
        // --- Draw Live Configuration Panel ---
        if show_config_panel {
            let panel_w = 360.0;
            let panel_h = (screen_height() - 40.0).min(800.0);
            let panel_x = screen_width() - panel_w - 20.0;
            let panel_y = screen_height() / 2.0 - panel_h / 2.0;
            draw_rectangle(panel_x, panel_y, panel_w, panel_h, Color::new(0.0, 0.0, 0.0, 0.9));
            draw_rectangle_lines(panel_x, panel_y, panel_w, panel_h, 1.0, Color::new(0.0, 1.0, 0.8, 1.0));
            draw_text("Configuration (Press C to close)", panel_x + 10.0, panel_y + 30.0, 20.0, WHITE);
            
            let save_btn_rect = Rect::new(panel_x + 10.0, panel_y + 50.0, 130.0, 30.0);
            draw_rectangle(save_btn_rect.x, save_btn_rect.y, save_btn_rect.w, save_btn_rect.h, DARKGRAY);
            draw_text("Save to JSON", save_btn_rect.x + 15.0, save_btn_rect.y + 20.0, 16.0, WHITE);
            
            if left_clicked && save_btn_rect.contains(vec2(mx, my)) {
                pending_save_config = true;
            }

            let mut y = panel_y + 100.0 + config_scroll;
            let row_h = 35.0;

            let mut draw_param = |label: &str, val: &mut f32, min_val: f32, max_val: f32, step: f32| {
                if y > panel_y + 80.0 && y < panel_y + panel_h - 10.0 {
                    draw_text(&format!("{}: {:.4}", label, val), panel_x + 10.0, y + 20.0, 16.0, WHITE);
                    let dec_rect = Rect::new(panel_x + 250.0, y + 5.0, 35.0, 25.0);
                    let inc_rect = Rect::new(panel_x + 295.0, y + 5.0, 35.0, 25.0);
                    draw_rectangle(dec_rect.x, dec_rect.y, dec_rect.w, dec_rect.h, RED);
                    draw_rectangle(inc_rect.x, inc_rect.y, inc_rect.w, inc_rect.h, GREEN);
                    draw_text("-", dec_rect.x + 12.0, dec_rect.y + 18.0, 20.0, WHITE);
                    draw_text("+", inc_rect.x + 12.0, inc_rect.y + 18.0, 20.0, WHITE);
                    
                    let dec_id = format!("{}_dec", label);
                    let inc_id = format!("{}_inc", label);
                    let dec_hover = dec_rect.contains(vec2(mx, my));
                    let inc_hover = inc_rect.contains(vec2(mx, my));
                    
                    let mut do_dec = false;
                    let mut do_inc = false;
                    let mut mult = 1.0;

                    if left_clicked && dec_hover {
                        active_config_button = Some(dec_id.clone());
                        config_button_hold_time = 0.0;
                        do_dec = true;
                    } else if is_mouse_down && active_config_button.as_ref() == Some(&dec_id) {
                        config_button_hold_time += frame_time;
                        if config_button_hold_time > 0.4 && dec_hover {
                            do_dec = true;
                            mult = (1.0 + (config_button_hold_time - 0.4) * 2.0).powi(2) * frame_time * 15.0;
                        }
                    }

                    if left_clicked && inc_hover {
                        active_config_button = Some(inc_id.clone());
                        config_button_hold_time = 0.0;
                        do_inc = true;
                    } else if is_mouse_down && active_config_button.as_ref() == Some(&inc_id) {
                        config_button_hold_time += frame_time;
                        if config_button_hold_time > 0.4 && inc_hover {
                            do_inc = true;
                            mult = (1.0 + (config_button_hold_time - 0.4) * 2.0).powi(2) * frame_time * 15.0;
                        }
                    }
                    
                    if do_dec { *val = (*val - step * mult).max(min_val); config_changed_by_ui = true; }
                    if do_inc { *val = (*val + step * mult).min(max_val); config_changed_by_ui = true; }
                }
                y += row_h;
            };

            draw_param("Base Speed", &mut local_config.base_speed, 0.0, 20.0, 0.5);
            draw_param("Baseline Cost", &mut local_config.baseline_cost, 0.0, 1.0, 0.005);
            draw_param("Move Cost", &mut local_config.move_cost_per_unit, 0.0, 0.1, 0.001);
            draw_param("Climb Penalty", &mut local_config.climb_penalty, 0.0, 20.0, 0.5);
            draw_param("Base Gather Rate", &mut local_config.base_gather_rate, 0.0, 5.0, 0.05);
            draw_param("Max Gather Rate", &mut local_config.max_gather_rate, 0.0, 10.0, 0.1);
            draw_param("Boat Cost", &mut local_config.boat_cost, 0.0, 50000.0, 500.0);
            draw_param("Water Transfer", &mut local_config.water_transfer_amount, 0.0, 50.0, 0.5);
            draw_param("Regeneration Rate", &mut local_config.regen_rate, 0.0, 1.0, 0.005);
            draw_param("Starvation Rate", &mut local_config.starvation_rate, 0.0, 5.0, 0.05);
            draw_param("Reproduction Cost", &mut local_config.reproduction_cost, 0.0, 5000.0, 50.0);
            draw_param("Mutation Rate", &mut local_config.mutation_rate, 0.0, 1.0, 0.01);
            draw_param("Mutation Strength", &mut local_config.mutation_strength, 0.0, 1.0, 0.01);
            draw_param("Infra Cost", &mut local_config.infra_cost, 0.0, 1000.0, 10.0);
            draw_param("Road Decay", &mut local_config.decay_rate_roads, 0.0, 1.0, 0.001);
            draw_param("Housing Decay", &mut local_config.decay_rate_housing, 0.0, 1.0, 0.001);
            draw_param("Farm Decay", &mut local_config.decay_rate_farms, 0.0, 1.0, 0.005);
            draw_param("Storage Decay", &mut local_config.decay_rate_storage, 0.0, 1.0, 0.001);
            draw_param("Base Spoilage", &mut local_config.base_spoilage_rate, 0.0, 1.0, 0.0001);
            draw_param("Attacker Damage", &mut local_config.combat_attacker_damage, 0.0, 10.0, 0.5);
            draw_param("Bystander Damage", &mut local_config.combat_bystander_damage, 0.0, 10.0, 0.5);
            draw_param("Steal Amount", &mut local_config.combat_steal_amount, 0.0, 50.0, 1.0);
            
            let total_content_h = y - (panel_y + 100.0 + config_scroll);
            let visible_h = panel_h - 110.0;
            let min_scroll = (visible_h - total_content_h).min(0.0);
            
            if mouse_wheel_y != 0.0 && mx > panel_x && mx < panel_x + panel_w && my > panel_y && my < panel_y + panel_h {
                config_scroll += mouse_wheel_y * 20.0;
            }
            config_scroll = config_scroll.clamp(min_scroll, 0.0);
        }

        next_frame().await;
    }
}