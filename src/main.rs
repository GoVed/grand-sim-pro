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
        window_width: cfg.world.display_width as i32,
        window_height: cfg.world.display_height as i32,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let loaded_config = load_config();
    let map_width = loaded_config.world.map_width;
    let map_height = loaded_config.world.map_height;
    let agent_count = loaded_config.sim.agent_count;
    let random_seed = ::rand::thread_rng().r#gen::<u32>();

    // --- Startup Load Prompt ---
    let mut founders = Vec::new();
    let saved_exists = std::fs::read_dir("saved_agents_weights").map(|d| d.count() > 0).unwrap_or(false);

    if saved_exists {
        match loaded_config.sim.load_saved_agents_on_start {
            1 => { founders = crate::simulation::load_founders(&loaded_config); } // Always
            2 => { // Ask
                let mut choice = None;
                while choice.is_none() {
                    clear_background(DARKGRAY);
                    let text = "SAVED AGENT WEIGHTS DETECTED";
                    let dims = measure_text(text, None, 40, 1.0);
                    draw_text(text, screen_width()/2.0 - dims.width/2.0, screen_height()/2.0 - 60.0, 40.0, YELLOW);

                    draw_text("Load existing evolved brains into new simulation?", screen_width()/2.0 - 250.0, screen_height()/2.0, 20.0, WHITE);

                    let btn_y = screen_height()/2.0 + 60.0;
                    let yes_rect = Rect::new(screen_width()/2.0 - 150.0, btn_y, 120.0, 40.0);
                    let no_rect = Rect::new(screen_width()/2.0 + 30.0, btn_y, 120.0, 40.0);

                    let mx = mouse_position().0;
                    let my = mouse_position().1;
                    let click = is_mouse_button_pressed(MouseButton::Left);

                    let y_hover = yes_rect.contains(vec2(mx, my));
                    draw_rectangle(yes_rect.x, yes_rect.y, yes_rect.w, yes_rect.h, if y_hover { GREEN } else { Color::new(0.0, 0.4, 0.0, 1.0) });
                    draw_text("YES [Y]", yes_rect.x + 25.0, yes_rect.y + 28.0, 20.0, WHITE);
                    if click && y_hover || is_key_pressed(KeyCode::Y) { choice = Some(true); }

                    let n_hover = no_rect.contains(vec2(mx, my));
                    draw_rectangle(no_rect.x, no_rect.y, no_rect.w, no_rect.h, if n_hover { RED } else { Color::new(0.4, 0.0, 0.0, 1.0) });
                    draw_text("NO [N]", no_rect.x + 30.0, no_rect.y + 28.0, 20.0, WHITE);
                    if click && n_hover || is_key_pressed(KeyCode::N) { choice = Some(false); }

                    next_frame().await;
                }
                if choice.unwrap() { founders = crate::simulation::load_founders(&loaded_config); }
            }
            _ => {} // Never (0)
        }
    }

    let (tx, rx) = std::sync::mpsc::channel();
    let loaded_config_clone = loaded_config;
    std::thread::spawn(move || {
        let sim = SimulationManager::new(map_width, map_height, random_seed, agent_count, &loaded_config_clone, founders);
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
        last_saved_config: loaded_config,
        is_paused: false,
        restart_message_active: false,
        ticks_per_loop: 5,
        total_ticks: 0,
        last_compute_time_micros: 0,
        ticks_per_second: 0.0,
        generation_survival_times: Vec::new(),
    }));

    let sim_thread_data = shared_data.clone();
    
    let initial_agents = sim_thread_data.lock().unwrap().sim.agents.clone();
    let height_map = sim_thread_data.lock().unwrap().sim.env.height_map.clone();
    let init_cells = sim_thread_data.lock().unwrap().sim.env.map_cells.clone();

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
    let mut compute_time: f32 = 0.0;
    let mut ticks_per_sec: f32 = 0.0;
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
    let mut config_search_query = String::new();
    let mut last_saved_config = loaded_config;
    let mut pending_save_config = false;
    let mut local_config = loaded_config;
    let mut config_changed_by_ui = false;
    let mut active_config_button: Option<String> = None;
    let mut config_button_hold_time: f32 = 0.0;
    
    let mut followed_agent_id: Option<u32> = None;
    let mut followed_agent: Option<crate::agent::Person> = None;
    let mut generation_times: Vec<u64> = Vec::new();

    loop {
        // Register UI inputs instantly
        if is_key_pressed(KeyCode::C) { 
            show_config_panel = !show_config_panel; 
            // Clear char queue when toggling config to avoid 'c' getting into search box
            while get_char_pressed().is_some() {}
        }
        
        if !show_config_panel {
            if is_key_pressed(KeyCode::Space) { pending_pause_toggle = !pending_pause_toggle; }
            if is_key_pressed(KeyCode::S) { pending_save_agents = true; }
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
        
        if !show_inspector && !show_config_panel {
            if mouse_wheel_y > 0.0 { zoom *= 1.1; }
            if mouse_wheel_y < 0.0 { zoom *= 0.9; }
            
            if is_mouse_button_down(MouseButton::Left) && followed_agent_id.is_none() {
                let mut hit_ui = false;
                if show_visuals_panel && mx > 280.0 && mx < 480.0 && my > 10.0 && my < 370.0 { hit_ui = true; }
                
                if !hit_ui {
                    offset_x += (mx - last_mouse.0) / zoom;
                    offset_y -= (my - last_mouse.1) / zoom;
                }
            }
        }
        last_mouse = (mx, my);

        let mode_u32 = match current_visual_mode {
            VisualMode::Default => 0, VisualMode::Resources => 1, VisualMode::Age => 2, VisualMode::Gender => 3,
            VisualMode::Pregnancy => 4, VisualMode::MarketWealth => 5, VisualMode::MarketFood => 6, VisualMode::AskPrice => 7,
            VisualMode::BidPrice => 8, VisualMode::Infrastructure => 9, VisualMode::Temperature => 10, VisualMode::DayNight => 11,
            VisualMode::Tribes => 12, VisualMode::Water => 13,
        };

        // --- Sync & Read State ---
        if let Ok(mut data) = shared_data.try_lock() {
            data.config.sim.visual_mode = mode_u32;
            if config_changed_by_ui {
                let current_tick = data.config.sim.current_tick;
                data.config = local_config;
                data.config.sim.current_tick = current_tick;
                data.config.sim.visual_mode = mode_u32;
                config_changed_by_ui = false;
            } else {
                local_config = data.config;
            }
            if pending_pause_toggle {
                data.is_paused = !data.is_paused;
                pending_pause_toggle = false;
            }
            if pending_speed_change != 0 {
                if pending_speed_change > 0 { data.ticks_per_loop = (data.ticks_per_loop as f32 * 1.5) as usize; }
                else { data.ticks_per_loop = (data.ticks_per_loop as f32 / 1.5).max(5.0) as usize; }
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
                let save_count = living.len().min(data.config.sim.founder_count as usize);
                for i in 0..save_count {
                    let weights = living[i].extract_weights();
                    if let Ok(json) = serde_json::to_string_pretty(&weights) {
                        let _ = std::fs::write(format!("saved_agents_weights/agent_{}.json", i), json);
                    }
                }
                pending_save_agents = false;
            }
            
            if pending_save_config {
                if let Ok(json) = serde_json::to_string_pretty(&data.config) {
                    if let Ok(_) = std::fs::write("sim_config.json", json) {
                        data.last_saved_config = data.config;
                    }
                }
                pending_save_config = false;
            }
            
            paused = data.is_paused;
            speed = data.ticks_per_loop;
            ticks = data.total_ticks;
            compute_time = data.last_compute_time_micros as f32 / 1000.0;
            ticks_per_sec = data.ticks_per_second;
            pop_count = data.sim.agents.iter().filter(|a| a.health > 0.0).count();
            restart_msg = data.restart_message_active;
            last_saved_config = data.last_saved_config;
            
            if show_generation_graph { generation_times = data.generation_survival_times.clone(); }

            local_agent_coords.clear();
            local_agent_coords.extend(data.sim.agents.iter().map(|a| AgentRenderData {
                x: a.x, y: a.y, health: a.health, food: a.food, age: a.age, wealth: a.wealth, gender: a.gender, is_pregnant: a.is_pregnant,
                pheno_r: a.pheno_r, pheno_g: a.pheno_g, pheno_b: a.pheno_b
            }));
            
            if show_inspector {
                if inspector_agents.len() != data.sim.agents.iter().filter(|a| a.health > 0.0).count() {
                    inspector_agents.clear();
                    for (i, a) in data.sim.agents.iter().enumerate() {
                        if a.health > 0.0 { inspector_agents.push((i, *a)); }
                    }
                    ui::apply_sort(&mut inspector_agents, sort_col, sort_desc);
                } else {
                    for i in 0..inspector_agents.len() {
                        let original_idx = inspector_agents[i].0;
                        inspector_agents[i].1 = data.sim.agents[original_idx];
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

        clear_background(BLACK);
        let ticks_per_day = 24.0 * 60.0 / loaded_config.world.tick_to_mins;
        let day_cycle_progress = (ticks as f32 % ticks_per_day) / ticks_per_day;

        let cam = Camera2D {
            target: vec2(map_width as f32 / 2.0 - offset_x, map_height as f32 / 2.0 - offset_y),
            zoom: vec2(zoom / (screen_width() / 2.0), -zoom / (screen_height() / 2.0)), 
            ..Default::default()
        };
        set_camera(&cam);

        let render_data = gpu.fetch_render(map_width, map_height);
        image.bytes.copy_from_slice(&render_data);
        texture.update(&image);
        draw_texture(&texture, 0.0, 0.0, WHITE);

        let max_inv_ln = (loaded_config.eco.boat_cost * 0.25 + 1.0).ln();
        for a in &local_agent_coords {
            if a.health <= 0.0 {
                if current_visual_mode == VisualMode::Default {
                    draw_circle(a.x, a.y, 1.5, Color::new(0.8, 0.2, 0.2, 0.6));
                }
                continue;
            }
            let radius = 1.0 + (a.age / loaded_config.bio.puberty_age).min(1.0) * 1.0;
            let color = match current_visual_mode {
                VisualMode::Resources | VisualMode::MarketWealth | VisualMode::MarketFood | VisualMode::AskPrice | VisualMode::BidPrice => {
                    let val = if current_visual_mode == VisualMode::MarketFood { a.food / 1000.0 } else { a.wealth };
                    let ratio = ((val.max(0.0) + 1.0).ln() / max_inv_ln).clamp(0.0, 1.0);
                    Color::new(1.0 - ratio, ratio, 0.0, 1.0)
                },
                VisualMode::Age => {
                    let ratio = (a.age / loaded_config.bio.max_age).clamp(0.0, 1.0);
                    Color::new(ratio, 1.0 - ratio, 1.0, 1.0)
                },
                VisualMode::Gender => if a.gender > 0.5 { Color::new(0.2, 0.6, 1.0, 1.0) } else { Color::new(1.0, 0.4, 0.7, 1.0) },
                VisualMode::Pregnancy => if a.is_pregnant > 0.5 { Color::new(1.0, 0.8, 0.0, 1.0) } else if a.gender > 0.5 { Color::new(0.2, 0.4, 0.8, 0.5) } else { Color::new(0.8, 0.2, 0.5, 0.5) },
                VisualMode::Tribes => Color::new(a.pheno_r * 0.5 + 0.5, a.pheno_g * 0.5 + 0.5, a.pheno_b * 0.5 + 0.5, 1.0),
                _ => WHITE,
            };
            draw_circle(a.x, a.y, radius, color);
        }

        if let Some(ref a) = followed_agent { draw_circle_lines(a.x, a.y, 8.0, 2.0, YELLOW); }

        set_default_camera();
        let clock_x = screen_width() - 220.0;
        let hours = (day_cycle_progress * 24.0).floor() as u32;
        let minutes = ((day_cycle_progress * 24.0).fract() * 60.0).floor() as u32;
        draw_rectangle(clock_x, 10.0, 210.0, 40.0, Color::new(0.0, 0.0, 0.0, 0.7));
        draw_text(&format!("Time: {:02}:{:02}", hours, minutes), clock_x + 10.0, 38.0, 28.0, WHITE);

        frame_times.push(get_frame_time());
        if frame_times.len() > 300 { frame_times.remove(0); }
        let mut avg_fps = 0.0;
        let mut low_1_fps = 0.0;
        if !frame_times.is_empty() {
            let sum_time: f32 = frame_times.iter().sum();
            avg_fps = frame_times.len() as f32 / sum_time;
            let mut sorted = frame_times.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let count_1_percent = (sorted.len() / 100).max(1);
            let low_1_sum: f32 = sorted.iter().rev().take(count_1_percent).sum();
            low_1_fps = count_1_percent as f32 / low_1_sum;
        }
        
        ui::draw_metrics(pop_count, compute_time, ticks_per_sec, speed, ticks, loaded_config.world.tick_to_mins, get_fps(), avg_fps, low_1_fps, current_visual_mode, show_inspector, show_generation_graph, show_config_panel, paused, restart_msg);
        if show_visuals_panel { ui::draw_visuals_panel(mx, my, left_clicked, &mut current_visual_mode); }
        if show_generation_graph { ui::draw_generation_graph(&generation_times, loaded_config.world.tick_to_mins); }
        if show_inspector {
            ui::draw_inspector(mx, my, left_clicked, mouse_wheel_y, &mut inspector_agents, &mut sort_col, &mut sort_desc, &mut inspector_scroll, &mut selected_agent, &mut followed_agent_id, &mut show_inspector, loaded_config.world.tick_to_mins);
        } else if let Some(a) = &followed_agent {
            ui::draw_tracker(mx, my, left_clicked, a, &mut followed_agent_id, &mut show_inspector, loaded_config.world.tick_to_mins);
        }
        
        if show_config_panel {
            ui::draw_config_panel(
                mx, my, left_clicked, is_mouse_down, frame_time,
                &mut local_config, &last_saved_config, &mut config_scroll, &mut config_search_query,
                &mut config_changed_by_ui, &mut pending_save_config,
                &mut active_config_button, &mut config_button_hold_time,
            );
        }
        next_frame().await;
    }
}
