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
use world_sim::agent::Person;
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
use shared::{SharedData, VisualMode, SortCol, AgentRenderData, FullState, save_everything, load_everything, ProgressReport};

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

fn list_saves() -> Vec<String> {
    let mut saves = Vec::new();
    if let Ok(entries) = std::fs::read_dir("saves") {
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    saves.push(name.to_string());
                }
            }
        }
    }
    saves.sort();
    saves.reverse();
    saves
}

fn draw_progress_bar(report: &ProgressReport) {
    let w = 400.0;
    let h = 30.0;
    let x = screen_width() / 2.0 - w / 2.0;
    let y = screen_height() / 2.0 + 50.0;
    
    draw_rectangle(x - 5.0, y - 5.0, w + 10.0, h + 10.0, DARKGRAY);
    draw_rectangle(x, y, w, h, BLACK);
    draw_rectangle(x, y, w * report.progress, h, Color::new(0.0, 0.8, 0.6, 1.0));
    
    let text = &report.message;
    let dims = measure_text(text, None, 20, 1.0);
    draw_text(text, screen_width() / 2.0 - dims.width / 2.0, y - 10.0, 20.0, WHITE);
    
    let pct = format!("{:.0}%", report.progress * 100.0);
    let dims_p = measure_text(&pct, None, 18, 1.0);
    draw_text(&pct, x + w / 2.0 - dims_p.width / 2.0, y + 22.0, 18.0, WHITE);
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut loaded_config = load_config();
    
    // --- Initial Screen State ---
    #[derive(Clone, Copy, PartialEq)]
    enum MenuState { Main, SelectSave, Loading, Running }
    #[derive(Clone, Copy, PartialEq)]
    enum StartChoice { Founders, SaveFile, LastRun, New }
    
    let mut menu_state = MenuState::Main;
    let mut start_choice = None;
    let mut selected_save_path = None;

    while menu_state != MenuState::Running {
        clear_background(DARKGRAY);
        
        match menu_state {
            MenuState::Main => {
                let text = "GRAND SIM PRO";
                let dims = measure_text(text, None, 60, 1.0);
                draw_text(text, screen_width()/2.0 - dims.width/2.0, 100.0, 60.0, Color::new(0.0, 1.0, 0.8, 1.0));

                let saves_exist = std::path::Path::new("saves").exists() && std::fs::read_dir("saves").map(|d| d.count() > 0).unwrap_or(false);
                let last_run_exists = std::path::Path::new("last_run").exists();
                let founders_exist = std::path::Path::new("saved_agents_weights").exists() && std::fs::read_dir("saved_agents_weights").map(|d| d.count() > 0).unwrap_or(false);

                let options = [
                    (StartChoice::Founders, "1. LOAD FROM FOUNDERS", founders_exist),
                    (StartChoice::SaveFile, "2. LOAD FROM SAVE FILE", saves_exist),
                    (StartChoice::LastRun, "3. CONTINUE LAST RUN", last_run_exists),
                    (StartChoice::New, "4. START NEW SIMULATION", true),
                ];

                let mut y = 200.0;
                let mx = mouse_position().0;
                let my = mouse_position().1;
                let click = is_mouse_button_pressed(MouseButton::Left);

                for (choice, label, enabled) in options.iter() {
                    let rect = Rect::new(screen_width()/2.0 - 150.0, y, 300.0, 40.0);
                    let hover = rect.contains(vec2(mx, my)) && *enabled;
                    let color = if !*enabled { GRAY } else if hover { Color::new(0.0, 0.8, 0.7, 1.0) } else { Color::new(0.0, 0.4, 0.3, 1.0) };
                    
                    draw_rectangle(rect.x, rect.y, rect.w, rect.h, color);
                    draw_text(label, rect.x + 20.0, rect.y + 28.0, 20.0, if *enabled { WHITE } else { DARKGRAY });
                    
                    if click && hover { 
                        start_choice = Some(*choice);
                        if *choice == StartChoice::SaveFile { menu_state = MenuState::SelectSave; }
                        else { menu_state = MenuState::Loading; }
                    }
                    y += 60.0;
                }
            }
            MenuState::SelectSave => {
                let text = "SELECT SAVE FILE";
                let dims = measure_text(text, None, 40, 1.0);
                draw_text(text, screen_width()/2.0 - dims.width/2.0, 80.0, 40.0, Color::new(0.0, 1.0, 1.0, 1.0));
                
                let saves = list_saves();
                let mut y = 150.0;
                let mx = mouse_position().0;
                let my = mouse_position().1;
                let click = is_mouse_button_pressed(MouseButton::Left);

                for save_name in saves {
                    let rect = Rect::new(screen_width()/2.0 - 200.0, y, 400.0, 35.0);
                    let hover = rect.contains(vec2(mx, my));
                    draw_rectangle(rect.x, rect.y, rect.w, rect.h, if hover { Color::new(0.2, 0.2, 0.2, 1.0) } else { BLACK });
                    draw_text(&save_name, rect.x + 10.0, rect.y + 25.0, 18.0, if hover { Color::new(0.0, 1.0, 1.0, 1.0) } else { WHITE });
                    
                    if click && hover {
                        selected_save_path = Some(std::path::PathBuf::from("saves").join(save_name));
                        menu_state = MenuState::Loading;
                    }
                    y += 40.0;
                    if y > screen_height() - 100.0 { break; }
                }
                if is_key_pressed(KeyCode::Escape) { menu_state = MenuState::Main; }
            }
            MenuState::Loading => break,
            _ => {}
        }
        next_frame().await;
    }

    let (final_tx, final_rx) = std::sync::mpsc::channel::<SharedData>();
    let (prog_tx, prog_rx) = std::sync::mpsc::channel::<ProgressReport>();
    
    let choice = start_choice.unwrap();
    let save_path = selected_save_path.clone();
    let config = loaded_config.clone();

    std::thread::spawn(move || {
        let mut initial_full_state: Option<FullState> = None;
        let mut founders = Vec::new();

        match choice {
            StartChoice::Founders => {
                let _ = std::fs::remove_file("telemetry.csv");
                founders = crate::simulation::load_founders(&config);
            }
            StartChoice::SaveFile => { if let Some(path) = save_path { initial_full_state = load_everything(&path, prog_tx.clone()); } }
            StartChoice::LastRun => { initial_full_state = load_everything(std::path::Path::new("last_run"), prog_tx.clone()); }
            StartChoice::New => { let _ = std::fs::remove_file("telemetry.csv"); }
        }

        let shared = if let Some(fs) = initial_full_state {
            fs.shared
        } else {
            let sim = SimulationManager::new(config.world.map_width, config.world.map_height, ::rand::thread_rng().r#gen(), config.sim.agent_count, &config, founders);
            SharedData {
                sim, config: config.clone(), last_saved_config: config.clone(),
                is_paused: false, restart_message_active: false,
                ticks_per_loop: 5, total_ticks: 0, 
                cumulative_ticks: 0, last_telemetry_tick: 0,
                cumulative_births: config.sim.agent_count as u64,
                cumulative_deaths: 0,
                last_compute_time_micros: 0,
                ticks_per_second: 0.0, generation_survival_times: Vec::new(),
            }
        };
        let _ = final_tx.send(shared);
    });

    let mut current_report = ProgressReport { message: "Initializing...".to_string(), progress: 0.0 };
    let initial_shared = loop {
        clear_background(DARKGRAY);
        while let Ok(r) = prog_rx.try_recv() { current_report = r; }
        draw_progress_bar(&current_report);
        if let Ok(s) = final_rx.try_recv() { break s; }
        next_frame().await;
    };

    loaded_config = initial_shared.config;
    let shared_data = Arc::new(Mutex::new(initial_shared));
    let sim_thread_data = shared_data.clone();
    
    let (initial_states, initial_genetics, height_map, init_cells) = {
        let lock = sim_thread_data.lock().unwrap();
        (lock.sim.states.clone(), lock.sim.genetics.clone(), lock.sim.env.height_map.clone(), lock.sim.env.map_cells.clone())
    };

    let gpu = Arc::new(GpuEngine::new(&initial_states, &initial_genetics, &height_map, &init_cells, &loaded_config));
    sim_thread::spawn(sim_thread_data.clone(), gpu.clone());

    let mut image = Image::gen_image_color(loaded_config.world.map_width as u16, loaded_config.world.map_height as u16, BLANK);
    let texture = Texture2D::from_image(&image);
    
    let mut zoom = 1.0f32; let mut offset_x = 0.0f32; let mut offset_y = 0.0f32;
    let mut last_mouse = mouse_position();
    let mut local_agent_coords: Vec<AgentRenderData> = Vec::with_capacity(loaded_config.sim.agent_count as usize);
    
    let mut show_visuals_panel = false;
    let mut current_visual_mode = VisualMode::Default;
    let mut show_inspector = false;
    let mut show_generation_graph = false;
    let mut sort_col = SortCol::Food;
    let mut sort_desc = true;
    let mut inspector_scroll: usize = 0;
    let mut selected_agent: Option<crate::agent::Person> = None;
    let mut inspector_agents: Vec<(usize, crate::agent::Person)> = Vec::new();
    let mut show_config_panel = false;
    let mut config_scroll = 0.0;
    let mut config_search_query = String::new();
    let mut active_config_button: Option<String> = None;
    let mut config_button_hold_time: f32 = 0.0;
    let mut followed_agent_id: Option<u32> = None;

    let mut saving_ui_report: Option<ProgressReport> = None;
    let (save_prog_tx, save_prog_rx) = std::sync::mpsc::channel::<ProgressReport>();

    let mut metrics = (0, 0.0f32, 0.0f32, 0usize, 0u64, false, false, Vec::new());
    let mut gen_graph_scroll = 0.0f32;
    let mut gen_graph_zoom = 1.0f32;
    let mut gen_graph_avg_period = 25usize;

    loop {
        if saving_ui_report.is_some() {
            clear_background(BLACK);
            while let Ok(r) = save_prog_rx.try_recv() {
                if r.progress >= 1.0 { saving_ui_report = None; break; }
                saving_ui_report = Some(r);
            }
            if let Some(ref r) = saving_ui_report {
                draw_progress_bar(r);
                next_frame().await;
                continue;
            }
        }

        if is_key_pressed(KeyCode::C) { show_config_panel = !show_config_panel; while get_char_pressed().is_some() {} }
        if !show_config_panel {
            if is_key_pressed(KeyCode::Space) { let mut d = shared_data.lock().unwrap(); d.is_paused = !d.is_paused; }
            if is_key_pressed(KeyCode::S) { 
                let data = shared_data.lock().unwrap();
                let _ = std::fs::create_dir_all("saved_agents_weights");
                let mut living: Vec<_> = data.sim.states.iter().enumerate().filter(|(_, s)| s.health > 0.0).collect();
                living.sort_by(|(_, a), (_, b)| (b.wealth + b.food).partial_cmp(&(a.wealth + a.food)).unwrap());
                for i in 0..living.len().min(data.config.sim.founder_count as usize) {
                    let (_, s) = living[i];
                    let p = Person { state: *s, genetics: data.sim.genetics[s.genetics_index as usize] };
                    let _ = std::fs::write(format!("saved_agents_weights/agent_{}.json", i), serde_json::to_string_pretty(&p.extract_weights()).unwrap());
                }
            }
            if is_key_pressed(KeyCode::F) {
                let mut data = shared_data.lock().unwrap();
                data.sim.env.map_cells = gpu.fetch_cells();
                let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
                let save_name = format!("save_{}", timestamp);
                let shared_clone = data.clone();
                let tx = save_prog_tx.clone();
                saving_ui_report = Some(ProgressReport { message: "Starting Save...".to_string(), progress: 0.0 });
                std::thread::spawn(move || { pollster::block_on(save_everything(&shared_clone, &save_name, false, Some(tx))); });
            }
            if is_key_pressed(KeyCode::R) { show_visuals_panel = !show_visuals_panel; }
            if is_key_pressed(KeyCode::G) { show_generation_graph = !show_generation_graph; }
            if is_key_pressed(KeyCode::Up) { let mut d = shared_data.lock().unwrap(); d.ticks_per_loop = (d.ticks_per_loop as f32 * 1.5) as usize; }
            if is_key_pressed(KeyCode::Down) { let mut d = shared_data.lock().unwrap(); d.ticks_per_loop = (d.ticks_per_loop as f32 / 1.5).max(5.0) as usize; }
            if is_key_pressed(KeyCode::Tab) { show_inspector = !show_inspector; selected_agent = None; }
        }

        let (_, mouse_wheel_y) = mouse_wheel();
        let (mx, my) = mouse_position();
        let left_clicked = is_mouse_button_pressed(MouseButton::Left);
        
        if let Ok(mut data) = shared_data.try_lock() {
            data.config.sim.visual_mode = match current_visual_mode {
                VisualMode::Default => 0, VisualMode::Resources => 1, VisualMode::Age => 2, VisualMode::Gender => 3,
                VisualMode::Pregnancy => 4, VisualMode::MarketWealth => 5, VisualMode::MarketFood => 6, VisualMode::AskPrice => 7,
                VisualMode::BidPrice => 8, VisualMode::Infrastructure => 9, VisualMode::Temperature => 10, VisualMode::DayNight => 11,
                VisualMode::Tribes => 12, VisualMode::Water => 13,
            };
            metrics = (data.sim.states.iter().filter(|s| s.health > 0.0).count(), data.last_compute_time_micros as f32 / 1000.0, data.ticks_per_second, data.ticks_per_loop, data.total_ticks, data.is_paused, data.restart_message_active, data.generation_survival_times.clone());
            local_agent_coords.clear();
            local_agent_coords.extend(data.sim.states.iter().map(|s| AgentRenderData { x: s.x, y: s.y, health: s.health, food: s.food, age: s.age, wealth: s.wealth, gender: s.gender, is_pregnant: s.is_pregnant, pheno_r: s.pheno_r, pheno_g: s.pheno_g, pheno_b: s.pheno_b }));
            if show_inspector {
                if inspector_agents.len() != metrics.0 {
                    inspector_agents.clear();
                    for (i, s) in data.sim.states.iter().enumerate() { if s.health > 0.0 { inspector_agents.push((i, Person { state: *s, genetics: data.sim.genetics[s.genetics_index as usize] })); } }
                    ui::apply_sort(&mut inspector_agents, sort_col, sort_desc);
                } else { for i in 0..inspector_agents.len() { let s = &data.sim.states[inspector_agents[i].0]; inspector_agents[i].1 = Person { state: *s, genetics: data.sim.genetics[s.genetics_index as usize] }; } }
                if let Some(ref mut sel) = selected_agent { if let Some(s) = data.sim.states.iter().find(|s| s.id == sel.state.id) { if s.health > 0.0 { sel.state = *s; } else { selected_agent = None; } } }
            }
            if let Some(fid) = followed_agent_id {
                if let Some(s) = data.sim.states.iter().find(|s| s.id == fid) {
                    let a = Person { state: *s, genetics: data.sim.genetics[s.genetics_index as usize] };
                    offset_x = loaded_config.world.map_width as f32 / 2.0 - a.state.x;
                    offset_y = loaded_config.world.map_height as f32 / 2.0 - a.state.y;
                } else { followed_agent_id = None; }
            }
        }

        clear_background(BLACK);
        set_camera(&Camera2D {
            target: vec2(loaded_config.world.map_width as f32 / 2.0 - offset_x, loaded_config.world.map_height as f32 / 2.0 - offset_y),
            zoom: vec2(zoom / (screen_width() / 2.0), -zoom / (screen_height() / 2.0)), ..Default::default()
        });
        let render_data = gpu.fetch_render(loaded_config.world.map_width, loaded_config.world.map_height);
        image.bytes.copy_from_slice(&render_data); texture.update(&image); draw_texture(&texture, 0.0, 0.0, WHITE);
        for a in &local_agent_coords {
            if a.health <= 0.0 { continue; }
            let r = 1.0 + (a.age / loaded_config.bio.puberty_age).min(1.0);
            let c = match current_visual_mode { VisualMode::Age => { let r = (a.age / loaded_config.bio.max_age).clamp(0.0, 1.0); Color::new(r, 1.0-r, 1.0, 1.0) }, VisualMode::Gender => if a.gender > 0.5 { Color::new(0.2, 0.6, 1.0, 1.0) } else { Color::new(1.0, 0.4, 0.7, 1.0) }, VisualMode::Tribes => Color::new(a.pheno_r*0.5+0.5, a.pheno_g*0.5+0.5, a.pheno_b*0.5+0.5, 1.0), _ => WHITE };
            draw_circle(a.x, a.y, r, c);
        }

        set_default_camera();
        let mut graph_consumed = false;
        ui::draw_metrics(metrics.0, metrics.1, metrics.2, metrics.3, metrics.4, loaded_config.world.tick_to_mins, get_fps(), get_fps() as f32, get_fps() as f32, current_visual_mode, show_inspector, show_generation_graph, show_config_panel, metrics.5, metrics.6);
        if show_visuals_panel { ui::draw_visuals_panel(mx, my, left_clicked, &mut current_visual_mode); }
        if show_generation_graph { graph_consumed = ui::draw_generation_graph(&metrics.7, loaded_config.world.tick_to_mins, mx, my, mouse_wheel_y, &mut gen_graph_scroll, &mut gen_graph_zoom, &mut gen_graph_avg_period); }
        if show_inspector { ui::draw_inspector(mx, my, left_clicked, mouse_wheel_y, &mut inspector_agents, &mut sort_col, &mut sort_desc, &mut inspector_scroll, &mut selected_agent, &mut followed_agent_id, &mut show_inspector, loaded_config.world.tick_to_mins); }
        if show_config_panel {
            let mut c = loaded_config; let last = loaded_config; let mut changed = false; let mut save_cfg = false;
            ui::draw_config_panel(mx, my, left_clicked, is_mouse_button_down(MouseButton::Left), get_frame_time(), &mut c, &last, &mut config_scroll, &mut config_search_query, &mut changed, &mut save_cfg, &mut active_config_button, &mut config_button_hold_time);
            if changed { let mut d = shared_data.lock().unwrap(); d.config = c; loaded_config = c; }
        }

        if !show_inspector && !show_config_panel && !graph_consumed {
            if mouse_wheel_y > 0.0 { zoom *= 1.1; } else if mouse_wheel_y < 0.0 { zoom *= 0.9; }
            if is_mouse_button_down(MouseButton::Left) && followed_agent_id.is_none() { offset_x += (mx - last_mouse.0) / zoom; offset_y -= (my - last_mouse.1) / zoom; }
        }
        last_mouse = (mx, my);

        if is_key_pressed(KeyCode::Escape) { break; }
        next_frame().await;
    }

    if let Ok(mut data) = shared_data.lock() {
        data.sim.env.map_cells = gpu.fetch_cells();
        let _ = pollster::block_on(save_everything(&data, "last_run", true, None));
    }
}
