/*
 * Grand Sim Pro: A high-performance GPGPU evolutionary agent simulation.
 * Part of an independent research project into emergent biological complexity.
 *
 * Copyright (C) 2026 Ved Hirenkumar Suthar
 * Licensed under the GNU General Public License v3.0 or later.
 * * This software is provided "as is", without warranty of any kind.
 * See the LICENSE file in the project root for full license details.
 */

use world_sim::agent::Person;
use world_sim::simulation;
use world_sim::gpu_engine;
use world_sim::config;
use world_sim::shared;
use world_sim::sim_thread;
use world_sim::ui;
use world_sim::api;

use macroquad::prelude::*;
use simulation::SimulationManager;
use gpu_engine::GpuEngine;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use ::rand::Rng;
use shared::{SharedData, VisualMode, SortCol, AgentRenderData, FullState, save_everything, load_everything, ProgressReport};
use clap::{Parser, ValueEnum};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long)] headless: bool,
    #[arg(long, default_value_t = 3000)] api_port: u16,
    #[arg(long, default_value_t = 50)] speed: usize,
    #[arg(long, value_enum, default_value_t = StartMode::New)] mode: StartMode,
    #[arg(long)] save: Option<String>,
    #[arg(long)] max_ticks: Option<u64>,
    #[arg(long)] screenshot_test: bool,
    #[arg(long)] ui_iteration_test: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum StartMode { Founders, SaveFile, LastRun, New }

fn load_config() -> config::SimConfig {
    if let Ok(data) = std::fs::read_to_string("sim_config.json") { if let Ok(cfg) = serde_json::from_str(&data) { return cfg; } }
    let default_cfg = config::SimConfig::default();
    let _ = std::fs::write("sim_config.json", serde_json::to_string_pretty(&default_cfg).unwrap());
    default_cfg
}

fn window_conf() -> Conf {
    let cfg = load_config();
    Conf { window_title: "World Sim (Native)".to_owned(), window_width: cfg.world.display_width as i32, window_height: cfg.world.display_height as i32, fullscreen: true, ..Default::default() }
}

fn list_saves() -> Vec<String> {
    let mut saves = Vec::new();
    if let Ok(entries) = std::fs::read_dir("saves") {
        for entry in entries.flatten() {
            if entry.path().is_dir() { if let Some(name) = entry.file_name().to_str() { saves.push(name.to_string()); } }
        }
    }
    saves.sort(); saves.reverse(); saves
}

fn draw_progress_bar(report: &ProgressReport) {
    let w = 400.0; let h = 30.0; let x = screen_width() / 2.0 - w / 2.0; let y = screen_height() / 2.0 + 50.0;
    draw_rectangle(x - 5.0, y - 5.0, w + 10.0, h + 10.0, DARKGRAY);
    draw_rectangle(x, y, w, h, BLACK);
    draw_rectangle(x, y, w * report.progress, h, Color::new(0.0, 0.8, 0.6, 1.0));
    let full_text = format!("{}: {:.0}%", report.message, report.progress * 100.0);
    let dims = measure_text(&full_text, None, 20, 1.0);
    draw_text(&full_text, screen_width() / 2.0 - dims.width / 2.0, y - 10.0, 20.0, WHITE);
}

#[macroquad::main(window_conf)]
async fn main() {
    let args = Cli::parse();
    let mut loaded_config = if args.ui_iteration_test {
        let mut c = load_config();
        c.world.map_width = 100; c.world.map_height = 100;
        c.sim.agent_count = 100;
        c
    } else { load_config() };
    
    let mut start_choice = if args.screenshot_test || args.ui_iteration_test { Some(StartChoice::New) } else { None };
    let mut selected_save_path = None;

    if args.headless {
        start_choice = Some(match args.mode { StartMode::Founders => StartChoice::Founders, StartMode::SaveFile => StartChoice::SaveFile, StartMode::LastRun => StartChoice::LastRun, StartMode::New => StartChoice::New });
        if let Some(s) = args.save { selected_save_path = Some(std::path::PathBuf::from(s)); }
    } else if start_choice.is_none() {
        #[derive(Clone, Copy, PartialEq)] enum MenuState { Main, SelectSave, Loading }
        let mut menu_state = MenuState::Main;
        while start_choice.is_none() {
            clear_background(DARKGRAY);
            match menu_state {
                MenuState::Main => {
                    draw_text("GRAND SIM PRO", screen_width()/2.0 - 180.0, 100.0, 60.0, Color::new(0.0, 1.0, 0.8, 1.0));
                    let options = [(StartChoice::Founders, "1. LOAD FROM FOUNDERS"), (StartChoice::SaveFile, "2. LOAD FROM SAVE FILE"), (StartChoice::LastRun, "3. CONTINUE LAST RUN"), (StartChoice::New, "4. START NEW SIMULATION")];
                    let mut y = 200.0;
                    for (choice, label) in options.iter() {
                        let rect = Rect::new(screen_width()/2.0 - 150.0, y, 300.0, 40.0);
                        let hover = rect.contains(vec2(mouse_position().0, mouse_position().1));
                        draw_rectangle(rect.x, rect.y, rect.w, rect.h, if hover { Color::new(0.0, 0.8, 0.7, 1.0) } else { Color::new(0.0, 0.4, 0.3, 1.0) });
                        draw_text(label, rect.x + 20.0, rect.y + 28.0, 20.0, WHITE);
                        if is_mouse_button_pressed(MouseButton::Left) && hover {
                            if *choice == StartChoice::SaveFile { menu_state = MenuState::SelectSave; } else { menu_state = MenuState::Loading; start_choice = Some(*choice); }
                        }
                        y += 60.0;
                    }
                }
                MenuState::SelectSave => {
                    draw_text("SELECT SAVE FILE", screen_width()/2.0 - 150.0, 80.0, 40.0, Color::new(0.0, 1.0, 1.0, 1.0));
                    let saves = list_saves(); let mut y = 150.0;
                    for save_name in saves {
                        let rect = Rect::new(screen_width()/2.0 - 200.0, y, 400.0, 35.0);
                        let hover = rect.contains(vec2(mouse_position().0, mouse_position().1));
                        draw_rectangle(rect.x, rect.y, rect.w, rect.h, if hover { Color::new(0.2, 0.2, 0.2, 1.0) } else { BLACK });
                        draw_text(&save_name, rect.x + 10.0, rect.y + 25.0, 18.0, if hover { Color::new(0.0, 1.0, 1.0, 1.0) } else { WHITE });
                        if is_mouse_button_pressed(MouseButton::Left) && hover { selected_save_path = Some(std::path::PathBuf::from("saves").join(save_name)); start_choice = Some(StartChoice::SaveFile); }
                        y += 40.0;
                    }
                    if is_key_pressed(KeyCode::Escape) { menu_state = MenuState::Main; }
                }
                _ => {}
            }
            next_frame().await;
        }
    }

    let (final_tx, final_rx) = std::sync::mpsc::channel::<SharedData>();
    let (prog_tx, prog_rx) = std::sync::mpsc::channel::<ProgressReport>();
    let choice = start_choice.expect("No start choice made");
    let save_path = selected_save_path.clone();
    let config = loaded_config.clone();

    std::thread::spawn(move || {
        let mut initial_full_state: Option<FullState> = None;
        let mut founders = Vec::new();
        match choice {
            StartChoice::Founders => { let _ = std::fs::remove_file("telemetry.csv"); founders = crate::simulation::load_founders(&config); }
            StartChoice::SaveFile => { if let Some(path) = save_path { initial_full_state = load_everything(&path, prog_tx.clone()); } }
            StartChoice::LastRun => { initial_full_state = load_everything(std::path::Path::new("last_run"), prog_tx.clone()); }
            StartChoice::New => { let _ = std::fs::remove_file("telemetry.csv"); }
        }
        let shared = if let Some(fs) = initial_full_state { fs.shared } else {
            let sim = SimulationManager::new(config.world.map_width, config.world.map_height, ::rand::thread_rng().r#gen(), config.sim.agent_count, &config, founders);
            SharedData {
                sim, config: config.clone(), last_saved_config: config.clone(), is_paused: false, restart_message_active: false, ticks_per_loop: args.speed, total_ticks: 0, cumulative_ticks: 0, last_telemetry_tick: 0, cumulative_births: config.sim.agent_count as u64, cumulative_deaths: 0, last_compute_time_micros: 0, ticks_per_second: 0.0, generation_survival_times: Vec::new(),
            }
        };
        let _ = final_tx.send(shared);
    });

    let mut current_report = ProgressReport { message: "Initializing...".to_string(), progress: 0.0 };
    let initial_shared = loop {
        if !args.headless {
            clear_background(DARKGRAY);
            while let Ok(r) = prog_rx.try_recv() { current_report = r; }
            draw_progress_bar(&current_report);
        }
        if let Ok(s) = final_rx.try_recv() { break s; }
        if !args.headless { next_frame().await; } else { std::thread::sleep(Duration::from_millis(10)); }
    };

    let shared_data = Arc::new(Mutex::new(initial_shared));
    let sim_thread_data = shared_data.clone();
    let (initial_states, initial_genetics, height_map, init_cells) = { let lock = sim_thread_data.lock().unwrap(); (lock.sim.states.clone(), lock.sim.genetics.clone(), lock.sim.env.height_map.clone(), lock.sim.env.map_cells.clone()) };
    let gpu = Arc::new(GpuEngine::new(&initial_states, &initial_genetics, &height_map, &init_cells, &loaded_config));
    
    let api_shared = shared_data.clone(); let api_gpu = gpu.clone(); let api_port = args.api_port;
    std::thread::spawn(move || { let rt = tokio::runtime::Runtime::new().unwrap(); rt.block_on(api::start_api(api_shared, api_gpu, api_port)); });

    sim_thread::spawn(sim_thread_data.clone(), gpu.clone(), args.headless);

    if args.headless {
        println!("Headless mode active. Running at maximum throughput.");
        loop {
            let ticks = { let d = shared_data.lock().unwrap(); d.cumulative_ticks };
            if let Some(max) = args.max_ticks { if ticks >= max { break; } }
            if is_key_pressed(KeyCode::Escape) { break; }
            std::thread::sleep(Duration::from_millis(100));
        }
    } else {
        let mut image = Image::gen_image_color(loaded_config.world.map_width as u16, loaded_config.world.map_height as u16, BLANK);
        let texture = Texture2D::from_image(&image);
        let mut zoom = 1.0f32; let mut offset_x = 0.0f32; let mut offset_y = 0.0f32;
        let mut last_mouse = mouse_position();
        let mut local_agent_coords: Vec<AgentRenderData> = Vec::with_capacity(loaded_config.sim.agent_count as usize);
        let mut current_visual_mode = VisualMode::Default;
        let mut show_visuals_panel = false; let mut show_inspector = false; let mut show_generation_graph = false;
        let mut sort_col = SortCol::Food; let mut sort_desc = true; let mut inspector_scroll: usize = 0;
        let mut selected_agent: Option<Person> = None;
        let mut inspector_agents: Vec<(usize, Person)> = Vec::new();
        let mut show_config_panel = false; let mut config_scroll = 0.0; let mut config_search_query = String::new();
        let mut active_config_button: Option<String> = None;
        let mut followed_agent_id: Option<u32> = None; let mut followed_agent: Option<Person> = None;
        let mut is_live_mode = false; let mut live_ticks_timer = 0;
        let mut pre_live_speed = loaded_config.sim.agent_count as usize / 10;
        let mut pre_live_paused = false;
        let mut last_live_mode = false;

        let mut is_quitting = false;
        let mut saving_ui_report: Option<ProgressReport> = None;
        let (save_prog_tx, save_prog_rx) = std::sync::mpsc::channel::<ProgressReport>();
        let mut metrics = (0, 0.0f32, 0.0f32, 0usize, 0u64, false, false, Vec::new());
        let mut gen_graph_scroll = 0.0f32; let mut gen_graph_zoom = 1.0f32; let mut gen_graph_avg_period = 25usize;
        
        let mut iter_test_phase = 0; let mut iter_test_timer = 0;
        let modes = [VisualMode::Default, VisualMode::Resources, VisualMode::Age, VisualMode::Gender, VisualMode::Pregnancy, VisualMode::MarketWealth, VisualMode::MarketFood, VisualMode::AskPrice, VisualMode::BidPrice, VisualMode::Infrastructure, VisualMode::Temperature, VisualMode::DayNight, VisualMode::Tribes, VisualMode::Water];

        loop {
            if saving_ui_report.is_some() {
                clear_background(BLACK);
                while let Ok(r) = save_prog_rx.try_recv() { if r.progress >= 1.0 { saving_ui_report = None; break; } saving_ui_report = Some(r); }
                if let Some(ref r) = saving_ui_report { draw_progress_bar(r); next_frame().await; continue; }
            }
            
            // Automation for UI Iteration Test
            if args.ui_iteration_test {
                iter_test_timer += 1;
                match iter_test_phase {
                    0 => if iter_test_timer > 120 { // Capture Main View
                        let _ = std::fs::create_dir_all("test_screenshots/iteration");
                        get_screen_data().export_png("test_screenshots/iteration/01_main_view.png");
                        // Find agent to follow
                        if let Ok(data) = shared_data.lock() { if let Some(s) = data.sim.states.iter().find(|s| s.health > 0.0) { followed_agent_id = Some(s.id); } }
                        iter_test_phase = 1; iter_test_timer = 0;
                    }
                    1 => if iter_test_timer > 60 { // Capture Follow View
                        get_screen_data().export_png("test_screenshots/iteration/02_follow_view.png");
                        is_live_mode = true; iter_test_phase = 2; iter_test_timer = 0;
                    }
                    2 => if iter_test_timer > 60 { // Capture Live POV
                        get_screen_data().export_png("test_screenshots/iteration/03_live_pov.png");
                        show_inspector = true; iter_test_phase = 3; iter_test_timer = 0;
                    }
                    3 => if iter_test_timer > 60 { // Capture Inspector
                        get_screen_data().export_png("test_screenshots/iteration/04_inspector.png");
                        show_config_panel = true; iter_test_phase = 4; iter_test_timer = 0;
                    }
                    4 => if iter_test_timer > 60 { // Finish
                        get_screen_data().export_png("test_screenshots/iteration/05_final_config.png");
                        println!("UI Iteration Test Complete.");
                        break;
                    }
                    _ => {}
                }
            }

            if args.screenshot_test { 
                current_visual_mode = modes[(iter_test_timer / 60) as usize % modes.len()]; 
                if iter_test_timer > 60 * modes.len() as i32 { break; }
                iter_test_timer += 1;
            }
            if is_key_pressed(KeyCode::C) { show_config_panel = !show_config_panel; while get_char_pressed().is_some() {} }
            if is_key_pressed(KeyCode::L) && followed_agent_id.is_some() { 
                if !is_live_mode {
                    // Entering live mode
                    let d = shared_data.lock().unwrap();
                    pre_live_speed = d.ticks_per_loop;
                    pre_live_paused = d.is_paused;
                    is_live_mode = true;
                } else {
                    // Exiting live mode
                    let mut d = shared_data.lock().unwrap();
                    d.ticks_per_loop = pre_live_speed;
                    d.is_paused = pre_live_paused;
                    is_live_mode = false;
                }
            }
            if is_key_pressed(KeyCode::Q) {
                if is_quitting { break; } // Second 'q' to quit
                is_quitting = true;
            }
            if !show_config_panel && !is_quitting {
                if is_key_pressed(KeyCode::Space) { let mut d = shared_data.lock().unwrap(); d.is_paused = !d.is_paused; }
                if is_key_pressed(KeyCode::S) { 
                    let data = shared_data.lock().unwrap(); let _ = std::fs::create_dir_all("saved_agents_weights");
                    let mut living: Vec<_> = data.sim.states.iter().enumerate().filter(|(_, s)| s.health > 0.0).collect();
                    living.sort_by(|(_, a), (_, b)| (b.wealth + b.food).partial_cmp(&(a.wealth + a.food)).unwrap());
                    for i in 0..living.len().min(data.config.sim.founder_count as usize) {
                        let (_, s) = living[i]; let p = Person { state: *s, genetics: data.sim.genetics[s.genetics_index as usize] };
                        let _ = std::fs::write(format!("saved_agents_weights/agent_{}.json", i), serde_json::to_string_pretty(&p.extract_weights()).unwrap());
                    }
                }
                if is_key_pressed(KeyCode::F) {
                    let mut data = shared_data.lock().unwrap(); data.sim.env.map_cells = gpu.fetch_cells();
                    let save_name = format!("save_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs());
                    let shared_clone = data.clone(); let tx = save_prog_tx.clone();
                    saving_ui_report = Some(ProgressReport { message: "Starting Save...".to_string(), progress: 0.0 });
                    std::thread::spawn(move || { pollster::block_on(save_everything(&shared_clone, &save_name, false, Some(tx))); });
                }
                if is_key_pressed(KeyCode::R) { show_visuals_panel = !show_visuals_panel; }
                if is_key_pressed(KeyCode::G) { show_generation_graph = !show_generation_graph; }
                if is_key_pressed(KeyCode::Up) { let mut d = shared_data.lock().unwrap(); d.ticks_per_loop = (d.ticks_per_loop as f32 * 1.2).max(d.ticks_per_loop as f32 + 1.0) as usize; }
                if is_key_pressed(KeyCode::Down) { let mut d = shared_data.lock().unwrap(); d.ticks_per_loop = (d.ticks_per_loop as f32 / 1.2).max(1.0) as usize; }
            }
            if is_key_pressed(KeyCode::Tab) { show_inspector = !show_inspector; selected_agent = None; }

            // Detect state change to restore ticks when exiting live mode via buttons
            if is_live_mode != last_live_mode {
                if is_live_mode {
                    // Just entered
                    let d = shared_data.lock().unwrap();
                    pre_live_speed = d.ticks_per_loop;
                    pre_live_paused = d.is_paused;
                } else {
                    // Just exited
                    let mut d = shared_data.lock().unwrap();
                    d.ticks_per_loop = pre_live_speed;
                    d.is_paused = pre_live_paused;
                }
                last_live_mode = is_live_mode;
            }

            if is_live_mode {
                live_ticks_timer += 1;
                let mut d = shared_data.lock().unwrap();
                if live_ticks_timer >= 7 { d.ticks_per_loop = 1; d.is_paused = false; live_ticks_timer = 0; } else { d.is_paused = true; }
                if followed_agent_id.is_none() { is_live_mode = false; }
            }
            let (_mw_x, mw_y) = mouse_wheel(); let (mx, my) = mouse_position(); let left_clicked = is_mouse_button_pressed(MouseButton::Left);
            if let Ok(mut data) = shared_data.try_lock() {
                data.config.sim.visual_mode = match current_visual_mode { VisualMode::Default => 0, VisualMode::Resources => 1, VisualMode::Age => 2, VisualMode::Gender => 3, VisualMode::Pregnancy => 4, VisualMode::MarketWealth => 5, VisualMode::MarketFood => 6, VisualMode::AskPrice => 7, VisualMode::BidPrice => 8, VisualMode::Infrastructure => 9, VisualMode::Temperature => 10, VisualMode::DayNight => 11, VisualMode::Tribes => 12, VisualMode::Water => 13 };
                metrics = (data.sim.states.iter().filter(|s| s.health > 0.0).count(), data.last_compute_time_micros as f32 / 1000.0, data.ticks_per_second, data.ticks_per_loop, data.total_ticks, data.is_paused, data.restart_message_active, data.generation_survival_times.clone());
                local_agent_coords.clear();
                local_agent_coords.extend(data.sim.states.iter().map(|s| AgentRenderData { x: s.x, y: s.y, health: s.health, food: s.food, age: s.age, wealth: s.wealth, gender: s.gender, is_pregnant: s.is_pregnant, pheno_r: s.pheno_r, pheno_g: s.pheno_g, pheno_b: s.pheno_b }));
                if show_inspector {
                    if inspector_agents.len() != metrics.0 {
                        inspector_agents.clear();
                        for (i, s) in data.sim.states.iter().enumerate() { if s.health > 0.0 { inspector_agents.push((i, Person { state: *s, genetics: data.sim.genetics[s.genetics_index as usize] })); } }
                        ui::apply_sort(&mut inspector_agents, sort_col, sort_desc);
                    } else { for i in 0..inspector_agents.len() { let s = &data.sim.states[inspector_agents[i].0]; inspector_agents[i].1 = Person { state: *s, genetics: data.sim.genetics[s.genetics_index as usize] }; } }
                }
                if let Some(fid) = followed_agent_id {
                    if let Some(s) = data.sim.states.iter().find(|s| s.id == fid) {
                        let a = Person { state: *s, genetics: data.sim.genetics[s.genetics_index as usize] };
                        offset_x = loaded_config.world.map_width as f32 / 2.0 - a.state.x; offset_y = loaded_config.world.map_height as f32 / 2.0 - a.state.y;
                        followed_agent = Some(a);
                    } else { followed_agent_id = None; followed_agent = None; }
                } else { followed_agent = None; }
            }
            clear_background(BLACK);
            let cam_rect = if is_live_mode { Rect::new(screen_width() - 310.0, 10.0, 300.0, 300.0) } else { Rect::new(0.0, 0.0, screen_width(), screen_height()) };
            let (c_target, c_zoom) = if is_live_mode {
                if let Some(ref a) = followed_agent { (vec2(a.state.x, a.state.y), vec2(1.0 / 25.0, -1.0 / 25.0)) }
                else { (vec2(loaded_config.world.map_width as f32 / 2.0 - offset_x, loaded_config.world.map_height as f32 / 2.0 - offset_y), vec2(zoom / (screen_width() / 2.0), -zoom / (screen_height() / 2.0))) }
            } else { (vec2(loaded_config.world.map_width as f32 / 2.0 - offset_x, loaded_config.world.map_height as f32 / 2.0 - offset_y), vec2(zoom / (screen_width() / 2.0), -zoom / (screen_height() / 2.0))) };
            
            set_camera(&Camera2D { render_target: None, viewport: Some((cam_rect.x as i32, cam_rect.y as i32, cam_rect.w as i32, cam_rect.h as i32)), target: c_target, zoom: c_zoom, ..Default::default() });
            let render_data = gpu.fetch_render(loaded_config.world.map_width, loaded_config.world.map_height);
            image.bytes.copy_from_slice(&render_data); texture.update(&image); draw_texture(&texture, 0.0, 0.0, WHITE);
            for (i, a) in local_agent_coords.iter().enumerate() {
                if a.health <= 0.0 { continue; }
                let r = 1.0 + (a.age / loaded_config.bio.puberty_age).min(1.0);
                let c = match current_visual_mode { VisualMode::Age => { let r = (a.age / loaded_config.bio.max_age).clamp(0.0, 1.0); Color::new(r, 1.0-r, 1.0, 1.0) }, VisualMode::Gender => if a.gender > 0.5 { Color::new(0.2, 0.6, 1.0, 1.0) } else { Color::new(1.0, 0.4, 0.7, 1.0) }, VisualMode::Tribes => Color::new(a.pheno_r*0.5+0.5, a.pheno_g*0.5+0.5, a.pheno_b*0.5+0.5, 1.0), _ => WHITE };
                draw_circle(a.x, a.y, r, c);
                
                // Highlight followed agent
                if let Some(fid) = followed_agent_id {
                    if let Ok(data) = shared_data.try_lock() {
                        if data.sim.states[i].id == fid {
                            draw_circle_lines(a.x, a.y, r + 2.0, 1.0, YELLOW);
                        }
                    }
                }
            }
            set_default_camera();
            if is_live_mode { if let Some(ref a) = followed_agent { if let Ok(data) = shared_data.lock() { ui::draw_live_pov(a, &data.sim, &loaded_config); } } }
            let mut graph_consumed = false;
            ui::draw_metrics(metrics.0, metrics.1, metrics.2, metrics.3, metrics.4, loaded_config.world.tick_to_mins, get_fps(), get_fps() as f32, get_fps() as f32, current_visual_mode, show_inspector, show_generation_graph, show_config_panel, metrics.5, metrics.6);
            if show_visuals_panel { ui::draw_visuals_panel(mx, my, left_clicked, &mut current_visual_mode); }
            if show_generation_graph { graph_consumed = ui::draw_generation_graph(&metrics.7, loaded_config.world.tick_to_mins, mx, my, mw_y, &mut gen_graph_scroll, &mut gen_graph_zoom, &mut gen_graph_avg_period); }
            if show_config_panel {
                let mut c = loaded_config; let mut changed = false; let mut save_cfg = false;
                ui::draw_config_panel(mx, my, left_clicked, is_mouse_button_down(MouseButton::Left), get_frame_time(), &mut c, &loaded_config, &mut config_scroll, &mut config_search_query, &mut changed, &mut save_cfg, &mut active_config_button);
                if changed { let mut d = shared_data.lock().unwrap(); d.config = c; loaded_config = c; }
            }
            if show_inspector { 
                ui::draw_inspector(mx, my, left_clicked, mw_y, &mut inspector_agents, &mut sort_col, &mut sort_desc, &mut inspector_scroll, &mut selected_agent, &mut followed_agent_id, &mut show_inspector, loaded_config.world.tick_to_mins, &mut is_live_mode); 
            }
            if let Some(a) = &followed_agent { 
                if (!show_inspector || is_live_mode) && !is_live_mode { // Draw tracker if inspector is closed, but NOT in live mode to avoid overlap
                    ui::draw_tracker(mx, my, left_clicked, a, &mut followed_agent_id, &mut show_inspector, loaded_config.world.tick_to_mins, &mut is_live_mode); 
                }
            }
            if !show_inspector && !show_config_panel && !graph_consumed {
                if mw_y > 0.0 { zoom *= 1.1; } else if mw_y < 0.0 { zoom *= 0.9; }
                if is_mouse_button_down(MouseButton::Left) && followed_agent_id.is_none() { offset_x += (mx - last_mouse.0) / zoom; offset_y -= (my - last_mouse.1) / zoom; }
            }

            if is_quitting {
                let mut should_exit = false;
                ui::draw_quit_confirmation(mx, my, left_clicked, &mut is_quitting, &mut should_exit);
                if should_exit { break; }
            }

            last_mouse = (mx, my); if is_key_pressed(KeyCode::Escape) { break; } next_frame().await;
        }
    }
    if let Ok(mut data) = shared_data.lock() { data.sim.env.map_cells = gpu.fetch_cells(); let _ = pollster::block_on(save_everything(&data, "last_run", true, None)); }
}

#[derive(Clone, Copy, PartialEq)] enum StartChoice { Founders, SaveFile, LastRun, New }
