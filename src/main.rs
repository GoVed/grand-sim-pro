mod agent;
mod environment;
mod simulation;
mod gpu_engine;

use macroquad::prelude::*;
use simulation::SimulationManager;
use gpu_engine::GpuEngine;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

fn window_conf() -> Conf {
    Conf {
        window_title: "World Sim (Native)".to_owned(),
        window_width: 800,
        window_height: 600,
        ..Default::default()
    }
}

struct SharedData {
    sim: SimulationManager,
    is_paused: bool,
    ticks_per_loop: usize,
    total_ticks: u64,
    last_compute_time_ms: u128,
}

#[macroquad::main(window_conf)]
async fn main() {
    let width = 800;
    let height = 600;
    
    let shared_data = Arc::new(Mutex::new(SharedData {
        sim: SimulationManager::new(width, height, 1234, 4000),
        is_paused: false,
        ticks_per_loop: 2000, // Safe to massively increase default speed now!
        total_ticks: 0,
        last_compute_time_ms: 0,
    }));

    let sim_thread_data = shared_data.clone();
    
    let initial_agents = sim_thread_data.lock().unwrap().sim.agents.clone();
    let height_map = sim_thread_data.lock().unwrap().sim.env.height_map.clone();
    let gpu = Arc::new(GpuEngine::new(&initial_agents, &height_map));

    // Spawn background simulation thread
    thread::spawn(move || loop {
        let mut ran_ticks = false;
        {
            let mut data = sim_thread_data.lock().unwrap();
            if !data.is_paused {
                let start = Instant::now();
                
                // Dispatch heavy work entirely to the AMD GPU
                gpu.compute_ticks(data.ticks_per_loop);
                
                // Download updated positions precisely when needed
                data.sim.agents = gpu.fetch_agents();
                
                data.total_ticks += data.ticks_per_loop as u64;
                data.last_compute_time_ms = start.elapsed().as_millis();
                ran_ticks = true;
            }
        }
        
        if !ran_ticks {
            thread::sleep(Duration::from_millis(16)); // Prevent CPU hogging when paused
        } else {
            thread::sleep(Duration::from_millis(1)); // Force a context switch so the UI can lock the Mutex
        }
    });
    
    let mut image = Image::gen_image_color(width as u16, height as u16, BLANK);
    let texture = Texture2D::from_image(&image);
    
    let mut zoom = 1.0f32;
    let mut offset_x = 0.0f32;
    let mut offset_y = 0.0f32;
    let mut last_mouse = mouse_position();

    let mut local_agent_coords = Vec::with_capacity(4000);
    
    let mut paused = false;
    let mut speed = 20;
    let mut ticks = 0;
    let mut compute_time = 0;
    let mut pop_count = 0;

    let mut pending_pause_toggle = false;
    let mut pending_speed_change: i32 = 0;

    let mut frame_times: Vec<f32> = Vec::with_capacity(300);

    loop {
        // Register UI inputs instantly
        if is_key_pressed(KeyCode::Space) { pending_pause_toggle = !pending_pause_toggle; }
        if is_key_pressed(KeyCode::Up) { pending_speed_change += 1; }
        if is_key_pressed(KeyCode::Down) { pending_speed_change -= 1; }

        let (_, mouse_wheel_y) = mouse_wheel();
        if mouse_wheel_y > 0.0 { zoom *= 1.1; }
        if mouse_wheel_y < 0.0 { zoom *= 0.9; }

        let (mx, my) = mouse_position();
        if is_mouse_button_down(MouseButton::Left) {
            offset_x += (mx - last_mouse.0) / zoom;
            offset_y -= (my - last_mouse.1) / zoom;
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
            pop_count = data.sim.agents.len();
            
            // Fast copy for rendering to avoid holding lock during draw
            image.bytes.copy_from_slice(&data.sim.env.map_data);
            local_agent_coords.clear();
            local_agent_coords.extend(data.sim.agents.iter().map(|a| (a.x, a.y)));
        }

        // --- Rendering ---
        clear_background(BLACK);

        let cam = Camera2D {
            target: vec2(width as f32 / 2.0 - offset_x, height as f32 / 2.0 - offset_y),
            // -zoom for the Y axis is used here to match a 2D top-left Origin standard
            zoom: vec2(zoom / (width as f32 / 2.0), -zoom / (height as f32 / 2.0)), 
            ..Default::default()
        };
        set_camera(&cam);

        // 1. Efficient Zero-Copy Map Rendering
        texture.update(&image);
        draw_texture(&texture, 0.0, 0.0, WHITE);

        // 2. High-performance Agent Rendering
        for (ax, ay) in &local_agent_coords {
            draw_circle(*ax, *ay, 1.5, WHITE);
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
        
        draw_rectangle(10.0, 10.0, 240.0, 180.0, Color::new(0.0, 0.04, 0.04, 0.9));
        draw_rectangle_lines(10.0, 10.0, 240.0, 180.0, 1.0, Color::new(0.0, 1.0, 0.8, 1.0));
        
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
        draw_text(&format!("Sim Time (Ticks): {}", ticks), 20.0, y, 16.0, WHITE);
        y += dy;
        draw_text(&format!("FPS: {}", fps), 20.0, y, 16.0, WHITE);
        y += dy;
        draw_text(&format!("Avg FPS: {:.1}", avg_fps), 20.0, y, 16.0, WHITE);
        y += dy;
        draw_text(&format!("1% Low: {:.1}", low_1_fps), 20.0, y, 16.0, WHITE);

        if paused {
            draw_text("PAUSED", width as f32 / 2.0 - 50.0, 30.0, 30.0, RED);
        }

        next_frame().await;
    }
}