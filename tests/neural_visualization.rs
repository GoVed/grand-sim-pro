/*
 * Grand Sim Pro: Architectural Neural Visualization (High-Fidelity Annotated Diagram)
 */

use world_sim::agent::{Person, INPUT_LABELS, OUTPUT_LABELS};
use world_sim::config::SimConfig;
use image::{RgbImage, Rgb};
use imageproc::drawing::{draw_line_segment_mut, draw_filled_circle_mut, draw_text_mut, draw_hollow_rect_mut, draw_filled_rect_mut, draw_antialiased_line_segment_mut};
use imageproc::rect::Rect;
use ab_glyph::{FontArc, PxScale};
use std::path::Path;

#[test]
fn generate_neural_structure_png() {
    let config = SimConfig::default();
    let _p = Person::new(0.0, 0.0, 0, &config);
    
    let output_path = "test_screenshots/neural_structure.png";
    if !Path::new("test_screenshots").exists() {
        let _ = std::fs::create_dir("test_screenshots");
    }

    let width: u32 = 2200;
    let height: u32 = 1300;
    let mut img = RgbImage::new(width, height);

    // Background
    for pixel in img.pixels_mut() { *pixel = Rgb([255, 255, 255]); }

    // Robust font loading from system
    let font_paths = [
        "/usr/share/fonts/google-droid-sans-fonts/DroidSans.ttf",
        "/usr/share/fonts/adwaita-sans-fonts/AdwaitaSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/google-carlito-fonts/Carlito-Regular.ttf"
    ];
    
    let mut font_opt = None;
    for path in font_paths {
        if let Ok(data) = std::fs::read(path) {
            if let Ok(f) = FontArc::try_from_vec(data) {
                font_opt = Some(f);
                break;
            }
        }
    }
    
    let font = font_opt.expect("CRITICAL ERROR: No valid .ttf font found on system! Text rendering will fail.");

    // Color Palette
    let title_color = Rgb([0, 0, 0]);
    let body_text = Rgb([60, 60, 60]);
    let block_blue = Rgb([235, 245, 255]);
    let block_orange = Rgb([255, 245, 235]);
    let block_green = Rgb([235, 255, 240]);
    let node_blue = Rgb([0, 102, 204]);
    let node_orange = Rgb([255, 153, 51]);
    let node_green = Rgb([76, 153, 0]);
    let synapse_color = Rgb([220, 220, 225]);

    // Helper: Draw Descriptive Header
    let draw_header = |img: &mut RgbImage, x: i32, y: i32, title: &str, subtitle: &str| {
        draw_text_mut(img, title_color, x, y, PxScale::from(32.0), &font, title);
        draw_text_mut(img, body_text, x, y + 35, PxScale::from(16.0), &font, subtitle);
    };

    // 1. GLOBAL/META INPUTS
    let meta_x = 100;
    let meta_y = 150;
    draw_filled_rect_mut(&mut img, Rect::at(meta_x, meta_y).of_size(320, 450), block_blue);
    draw_hollow_rect_mut(&mut img, Rect::at(meta_x, meta_y).of_size(320, 450), node_blue);
    draw_header(&mut img, meta_x, meta_y - 80, "1. PROPRIOCEPTIVE INPUTS", "Physical state & Internal memory");

    for i in 0..16usize {
        let iy = meta_y + 30 + (i as i32) * 26;
        draw_filled_circle_mut(&mut img, (meta_x + 30, iy), 8, node_blue);
        if i < INPUT_LABELS.len() {
            draw_text_mut(&mut img, body_text, meta_x + 50, iy - 8, PxScale::from(14.0), &font, INPUT_LABELS[i]);
        }
    }

    // 2. 5x5 SPATIAL VISION
    let vis_x = 100;
    let vis_y = 750;
    let grid_cell = 50;
    draw_filled_rect_mut(&mut img, Rect::at(vis_x - 10, vis_y - 10).of_size(270, 270), block_green);
    draw_hollow_rect_mut(&mut img, Rect::at(vis_x - 10, vis_y - 10).of_size(270, 270), node_green);
    draw_header(&mut img, vis_x, vis_y - 80, "2. 5x5 SPATIAL VISION", "24 distant cells (3 rows fwd, 1 back)");

    for gy in 0..5i32 {
        for gx in 0..5i32 {
            let cx = vis_x + gx * grid_cell;
            let cy = vis_y + gy * grid_cell;
            let is_agent = gx == 2 && gy == 3; // Center-ish relative pos
            let color = if is_agent { Rgb([255, 215, 0]) } else { Rgb([200, 230, 210]) };
            draw_filled_rect_mut(&mut img, Rect::at(cx, cy).of_size(grid_cell as u32 - 4, grid_cell as u32 - 4), color);
            if is_agent {
                draw_text_mut(&mut img, Rgb([0,0,0]), cx + 10, cy + 18, PxScale::from(12.0), &font, "SELF");
            }
        }
    }

    // 3. SPATIAL CNN
    let cnn_x = 550;
    let cnn_y = 650;
    draw_header(&mut img, cnn_x, cnn_y - 150, "3. SPATIAL CNN", "8 pseudo-random kernels (3x3)");
    for i in 0..8i32 {
        let offset = i * 20;
        let mx = cnn_x + offset;
        let my = cnn_y - offset;
        draw_filled_rect_mut(&mut img, Rect::at(mx, my).of_size(140, 140), Rgb([255 - (i as u8 * 10), 220 - (i as u8 * 10), 180]));
        draw_hollow_rect_mut(&mut img, Rect::at(mx, my).of_size(140, 140), node_orange);
        draw_text_mut(&mut img, node_orange, mx + 5, my + 20, PxScale::from(14.0), &font, &format!("Feature {}", i+1));
    }
    draw_antialiased_line_segment_mut(&mut img, (vis_x as i32 + 260, vis_y as i32 + 100), (cnn_x as i32, cnn_y as i32 + 70), Rgb([100, 100, 100]), imageproc::pixelops::interpolate);

    // 4. DEEP NEURAL NETWORK (Integration)
    let dnn_x = 1100;
    let dnn_y = 200;
    let dnn_w = 550u32;
    let dnn_h = 900u32;
    draw_hollow_rect_mut(&mut img, Rect::at(dnn_x - 50, dnn_y - 50).of_size(dnn_w + 100, dnn_h + 100), Rgb([180, 180, 180]));
    draw_header(&mut img, dnn_x + 100, dnn_y - 80, "4. INTEGRATION (MLP)", "Fully connected deep decision logic");

    let nodes_h = 12;
    let spacing = 75;

    for i in 0..nodes_h {
        let ny = dnn_y + i * spacing;
        draw_filled_circle_mut(&mut img, (dnn_x + 60, ny), 22, node_blue);
        draw_filled_circle_mut(&mut img, (dnn_x + 60, ny), 10, Rgb([255, 255, 255]));
        for j in 0..nodes_h {
            draw_line_segment_mut(&mut img, (dnn_x as f32 + 82.0, ny as f32), (dnn_x as f32 + 418.0, (dnn_y + j * spacing) as f32), synapse_color);
        }
    }
    draw_text_mut(&mut img, node_blue, dnn_x + 20, dnn_y + (dnn_h as i32) + 10, PxScale::from(20.0), &font, "Hidden Layer 1");

    for i in 0..nodes_h {
        let ny = dnn_y + i * spacing;
        draw_filled_circle_mut(&mut img, (dnn_x + 440, ny), 22, node_blue);
        draw_filled_circle_mut(&mut img, (dnn_x + 440, ny), 10, Rgb([255, 255, 255]));
    }
    draw_text_mut(&mut img, node_blue, dnn_x + 400, dnn_y + (dnn_h as i32) + 10, PxScale::from(20.0), &font, "Hidden Layer 2");

    draw_antialiased_line_segment_mut(&mut img, (cnn_x as i32 + 250, cnn_y as i32), (dnn_x as i32, dnn_y as i32 + 400), Rgb([100, 100, 100]), imageproc::pixelops::interpolate);
    draw_antialiased_line_segment_mut(&mut img, (meta_x as i32 + 320, meta_y as i32 + 200), (dnn_x as i32, dnn_y as i32 + 100), Rgb([100, 100, 100]), imageproc::pixelops::interpolate);

    // 5. BEHAVIORAL OUTPUTS
    let out_x = 1850;
    let out_y = 200;
    draw_header(&mut img, out_x, out_y - 80, "5. OUTPUTS", "Action Vectors");
    for i in 0..nodes_h {
        let oy = out_y + i * spacing;
        draw_filled_circle_mut(&mut img, (out_x + 30, oy), 24, node_orange);
        draw_filled_circle_mut(&mut img, (out_x + 30, oy), 14, Rgb([255, 255, 150]));
        if (i as usize) < OUTPUT_LABELS.len() {
            draw_text_mut(&mut img, title_color, out_x + 70, oy - 10, PxScale::from(18.0), &font, OUTPUT_LABELS[i as usize]);
        }
        draw_line_segment_mut(&mut img, (dnn_x as f32 + 462.0, oy as f32), (out_x as f32 + 10.0, oy as f32), Rgb([255, 200, 100]));
    }

    // 6. HEBBIAN PLASTICITY
    let plastic_y = 1100;
    draw_filled_rect_mut(&mut img, Rect::at(dnn_x as i32, plastic_y).of_size(550, 100), Rgb([255, 235, 255]));
    draw_hollow_rect_mut(&mut img, Rect::at(dnn_x as i32, plastic_y).of_size(550, 100), Rgb([200, 0, 200]));
    draw_text_mut(&mut img, Rgb([150, 0, 150]), dnn_x + 20, plastic_y + 35, PxScale::from(24.0), &font, "HEBBIAN LEARNING LOOP");
    draw_text_mut(&mut img, Rgb([100, 0, 100]), dnn_x + 20, plastic_y + 65, PxScale::from(16.0), &font, "Synaptic weight adjustments based on Dopamine/Interaction");

    img.save(output_path).unwrap();
    println!("High-fidelity architectural graph saved to {}", output_path);
}
