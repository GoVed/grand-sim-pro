/*
 * Grand Sim Pro: Architectural Neural Visualization (High-Fidelity)
 */

use world_sim::agent::{Person, INPUT_LABELS, OUTPUT_LABELS};
use world_sim::config::SimConfig;
use image::{RgbImage, Rgb};
use imageproc::drawing::{draw_line_segment_mut, draw_filled_circle_mut, draw_text_mut, draw_hollow_rect_mut, draw_filled_rect_mut};
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

    let width: u32 = 1800;
    let height: u32 = 1200;
    let mut img = RgbImage::new(width, height);

    // Modern Dark Background
    for pixel in img.pixels_mut() { *pixel = Rgb([15, 18, 22]); }

    let font_data = std::fs::read("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf").ok()
        .or_else(|| std::fs::read("/usr/share/fonts/TTF/DejaVuSans.ttf").ok());
    let font = font_data.as_ref().and_then(|data| FontArc::try_from_vec(data.clone()).ok());

    // --- SECTION 1: GLOBAL/META INPUTS (Top Left) ---
    let meta_x: i32 = 100;
    let meta_y: i32 = 100;
    let meta_w: u32 = 280;
    let meta_h: u32 = 350;
    draw_filled_rect_mut(&mut img, Rect::at(meta_x, meta_y).of_size(meta_w, meta_h), Rgb([30, 35, 45]));
    draw_hollow_rect_mut(&mut img, Rect::at(meta_x, meta_y).of_size(meta_w, meta_h), Rgb([100, 100, 120]));
    if let Some(ref f) = font {
        draw_text_mut(&mut img, Rgb([200, 200, 255]), meta_x + 20, meta_y - 35, PxScale::from(24.0), f, "GLOBAL/META INPUTS");
    }

    for i in 0..12usize {
        let iy = meta_y + 30 + (i as i32) * 25;
        draw_filled_circle_mut(&mut img, (meta_x + 30, iy), 8, Rgb([0, 180, 255]));
        if let Some(ref f) = font {
            draw_text_mut(&mut img, Rgb([180, 180, 180]), meta_x + 50, iy - 7, PxScale::from(14.0), f, INPUT_LABELS[i]);
        }
    }

    // --- SECTION 2: 5x5 VISION GRID (Bottom Left) ---
    let vis_x: i32 = 100;
    let vis_y: i32 = 550;
    let grid_cell: i32 = 45;
    if let Some(ref f) = font {
        draw_text_mut(&mut img, Rgb([0, 255, 150]), vis_x, vis_y - 35, PxScale::from(24.0), f, "5x5 SPATIAL VISION");
    }
    for gy in 0..5i32 {
        for gx in 0..5i32 {
            let cx = vis_x + gx * grid_cell;
            let cy = vis_y + gy * grid_cell;
            let color = if gx == 2 && gy == 3 { Rgb([255, 255, 0]) } else { Rgb([40, 60, 50]) };
            draw_filled_rect_mut(&mut img, Rect::at(cx, cy).of_size((grid_cell - 4) as u32, (grid_cell - 4) as u32), color);
            draw_hollow_rect_mut(&mut img, Rect::at(cx, cy).of_size((grid_cell - 4) as u32, (grid_cell - 4) as u32), Rgb([100, 150, 120]));
        }
    }

    // --- SECTION 3: SPATIAL CNN (Middle Left) ---
    let cnn_x: i32 = 550;
    let cnn_y: i32 = 500;
    if let Some(ref f) = font {
        draw_text_mut(&mut img, Rgb([255, 100, 0]), cnn_x, cnn_y - 120, PxScale::from(24.0), f, "SPATIAL CNN (8 Kernels)");
    }
    for i in 0..8i32 {
        let offset = i * 15;
        let mx = cnn_x + offset;
        let my = cnn_y - offset;
        draw_filled_rect_mut(&mut img, Rect::at(mx, my).of_size(120, 120), Rgb([200 - (i as u8 * 20), 80, 0]));
        draw_hollow_rect_mut(&mut img, Rect::at(mx, my).of_size(120, 120), Rgb([255, 255, 255]));
        if i == 0 {
            draw_line_segment_mut(&mut img, (vis_x as f32 + 100.0, vis_y as f32 + 100.0), (mx as f32, my as f32 + 60.0), Rgb([100, 100, 100]));
        }
    }

    // --- SECTION 4: DEEP NEURAL NETWORK (Center) ---
    let dnn_x: i32 = 900;
    let dnn_y: i32 = 150;
    let dnn_w: u32 = 450;
    let dnn_h: u32 = 900;
    draw_hollow_rect_mut(&mut img, Rect::at(dnn_x - 50, dnn_y - 50).of_size(dnn_w + 100, dnn_h + 100), Rgb([60, 60, 70]));
    if let Some(ref f) = font {
        draw_text_mut(&mut img, Rgb([255, 255, 255]), dnn_x + 100, dnn_y - 80, PxScale::from(28.0), f, "DEEP NEURAL NETWORK");
    }

    let nodes_h1 = 12;
    let nodes_h2 = 12;
    let node_spacing = 70;

    for i in 0..nodes_h1 {
        let ny = dnn_y + i * node_spacing;
        draw_filled_circle_mut(&mut img, (dnn_x + 50, ny), 18, Rgb([0, 120, 255]));
        draw_filled_circle_mut(&mut img, (dnn_x + 50, ny), 12, Rgb([100, 200, 255]));
        for j in 0..nodes_h2 {
            let n2y = dnn_y + j * node_spacing;
            draw_line_segment_mut(&mut img, (dnn_x as f32 + 68.0, ny as f32), (dnn_x as f32 + 332.0, n2y as f32), Rgb([40, 45, 60]));
        }
    }
    if let Some(ref f) = font {
        draw_text_mut(&mut img, Rgb([150, 150, 255]), dnn_x + 20, dnn_y + (dnn_h as i32) + 20, PxScale::from(18.0), f, "Hidden Layer 1");
        draw_text_mut(&mut img, Rgb([150, 150, 255]), dnn_x + 320, dnn_y + (dnn_h as i32) + 20, PxScale::from(18.0), f, "Hidden Layer 2");
    }

    for i in 0..nodes_h2 {
        let ny = dnn_y + i * node_spacing;
        draw_filled_circle_mut(&mut img, (dnn_x + 350, ny), 18, Rgb([0, 120, 255]));
        draw_filled_circle_mut(&mut img, (dnn_x + 350, ny), 12, Rgb([100, 200, 255]));
    }

    // --- SECTION 5: BEHAVIORAL OUTPUTS (Right) ---
    let out_x: i32 = 1500;
    let out_y: i32 = 150;
    if let Some(ref f) = font {
        draw_text_mut(&mut img, Rgb([255, 255, 0]), out_x, out_y - 80, PxScale::from(28.0), f, "OUTPUTS");
    }

    for i in 0..nodes_h2 {
        let oy = out_y + i * node_spacing;
        draw_filled_circle_mut(&mut img, (out_x + 30, oy), 20, Rgb([200, 150, 0]));
        draw_filled_circle_mut(&mut img, (out_x + 30, oy), 14, Rgb([255, 255, 0]));
        if let Some(ref f) = font {
            if (i as usize) < OUTPUT_LABELS.len() {
                draw_text_mut(&mut img, Rgb([255, 255, 0]), out_x + 65, oy - 10, PxScale::from(18.0), f, OUTPUT_LABELS[i as usize]);
            }
        }
        draw_line_segment_mut(&mut img, (dnn_x as f32 + 368.0, oy as f32), (out_x as f32 + 10.0, oy as f32), Rgb([80, 80, 0]));
    }

    if let Some(ref f) = font {
        let plastic_x = dnn_x + 50;
        let plastic_y = dnn_y + 400;
        draw_text_mut(&mut img, Rgb([255, 0, 255]), plastic_x - 100, plastic_y + 480, PxScale::from(20.0), f, "HEBBIAN PLASTICITY LOOP");
    }

    img.save(output_path).unwrap();
    println!("High-fidelity architectural graph saved to {}", output_path);
}
