/*
 * Grand Sim Pro: Architectural Neural Visualization (White Theme with Annotations)
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

    let width: u32 = 2000;
    let height: u32 = 1200;
    let mut img = RgbImage::new(width, height);

    // Clean White Background
    for pixel in img.pixels_mut() { *pixel = Rgb([255, 255, 255]); }

    let font_data = std::fs::read("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf").ok()
        .or_else(|| std::fs::read("/usr/share/fonts/TTF/DejaVuSans.ttf").ok());
    let font = font_data.as_ref().and_then(|data| FontArc::try_from_vec(data.clone()).ok());

    // Color Palette for White Theme
    let text_color = Rgb([20, 20, 30]);
    let label_color = Rgb([80, 80, 100]);
    let box_bg = Rgb([245, 245, 250]);
    let box_border = Rgb([200, 200, 210]);
    let accent_blue = Rgb([0, 100, 200]);
    let accent_orange = Rgb([200, 80, 0]);

    // --- SECTION 1: GLOBAL/META INPUTS ---
    let meta_x: i32 = 100;
    let meta_y: i32 = 150;
    let meta_w: u32 = 300;
    let meta_h: u32 = 400;
    draw_filled_rect_mut(&mut img, Rect::at(meta_x, meta_y).of_size(meta_w, meta_h), box_bg);
    draw_hollow_rect_mut(&mut img, Rect::at(meta_x, meta_y).of_size(meta_w, meta_h), box_border);
    if let Some(ref f) = font {
        draw_text_mut(&mut img, text_color, meta_x, meta_y - 45, PxScale::from(30.0), f, "1. GLOBAL/META INPUTS");
        draw_text_mut(&mut img, label_color, meta_x, meta_y - 15, PxScale::from(14.0), f, "Proprioception: Health, Food, Water, Age, Memory");
    }

    for i in 0..14usize {
        let iy = meta_y + 30 + (i as i32) * 26;
        draw_filled_circle_mut(&mut img, (meta_x + 30, iy), 8, accent_blue);
        if let Some(ref f) = font {
            if i < INPUT_LABELS.len() {
                draw_text_mut(&mut img, label_color, meta_x + 50, iy - 7, PxScale::from(14.0), f, INPUT_LABELS[i]);
            }
        }
    }

    // --- SECTION 2: 5x5 VISION GRID ---
    let vis_x: i32 = 100;
    let vis_y: i32 = 650;
    let grid_cell: i32 = 45;
    if let Some(ref f) = font {
        draw_text_mut(&mut img, text_color, vis_x, vis_y - 45, PxScale::from(30.0), f, "2. 5x5 SPATIAL VISION");
        draw_text_mut(&mut img, label_color, vis_x, vis_y - 15, PxScale::from(14.0), f, "Active Field: 3 rows ahead, 1 row behind");
    }
    for gy in 0..5i32 {
        for gx in 0..5i32 {
            let cx = vis_x + gx * grid_cell;
            let cy = vis_y + gy * grid_cell;
            let color = if gx == 2 && gy == 3 { Rgb([255, 200, 0]) } else { Rgb([230, 240, 235]) };
            draw_filled_rect_mut(&mut img, Rect::at(cx, cy).of_size((grid_cell - 4) as u32, (grid_cell - 4) as u32), color);
            draw_hollow_rect_mut(&mut img, Rect::at(cx, cy).of_size((grid_cell - 4) as u32, (grid_cell - 4) as u32), Rgb([180, 200, 190]));
        }
    }

    // --- SECTION 3: SPATIAL CNN ---
    let cnn_x: i32 = 600;
    let cnn_y: i32 = 600;
    if let Some(ref f) = font {
        draw_text_mut(&mut img, text_color, cnn_x, cnn_y - 150, PxScale::from(30.0), f, "3. SPATIAL CNN");
        draw_text_mut(&mut img, accent_orange, cnn_x, cnn_y - 120, PxScale::from(18.0), f, "Pattern Recognition (8 Kernels)");
        draw_text_mut(&mut img, label_color, cnn_x, cnn_y + 30, PxScale::from(14.0), f, "Convolves vision grid into");
        draw_text_mut(&mut img, label_color, cnn_x, cnn_y + 50, PxScale::from(14.0), f, "high-level topological features.");
    }
    for i in 0..8i32 {
        let offset = i * 18;
        let mx = cnn_x + offset;
        let my = cnn_y - offset - 100;
        draw_filled_rect_mut(&mut img, Rect::at(mx, my).of_size(130, 130), Rgb([255 - (i as u8 * 15), 230 - (i as u8 * 10), 200]));
        draw_hollow_rect_mut(&mut img, Rect::at(mx, my).of_size(130, 130), accent_orange);
        if i == 0 {
            draw_line_segment_mut(&mut img, (vis_x as f32 + 100.0, vis_y as f32 + 100.0), (mx as f32, my as f32 + 65.0), Rgb([150, 150, 150]));
        }
    }

    // --- SECTION 4: DEEP NEURAL NETWORK ---
    let dnn_x: i32 = 1050;
    let dnn_y: i32 = 150;
    let dnn_w: u32 = 450;
    let dnn_h: u32 = 900;
    draw_hollow_rect_mut(&mut img, Rect::at(dnn_x - 50, dnn_y - 50).of_size(dnn_w + 100, dnn_h + 100), box_border);
    if let Some(ref f) = font {
        draw_text_mut(&mut img, text_color, dnn_x + 60, dnn_y - 80, PxScale::from(32.0), f, "4. INTEGRATION (MLP)");
        draw_text_mut(&mut img, label_color, dnn_x + 50, dnn_y - 40, PxScale::from(16.0), f, "Dense processing of global + spatial data.");
    }

    let nodes_h1 = 12;
    let nodes_h2 = 12;
    let node_spacing = 70;

    for i in 0..nodes_h1 {
        let ny = dnn_y + i * node_spacing;
        draw_filled_circle_mut(&mut img, (dnn_x + 50, ny), 18, accent_blue);
        draw_filled_circle_mut(&mut img, (dnn_x + 50, ny), 10, Rgb([255, 255, 255]));
        for j in 0..nodes_h2 {
            let n2y = dnn_y + j * node_spacing;
            draw_line_segment_mut(&mut img, (dnn_x as f32 + 68.0, ny as f32), (dnn_x as f32 + 332.0, n2y as f32), Rgb([230, 235, 240]));
        }
    }
    if let Some(ref f) = font {
        draw_text_mut(&mut img, accent_blue, dnn_x + 20, dnn_y + (dnn_h as i32) + 20, PxScale::from(18.0), f, "Hidden Layer 1");
        draw_text_mut(&mut img, accent_blue, dnn_x + 320, dnn_y + (dnn_h as i32) + 20, PxScale::from(18.0), f, "Hidden Layer 2");
    }

    for i in 0..nodes_h2 {
        let ny = dnn_y + i * node_spacing;
        draw_filled_circle_mut(&mut img, (dnn_x + 350, ny), 18, accent_blue);
        draw_filled_circle_mut(&mut img, (dnn_x + 350, ny), 10, Rgb([255, 255, 255]));
    }

    // --- SECTION 5: BEHAVIORAL OUTPUTS ---
    let out_x: i32 = 1650;
    let out_y: i32 = 150;
    if let Some(ref f) = font {
        draw_text_mut(&mut img, text_color, out_x, out_y - 80, PxScale::from(32.0), f, "5. OUTPUTS");
        draw_text_mut(&mut img, label_color, out_x, out_y - 45, PxScale::from(16.0), f, "Decision & Action");
    }

    for i in 0..nodes_h2 {
        let oy = out_y + i * node_spacing;
        draw_filled_circle_mut(&mut img, (out_x + 30, oy), 20, Rgb([255, 180, 0]));
        draw_filled_circle_mut(&mut img, (out_x + 30, oy), 14, Rgb([255, 255, 100]));
        if let Some(ref f) = font {
            if (i as usize) < OUTPUT_LABELS.len() {
                draw_text_mut(&mut img, text_color, out_x + 65, oy - 10, PxScale::from(18.0), f, OUTPUT_LABELS[i as usize]);
            }
        }
        draw_line_segment_mut(&mut img, (dnn_x as f32 + 368.0, oy as f32), (out_x as f32 + 10.0, oy as f32), Rgb([220, 200, 100]));
    }

    // Hebbian Plasticity Annotation
    if let Some(ref f) = font {
        let h_loop_x = dnn_x + 50;
        let h_loop_y = dnn_y + 150;
        draw_text_mut(&mut img, Rgb([200, 0, 200]), h_loop_x - 180, h_loop_y + 650, PxScale::from(24.0), f, "HEBBIAN LEARNING LOOP");
        draw_text_mut(&mut img, Rgb([150, 0, 150]), h_loop_x - 180, h_loop_y + 680, PxScale::from(14.0), f, "Online weight updates based on dopamine");
        
        // Arrow representing feedback
        draw_line_segment_mut(&mut img, (h_loop_x as f32 - 40.0, h_loop_y as f32 + 600.0), (h_loop_x as f32 - 100.0, h_loop_y as f32 + 630.0), Rgb([200, 0, 200]));
        draw_line_segment_mut(&mut img, (h_loop_x as f32 - 100.0, h_loop_y as f32 + 630.0), (h_loop_x as f32 - 40.0, h_loop_y as f32 + 660.0), Rgb([200, 0, 200]));
    }

    img.save(output_path).unwrap();
    println!("Descriptive white-theme architectural graph saved to {}", output_path);
}
