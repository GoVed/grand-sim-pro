/*
 * Grand Sim Pro: Neural Network Visualization Utility (ab_glyph Implementation)
 */

use world_sim::agent::{Person, NUM_INPUTS, NUM_OUTPUTS, NUM_HIDDEN_MAX, INPUT_LABELS, OUTPUT_LABELS};
use world_sim::config::SimConfig;
use image::{RgbImage, Rgb};
use imageproc::drawing::{draw_line_segment_mut, draw_filled_circle_mut, draw_text_mut};
use ab_glyph::{FontArc, PxScale};
use std::path::Path;

#[test]
fn generate_neural_structure_png() {
    let config = SimConfig::default();
    let p = Person::new(0.0, 0.0, 0, &config);
    
    let output_path = "test_screenshots/neural_structure.png";
    if !Path::new("test_screenshots").exists() {
        let _ = std::fs::create_dir("test_screenshots");
    }

    let width = 1600;
    let height = 1200;
    let mut img = RgbImage::new(width, height);

    // Background
    for pixel in img.pixels_mut() {
        *pixel = Rgb([5, 10, 15]);
    }

    // Layer Positions
    let x_input = 150.0;
    let x_h1 = 550.0;
    let x_h2 = 950.0;
    let x_out = 1350.0;

    let h_count = p.state.hidden_count as usize;

    // --- 1. Draw Connections (W1: Input -> H1) ---
    for h in 0..h_count {
        let hy = 100.0 + (h as f32 / h_count as f32) * 1000.0;
        for k in 0..8 {
            let in_idx = p.genetics.w1_indices[h * 8 + k] as usize;
            let weight = p.genetics.w1_weights[h * 8 + k];
            if weight.abs() < 0.3 { continue; }

            let iy = 50.0 + (in_idx as f32 / NUM_INPUTS as f32) * 1100.0;
            let alpha = (weight.abs() * 0.5).min(1.0);
            let color = if weight > 0.0 { 
                Rgb([(0.0 * 255.0) as u8, (1.0 * 255.0 * alpha) as u8, (1.0 * 255.0 * alpha) as u8]) 
            } else { 
                Rgb([(1.0 * 255.0 * alpha) as u8, (0.0 * 255.0) as u8, (1.0 * 255.0 * alpha) as u8]) 
            };
            
            draw_line_segment_mut(&mut img, (x_input, iy), (x_h1, hy), color);
        }
    }

    // --- 2. Draw Connections (W2: H1 -> H2) ---
    for i in 0..h_count {
        let iy = 100.0 + (i as f32 / h_count as f32) * 1000.0;
        for j in 0..h_count {
            let weight = p.genetics.w2[i * NUM_HIDDEN_MAX + j];
            if weight.abs() < 0.6 { continue; } 

            let jy = 100.0 + (j as f32 / h_count as f32) * 1000.0;
            let alpha = (weight.abs() * 0.4).min(1.0);
            let color = if weight > 0.0 { Rgb([0, (255.0 * alpha) as u8, 0]) } else { Rgb([(255.0 * alpha) as u8, 0, 0]) };
            
            draw_line_segment_mut(&mut img, (x_h1, iy), (x_h2, jy), color);
        }
    }

    // --- 3. Draw Connections (W3: H2 -> Output) ---
    for h in 0..h_count {
        let hy = 100.0 + (h as f32 / h_count as f32) * 1000.0;
        for o in 0..NUM_OUTPUTS {
            let weight = p.genetics.w3[h * NUM_OUTPUTS + o];
            if weight.abs() < 0.4 { continue; }

            let oy = 50.0 + (o as f32 / NUM_OUTPUTS as f32) * 1100.0;
            let alpha = (weight.abs() * 0.6).min(1.0);
            let color = if weight > 0.0 { Rgb([(255.0 * alpha) as u8, (200.0 * alpha) as u8, 0]) } else { Rgb([0, 0, (255.0 * alpha) as u8]) };
            
            draw_line_segment_mut(&mut img, (x_h2, hy), (x_out, oy), color);
        }
    }

    // --- 4. Draw Nodes ---
    let font_data = std::fs::read("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf").ok()
        .or_else(|| std::fs::read("/usr/share/fonts/TTF/DejaVuSans.ttf").ok());
    
    let font = font_data.as_ref().and_then(|data| FontArc::try_from_vec(data.clone()).ok());

    // Inputs
    for i in (0..NUM_INPUTS).step_by(5) {
        let iy = 50.0 + (i as f32 / NUM_INPUTS as f32) * 1100.0;
        draw_filled_circle_mut(&mut img, (x_input as i32, iy as i32), 2, Rgb([150, 150, 150]));
        if i % 15 == 0 {
            if let Some(ref f) = font {
                draw_text_mut(&mut img, Rgb([200, 200, 200]), (x_input - 130.0) as i32, (iy - 7.0) as i32, PxScale::from(14.0), f, INPUT_LABELS[i]);
            }
        }
    }

    // H1 & H2
    for h in 0..h_count {
        let hy = 100.0 + (h as f32 / h_count as f32) * 1000.0;
        draw_filled_circle_mut(&mut img, (x_h1 as i32, hy as i32), 4, Rgb([255, 255, 255]));
        draw_filled_circle_mut(&mut img, (x_h2 as i32, hy as i32), 4, Rgb([255, 255, 255]));
    }

    // Outputs
    for o in 0..NUM_OUTPUTS {
        let oy = 50.0 + (o as f32 / NUM_OUTPUTS as f32) * 1100.0;
        draw_filled_circle_mut(&mut img, (x_out as i32, oy as i32), 5, Rgb([255, 255, 0]));
        if let Some(ref f) = font {
            draw_text_mut(&mut img, Rgb([255, 255, 0]), (x_out + 15.0) as i32, (oy - 7.0) as i32, PxScale::from(14.0), f, OUTPUT_LABELS[o]);
        }
    }

    img.save(output_path).unwrap();
    println!("Neural structure graph saved to {}", output_path);
}
