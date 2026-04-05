/*
 * Grand Sim Pro: A high-performance GPGPU evolutionary agent simulation.
 * Part of an independent research project into emergent biological complexity.
 *
 * Copyright (C) 2026 Ved Hirenkumar Suthar
 * Licensed under the GNU General Public License v3.0 or later.
 * * This software is provided "as is", without warranty of any kind.
 * See the LICENSE file in the project root for full license details.
 */

use macroquad::prelude::*;
use crate::shared::{VisualMode, SortCol, format_time};

pub fn draw_metrics(
    pop_count: usize, compute_time: u128, speed: usize, ticks: u64, tick_to_mins: f32,
    fps: i32, avg_fps: f32, low_1_fps: f32, current_visual_mode: VisualMode, show_inspector: bool, show_generation_graph: bool,
    paused: bool, restart_msg: bool
) {
    draw_rectangle(10.0, 10.0, 260.0, 280.0, Color::new(0.0, 0.04, 0.04, 0.9));
    draw_rectangle_lines(10.0, 10.0, 260.0, 280.0, 1.0, Color::new(0.0, 1.0, 0.8, 1.0));
    
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
    draw_text(&format!("Sim Time: {}", format_time(ticks, tick_to_mins)), 20.0, y, 16.0, WHITE);
    y += dy;
    draw_text(&format!("FPS: {}", fps), 20.0, y, 16.0, WHITE);
    y += dy;
    draw_text(&format!("Avg FPS: {:.1}", avg_fps), 20.0, y, 16.0, WHITE);
    y += dy;
    draw_text(&format!("1% Low: {:.1}", low_1_fps), 20.0, y, 16.0, WHITE);
    y += dy;
    let mode_str = match current_visual_mode {
        VisualMode::Default => "Default",
        VisualMode::Resources => "Resources",
        VisualMode::Age => "Age",
        VisualMode::Gender => "Gender",
        VisualMode::Pregnancy => "Pregnancy",
        VisualMode::MarketWealth => "Market Wealth",
        VisualMode::MarketFood => "Market Food",
        VisualMode::AskPrice => "Ask Price",
        VisualMode::BidPrice => "Bid Price",
        VisualMode::Infrastructure => "Infrastructure",
        VisualMode::DayNight => "Day/Night",
        VisualMode::Temperature => "Temperature",
        VisualMode::Tribes => "Identity / Tribes",
        VisualMode::Water => "Drinkable Water",
    };
    draw_text(&format!("Visuals [R]: {}", mode_str), 20.0, y, 16.0, WHITE);
    y += dy;
    draw_text(&format!("Inspector [TAB]: {}", if show_inspector { "OPEN" } else { "CLOSED" }), 20.0, y, 16.0, WHITE);
    y += dy;
    draw_text(&format!("Gen Graph [G]: {}", if show_generation_graph { "OPEN" } else { "CLOSED" }), 20.0, y, 16.0, WHITE);
    y += dy;
    draw_text("Save Agents [S]", 20.0, y, 16.0, WHITE);

    if paused { draw_text("PAUSED", screen_width() / 2.0 - 50.0, 30.0, 30.0, RED); }
    
    if restart_msg {
        let text = "POPULATION CRITICAL, RESTARTING...";
        let text_dims = measure_text(text, None, 40, 1.0);
        draw_text(text, screen_width() / 2.0 - text_dims.width / 2.0, screen_height() / 2.0, 40.0, RED);
    }
}

pub fn draw_visuals_panel(mx: f32, my: f32, left_clicked: bool, current_visual_mode: &mut VisualMode) {
    draw_rectangle(280.0, 10.0, 200.0, 360.0, Color::new(0.0, 0.04, 0.04, 0.9));
    draw_rectangle_lines(280.0, 10.0, 200.0, 360.0, 1.0, Color::new(0.0, 1.0, 0.8, 1.0));
    draw_text("VISUALS", 290.0, 30.0, 16.0, Color::new(0.0, 1.0, 0.8, 1.0));
    
    let modes = [
        (VisualMode::Default, "1. Default"),
        (VisualMode::Resources, "2. Resources"),
        (VisualMode::Age, "3. Age"),
        (VisualMode::Gender, "4. Gender"),
        (VisualMode::Pregnancy, "5. Pregnancy"),
        (VisualMode::MarketWealth, "6. Market Wealth"),
        (VisualMode::MarketFood, "7. Market Food"),
        (VisualMode::AskPrice, "8. Ask Price"),
        (VisualMode::BidPrice, "9. Bid Price"),
        (VisualMode::Infrastructure, "0. Infrastructure"),
        (VisualMode::Temperature, "T. Temperature"),
        (VisualMode::DayNight, "N. Day/Night"),
        (VisualMode::Tribes, "I. Identity / Tribes"),
        (VisualMode::Water, "W. Drinkable Water"),
    ];
    
    let mut vy = 55.0;
    for (mode, label) in modes.iter() {
        let is_hover = mx > 290.0 && mx < 470.0 && my > vy - 12.0 && my < vy + 4.0;
        let color = if *current_visual_mode == *mode { WHITE } else if is_hover { GRAY } else { DARKGRAY };
        draw_text(label, 290.0, vy, 16.0, color);
        if left_clicked && is_hover { *current_visual_mode = *mode; }
        vy += 22.0;
    }
}

pub fn draw_inspector(
    mx: f32, my: f32, left_clicked: bool, mouse_wheel_y: f32,
    inspector_agents: &mut Vec<(usize, crate::agent::Person)>,
    sort_col: &mut SortCol, sort_desc: &mut bool,
    inspector_scroll: &mut usize,
    selected_agent: &mut Option<crate::agent::Person>,
    followed_agent_id: &mut Option<u32>,
    show_inspector: &mut bool,
    tick_to_mins: f32, base_speed: f32
) {
    draw_rectangle(40.0, 40.0, screen_width() - 80.0, screen_height() - 80.0, Color::new(0.05, 0.05, 0.05, 0.95));
    draw_rectangle_lines(40.0, 40.0, screen_width() - 80.0, screen_height() - 80.0, 2.0, Color::new(0.0, 1.0, 0.8, 1.0));
    
    if let Some(a) = *selected_agent {
        let is_hover_back = mx > 60.0 && mx < 140.0 && my > 60.0 && my < 90.0;
        draw_rectangle(60.0, 60.0, 80.0, 30.0, if is_hover_back { Color::new(0.4, 0.4, 0.4, 1.0) } else { Color::new(0.2, 0.2, 0.2, 1.0) });
        draw_text("<- BACK", 65.0, 80.0, 20.0, WHITE);
        if left_clicked && is_hover_back { *selected_agent = None; }
        
        draw_text(&format!("Stats: Age {} | HP {:.1} | Food {:.0}g | H2O {:.1} | Wealth ${:.1}", format_time(a.age as u64, tick_to_mins), a.health, a.food, a.water, a.wealth), 160.0, 80.0, 20.0, WHITE);

        // --- Draw Memory Registers ---
        let mem_x = 900.0;
        let mem_y = 80.0;
        draw_text("Memory State:", mem_x, mem_y, 18.0, LIGHTGRAY);
        let mems = [a.mem1, a.mem2, a.mem3, a.mem4, a.mem5, a.mem6, a.mem7, a.mem8];
        for (idx, &m) in mems.iter().enumerate() {
            let m_cx = mem_x + 130.0 + idx as f32 * 24.0;
            let m_col = if m > 0.0 { Color::new(0.0, m.min(1.0), 0.8, 1.0) } else { Color::new(m.abs().min(1.0), 0.0, 0.0, 1.0) };
            draw_rectangle(m_cx, mem_y - 14.0, 20.0, 16.0, m_col);
            draw_rectangle_lines(m_cx, mem_y - 14.0, 20.0, 16.0, 1.0, DARKGRAY);
        }

        // --- Calculate Linearized Effective Influence Matrix ---
        let mut w_eff = [[0.0_f32; crate::agent::NUM_INPUTS]; crate::agent::NUM_OUTPUTS];
        let mut max_abs_val = 0.0_f32;
        let h_count = a.hidden_count as usize;
        
        for o in 0..crate::agent::NUM_OUTPUTS {
            for i in 0..crate::agent::NUM_INPUTS {
                let mut sum = 0.0;
                for h2 in 0..h_count {
                    let w3_val = a.w3[h2 * crate::agent::NUM_OUTPUTS + o];
                    let mut h1_sum = 0.0;
                    for h1 in 0..h_count {
                        let mut w1_val = 0.0;
                        for k in 0..8 {
                            if a.w1_indices[h1 * 8 + k] as usize == i {
                                w1_val = a.w1_weights[h1 * 8 + k];
                                break;
                            }
                        }
                        h1_sum += a.w2[h1 * crate::agent::NUM_HIDDEN_MAX + h2] * w1_val;
                    }
                    sum += w3_val * h1_sum;
                }
                w_eff[o][i] = sum;
                if sum.abs() > max_abs_val { max_abs_val = sum.abs(); }
            }
        }

        let input_labels = crate::agent::INPUT_LABELS;
        let output_labels = crate::agent::OUTPUT_LABELS;

        // --- Extract Baselines & Remove from reactive matrix ---
        let mut baselines = [0.0_f32; crate::agent::NUM_OUTPUTS];
        for o in 0..crate::agent::NUM_OUTPUTS {
            baselines[o] = w_eff[o][0]; // Input 0 is the constant Bias
            w_eff[o][0] = 0.0; // Remove bias from the reactive pathway calculations
        }
        
        let traits = [
            ("Hostility (Attack)", baselines[4]),
            ("Defensiveness", baselines[25]),
            ("Wanderlust (Speed)", baselines[1]),
            ("Laziness (Rest)", baselines[5]),
            ("Generosity (Drop Items)", baselines[2] + baselines[23]),
            ("Commerce (Trade)", baselines[19] + baselines[20]),
            ("Sociability (Comm/Mate)", baselines[3] + baselines[6] + baselines[7] + baselines[8] + baselines[9]),
            ("Curiosity (Learn)", baselines[10]),
            ("Construct (Road)", baselines[26]),
            ("Construct (House)", baselines[27]),
            ("Construct (Farm)", baselines[28]),
            ("Construct (Granary)", baselines[29]),
            ("Destroy (Infra)", baselines[30]),
        ];

        let trait_x = 60.0;
        let mut trait_y = 150.0;
        draw_text("INNATE PERSONALITY (Baseline Bias)", trait_x, trait_y, 18.0, WHITE);
        draw_text("Natural tendencies excluding environmental stimuli.", trait_x, trait_y + 20.0, 14.0, GRAY);
        trait_y += 50.0;

        let bar_w = 90.0; 
        for (name, val) in traits.iter() {
            draw_text(name, trait_x, trait_y + 5.0, 16.0, WHITE);
            let center_x = trait_x + 190.0 + bar_w;
            draw_line(center_x, trait_y - 10.0, center_x, trait_y + 10.0, 2.0, GRAY); 
            
            let draw_val = val.clamp(-3.0, 3.0) / 3.0; 
            let bar_len = draw_val.abs() * bar_w;
            let color = if *val > 0.0 { Color::new(0.0, 1.0, 0.5, 1.0) } else { Color::new(1.0, 0.2, 0.2, 1.0) };
            
            if *val > 0.0 { draw_rectangle(center_x, trait_y - 8.0, bar_len, 16.0, color); } 
            else { draw_rectangle(center_x - bar_len, trait_y - 8.0, bar_len, 16.0, color); }
            
            draw_text(&format!("{:.2}", val), trait_x + 190.0 + bar_w * 2.0 + 15.0, trait_y + 5.0, 16.0, color);
            trait_y += 35.0;
        }

        trait_y += 20.0;
        draw_text("STRONGEST SINGLE TRIGGERS", trait_x, trait_y, 18.0, WHITE);
        trait_y += 25.0;
        
        let mut all_weights = Vec::with_capacity(crate::agent::NUM_INPUTS * 26);
        for o in 0..crate::agent::NUM_OUTPUTS { for i in 1..crate::agent::NUM_INPUTS { all_weights.push((i, o, w_eff[o][i])); } }
        all_weights.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap_or(std::cmp::Ordering::Equal));
        
        for (idx, &(i, o, w)) in all_weights.iter().take(6).enumerate() {
            let color = if w > 0.0 { Color::new(0.0, 1.0, 0.5, 1.0) } else { Color::new(1.0, 0.2, 0.2, 1.0) };
            let text = format!("{}. {} -> {} ({:.2})", idx + 1, input_labels[i], output_labels[o], w);
            draw_text(&text, trait_x, trait_y, 15.0, color);
            trait_y += 25.0;
        }

        // --- Cognitive Architecture (Bipartite Graph) ---
        let graph_x = 520.0; // Shifted right to balance spacing with the Personality panel
        let graph_y = 150.0;
        draw_text("DOMINANT REACTIVE PATHWAYS", graph_x, graph_y, 18.0, WHITE);
        draw_text("How sensory inputs dynamically trigger behavioral outputs.", graph_x, graph_y + 20.0, 14.0, GRAY);

        let mut input_importance = [(0usize, 0.0_f32); crate::agent::NUM_INPUTS];
        for i in 1..crate::agent::NUM_INPUTS {
            let mut sum = 0.0;
            for o in 0..crate::agent::NUM_OUTPUTS { sum += w_eff[o][i].abs(); }
            input_importance[i] = (i, sum);
        }
        input_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut output_importance = [(0usize, 0.0_f32); crate::agent::NUM_OUTPUTS];
        for o in 0..crate::agent::NUM_OUTPUTS {
            let mut sum = 0.0;
            for i in 1..crate::agent::NUM_INPUTS { sum += w_eff[o][i].abs(); }
            output_importance[o] = (o, sum);
        }
        output_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_n = 12;
        let top_inputs = &input_importance[0..top_n];
        let top_outputs = &output_importance[0..top_n];

        let mut max_path_w = 0.0_f32;
        for &(i, _) in top_inputs {
            for &(o, _) in top_outputs {
                let w = w_eff[o][i].abs();
                if w > max_path_w { max_path_w = w; }
            }
        }

        let path_y_start = graph_y + 60.0;
        let left_col_x = graph_x;
        let right_col_x = graph_x + 250.0;
        let row_spacing = 35.0;

        let mut hovered_in = None;
        let mut hovered_out = None;
        
        // Detect Mouse Hover over Input/Output labels
        for (l_idx, &(i, _)) in top_inputs.iter().enumerate() {
            let y = path_y_start + l_idx as f32 * row_spacing;
            if mx > left_col_x - 10.0 && mx < left_col_x + 110.0 && my > y - 14.0 && my < y + 6.0 { hovered_in = Some(i); }
        }
        for (r_idx, &(o, _)) in top_outputs.iter().enumerate() {
            let y = path_y_start + r_idx as f32 * row_spacing;
            if mx > right_col_x - 10.0 && mx < right_col_x + 130.0 && my > y - 14.0 && my < y + 6.0 { hovered_out = Some(o); }
        }

        // S-Curve Bezier function for elegant biological routing
        let draw_bezier = |x1: f32, y1: f32, x2: f32, y2: f32, thickness: f32, color: Color| {
            let segments = 15;
            let mut prev_x = x1; let mut prev_y = y1;
            let cp_x = x1 + (x2 - x1) * 0.5;
            for i in 1..=segments {
                let t = i as f32 / segments as f32; let u = 1.0 - t;
                let x = u * u * u * x1 + 3.0 * u * u * t * cp_x + 3.0 * u * t * t * cp_x + t * t * t * x2;
                let y = u * u * u * y1 + 3.0 * u * u * t * y1 + 3.0 * u * t * t * y2 + t * t * t * y2;
                draw_line(prev_x, prev_y, x, y, thickness, color);
                prev_x = x; prev_y = y;
            }
        };

        for (l_idx, &(i, _)) in top_inputs.iter().enumerate() {
            for (r_idx, &(o, _)) in top_outputs.iter().enumerate() {
                let w = w_eff[o][i];
                if w.abs() > max_path_w * 0.15 { // Only draw significant pathways to avoid visual clutter
                    let is_active = hovered_in == Some(i) || hovered_out == Some(o);
                    let any_hovered = hovered_in.is_some() || hovered_out.is_some();
                    
                    let mut alpha = (w.abs() / max_path_w).clamp(0.15, 0.8);
                    let mut thickness = (w.abs() / max_path_w) * 5.0;
                    
                    // Apply hover dimming / highlighting
                    if any_hovered {
                        if is_active {
                            alpha = (w.abs() / max_path_w).clamp(0.4, 1.0) + 0.2; // Boost
                            thickness += 2.0;
                        } else {
                            alpha = 0.03; // Dim heavily to highlight selected
                        }
                    }

                    let color = if w > 0.0 { Color::new(0.0, 1.0, 0.5, alpha) } else { Color::new(1.0, 0.2, 0.2, alpha) };
                    
                    let x1 = left_col_x + 110.0; 
                    let y1 = path_y_start + l_idx as f32 * row_spacing - 5.0;
                    let x2 = right_col_x - 10.0; 
                    let y2 = path_y_start + r_idx as f32 * row_spacing - 5.0;
                    
                    draw_bezier(x1, y1, x2, y2, thickness, color);
                }
            }
        }

        for (l_idx, &(i, _)) in top_inputs.iter().enumerate() {
            let y = path_y_start + l_idx as f32 * row_spacing;
            let is_hovered = hovered_in == Some(i);
            let color = if is_hovered { WHITE } else if hovered_in.is_some() || hovered_out.is_some() { Color::new(0.4, 0.4, 0.4, 0.5) } else { LIGHTGRAY };
            draw_text(input_labels[i], left_col_x, y, if is_hovered { 18.0 } else { 16.0 }, color);
        }
        
        for (r_idx, &(o, _)) in top_outputs.iter().enumerate() {
            let y = path_y_start + r_idx as f32 * row_spacing;
            let is_hovered = hovered_out == Some(o);
            let color = if is_hovered { WHITE } else if hovered_in.is_some() || hovered_out.is_some() { Color::new(0.4, 0.4, 0.4, 0.5) } else { LIGHTGRAY };
            draw_text(output_labels[o], right_col_x, y, if is_hovered { 18.0 } else { 16.0 }, color);
        }
        
        // --- Full Effective Influence Heatmap ---
        let hm_x = 920.0;
        let hm_y = 150.0;
        draw_text("FULL INFLUENCE HEATMAP", hm_x, hm_y, 18.0, WHITE);
        draw_text("Matrix of Top Inputs vs All Outputs.", hm_x, hm_y + 20.0, 14.0, GRAY);
        
        let hm_start_y = hm_y + 50.0;
        let cell_size = 12.0;
        let cell_pad = 2.0;
        let top_16_inputs = &input_importance[0..16.min(input_importance.len())];
        let mut tooltip = None;

        for (row, _o) in (0..crate::agent::NUM_OUTPUTS).enumerate() {
            let cy = hm_start_y + row as f32 * (cell_size + cell_pad);
            draw_text(output_labels[row], hm_x + 16.0 * (cell_size + cell_pad) + 8.0, cy + 10.0, 14.0, Color::new(0.7, 0.7, 0.7, 0.8));
            for (col, &(i, _)) in top_16_inputs.iter().enumerate() {
                let w = w_eff[row][i];
                let cx = hm_x + col as f32 * (cell_size + cell_pad);
                let norm_w = (w.abs() / max_path_w).clamp(0.0, 1.0);
                let color = if w > 0.0 { Color::new(0.0, norm_w, 0.0, 1.0) } else { Color::new(norm_w, 0.0, 0.0, 1.0) };
                draw_rectangle(cx, cy, cell_size, cell_size, color);
                
                if mx > cx && mx < cx + cell_size && my > cy && my < cy + cell_size {
                    draw_rectangle_lines(cx - 1.0, cy - 1.0, cell_size + 2.0, cell_size + 2.0, 1.0, WHITE);
                    tooltip = Some((input_labels[i].to_string(), output_labels[row].to_string(), w));
                }
            }
        }
        
        if let Some((in_lbl, out_lbl, w_val)) = tooltip {
            let tt_str = format!("{} -> {}: {:.3}", in_lbl, out_lbl, w_val);
            let tt_w = measure_text(&tt_str, None, 16, 1.0).width + 16.0;
            draw_rectangle(mx + 15.0, my + 15.0, tt_w, 24.0, Color::new(0.1, 0.1, 0.1, 0.95));
            draw_rectangle_lines(mx + 15.0, my + 15.0, tt_w, 24.0, 1.0, GRAY);
            draw_text(&tt_str, mx + 23.0, my + 32.0, 16.0, if w_val > 0.0 { GREEN } else { RED });
        }
    } else {
        inspector_agents.sort_by(|a, b| {
            let cmp = match *sort_col {
                SortCol::Index => a.0.cmp(&b.0),
                SortCol::Age => a.1.age.partial_cmp(&b.1.age).unwrap_or(std::cmp::Ordering::Equal),
                SortCol::Health => a.1.health.partial_cmp(&b.1.health).unwrap_or(std::cmp::Ordering::Equal),
                SortCol::Food => a.1.food.partial_cmp(&b.1.food).unwrap_or(std::cmp::Ordering::Equal),
                SortCol::Wealth => a.1.wealth.partial_cmp(&b.1.wealth).unwrap_or(std::cmp::Ordering::Equal),
                SortCol::Gender => a.1.gender.partial_cmp(&b.1.gender).unwrap_or(std::cmp::Ordering::Equal),
                SortCol::Speed => a.1.speed.partial_cmp(&b.1.speed).unwrap_or(std::cmp::Ordering::Equal),
                SortCol::Heading => a.1.heading.partial_cmp(&b.1.heading).unwrap_or(std::cmp::Ordering::Equal),
                SortCol::State => a.1.is_pregnant.partial_cmp(&b.1.is_pregnant).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.1.rest_intent.partial_cmp(&b.1.rest_intent).unwrap_or(std::cmp::Ordering::Equal)),
                SortCol::Outputs => a.1.reproduce_desire.partial_cmp(&b.1.reproduce_desire).unwrap_or(std::cmp::Ordering::Equal),
            };
            if *sort_desc { cmp.reverse() } else { cmp }
        });

        let headers = [
            ("ID", 60.0, SortCol::Index), ("Age", 120.0, SortCol::Age), ("HP", 180.0, SortCol::Health), 
            ("Fd", 230.0, SortCol::Food), ("Wlth", 300.0, SortCol::Wealth), ("Gen", 360.0, SortCol::Gender), 
            ("Spd", 410.0, SortCol::Speed), ("Dir", 460.0, SortCol::Heading), ("State", 500.0, SortCol::State), 
            ("Markets (Buy, Sel, Ask, Bid)", 800.0, SortCol::Outputs)
        ];

        for (label, hx, col) in headers.iter() {
            let is_hover = mx > *hx && mx < *hx + 40.0 && my > 50.0 && my < 80.0;
            let color = if *sort_col == *col { Color::new(0.0, 1.0, 0.8, 1.0) } else if is_hover { GRAY } else { WHITE };
            draw_text(label, *hx, 70.0, 20.0, color);
            if left_clicked && is_hover {
                if *sort_col == *col { *sort_desc = !*sort_desc; } else { *sort_col = *col; *sort_desc = true; }
            }
        }

        let row_h = 20.0;
        let visible = 22;
        if mouse_wheel_y < 0.0 { *inspector_scroll = inspector_scroll.saturating_add(1); }
        if mouse_wheel_y > 0.0 { *inspector_scroll = inspector_scroll.saturating_sub(1); }
        *inspector_scroll = (*inspector_scroll).min(inspector_agents.len().saturating_sub(visible));

        for i in 0..visible {
            let idx = *inspector_scroll + i;
            if idx >= inspector_agents.len() { break; }
            let (a_id, a) = &inspector_agents[idx];
            let y = 100.0 + i as f32 * row_h;
            
            let loc_x = 1050.0;
            let is_hover_locate = mx > loc_x && mx < loc_x + 60.0 && my > y - 12.0 && my < y + 4.0;
            
            if mx > 50.0 && mx < screen_width() - 50.0 && my > y - 15.0 && my < y + 5.0 {
                draw_rectangle(50.0, y - 15.0, screen_width() - 100.0, row_h, Color::new(0.2, 0.2, 0.2, 0.8));
                if left_clicked {
                    if is_hover_locate { *followed_agent_id = Some(a.id); *show_inspector = false; *selected_agent = None; } 
                    else { *selected_agent = Some(*a); }
                }
            }

            draw_text(&format!("{}", a_id), 60.0, y, 16.0, WHITE);
            draw_text(&format!("{}", format_time(a.age as u64, tick_to_mins)), 120.0, y, 16.0, WHITE);
            draw_text(&format!("{:.0}", a.health), 180.0, y, 16.0, WHITE);
            draw_text(&format!("{:.0}g", a.food), 230.0, y, 16.0, WHITE);
            draw_text(&format!("${:.0}", a.wealth), 300.0, y, 16.0, WHITE);
            draw_text(if a.gender > 0.5 { "M" } else { "F" }, 360.0, y, 16.0, WHITE);
            
            draw_text(&format!("{:.1}", a.speed), 410.0, y, 16.0, WHITE);
            let spd_ratio = (a.speed / base_speed).clamp(0.0, 1.0);
            draw_rectangle(410.0, y + 4.0, spd_ratio * 25.0, 2.0, Color::new(0.0, 1.0, 0.5, 1.0));
            
            let dir_cx = 470.0; let dir_cy = y - 5.0;
            let dx = a.heading.cos() * 8.0; let dy = a.heading.sin() * 8.0;
            draw_line(dir_cx - dx, dir_cy - dy, dir_cx + dx, dir_cy + dy, 1.5, LIGHTGRAY);
            draw_circle(dir_cx + dx, dir_cy + dy, 2.0, WHITE);
            
            let mut st_x = 500.0;
            if a.is_pregnant > 0.5 { draw_text("[PRG]", st_x, y, 14.0, YELLOW); st_x += 35.0; }
            if a.rest_intent > 0.5 { draw_text("[Zzz]", st_x, y, 14.0, SKYBLUE); st_x += 35.0; }
            if a.attack_intent > 0.5 { draw_text("[ATK]", st_x, y, 14.0, RED); st_x += 35.0; }
            if a.defend_intent > 0.5 { draw_text("[DEF]", st_x, y, 14.0, GREEN); st_x += 35.0; }
            if a.build_road_intent > 0.5 { draw_text("[ROD]", st_x, y, 14.0, GRAY); st_x += 35.0; }
            if a.build_house_intent > 0.5 { draw_text("[HOU]", st_x, y, 14.0, ORANGE); st_x += 35.0; }
            if a.build_farm_intent > 0.5 { draw_text("[FRM]", st_x, y, 14.0, GREEN); st_x += 35.0; }
            if a.build_storage_intent > 0.5 { draw_text("[STO]", st_x, y, 14.0, MAGENTA); st_x += 35.0; }
            if a.destroy_infra_intent > 0.5 { draw_text("[DEL]", st_x, y, 14.0, RED); }
            
            let out_str = format!("B:{:.1} S:{:.1} A:{:.1} B:{:.1}", a.buy_intent, a.sell_intent, a.ask_price, a.bid_price);
            draw_text(&out_str, 800.0, y, 16.0, WHITE);
            draw_text("[Locate]", loc_x, y, 16.0, if is_hover_locate { YELLOW } else { LIGHTGRAY });
        }
        draw_text(&format!("Showing {} - {} of {}", *inspector_scroll, (*inspector_scroll + visible).min(inspector_agents.len()), inspector_agents.len()), 60.0, 550.0, 16.0, GRAY);
        draw_text("Scroll to view more. Click row to inspect Neural Network.", 300.0, 550.0, 16.0, GRAY);
    }
}

pub fn draw_tracker(mx: f32, my: f32, left_clicked: bool, a: &crate::agent::Person, followed_agent_id: &mut Option<u32>, show_inspector: &mut bool, tick_to_mins: f32) {
    let panel_w = 260.0; let panel_h = 360.0;
    let panel_x = screen_width() - panel_w - 20.0; let panel_y = 20.0;
    draw_rectangle(panel_x, panel_y, panel_w, panel_h, Color::new(0.0, 0.04, 0.04, 0.9));
    draw_rectangle_lines(panel_x, panel_y, panel_w, panel_h, 1.0, Color::new(0.0, 1.0, 0.8, 1.0));

    let close_x = panel_x + panel_w - 30.0; let close_y = panel_y + 10.0;
    let is_close_hover = mx > close_x && mx < close_x + 20.0 && my > close_y && my < close_y + 20.0;
    draw_text("X", close_x, close_y + 15.0, 20.0, if is_close_hover { RED } else { GRAY });
    if left_clicked && is_close_hover { *followed_agent_id = None; *show_inspector = true; }

    let mut py = panel_y + 30.0; let dy = 22.0;
    draw_text("AGENT TRACKER", panel_x + 20.0, py, 18.0, Color::new(0.0, 1.0, 0.8, 1.0)); py += dy;
    draw_text(&format!("ID: {}", a.id), panel_x + 20.0, py, 16.0, WHITE); py += dy;
    draw_text(&format!("Age: {}", format_time(a.age as u64, tick_to_mins)), panel_x + 20.0, py, 16.0, WHITE); py += dy;
    draw_text(&format!("Health: {:.1}", a.health), panel_x + 20.0, py, 16.0, WHITE); py += dy;
    draw_text(&format!("Food: {:.0}g | H2O: {:.1}", a.food, a.water), panel_x + 20.0, py, 16.0, WHITE); py += dy;
    draw_text(&format!("Wealth: ${:.1}", a.wealth), panel_x + 20.0, py, 16.0, WHITE); py += dy;
    draw_text(&format!("Gender: {}", if a.gender > 0.5 { "Male" } else { "Female" }), panel_x + 20.0, py, 16.0, WHITE); py += dy;
    draw_text(&format!("Speed: {:.2}", a.speed), panel_x + 20.0, py, 16.0, WHITE); py += dy;
    
    let mut state_str = String::new();
    if a.health <= 0.0 { state_str.push_str("[DEAD] "); }
    if a.is_pregnant > 0.5 { state_str.push_str("[PRG] "); }
    if a.rest_intent > 0.5 { state_str.push_str("[Zzz] "); }
    if a.attack_intent > 0.5 { state_str.push_str("[ATK] "); }
    if a.defend_intent > 0.5 { state_str.push_str("[DEF] "); }
    if a.build_road_intent > 0.5 { state_str.push_str("[ROD] "); }
    if a.build_house_intent > 0.5 { state_str.push_str("[HOU] "); }
    if a.build_farm_intent > 0.5 { state_str.push_str("[FRM] "); }
    if a.build_storage_intent > 0.5 { state_str.push_str("[STO] "); }
    if a.destroy_infra_intent > 0.5 { state_str.push_str("[DEL] "); }
    if state_str.is_empty() { state_str.push_str("[IDLE]"); }
    
    draw_text(&format!("State: {}", state_str), panel_x + 20.0, py, 16.0, WHITE); py += dy + 10.0;
    draw_text("INTENTS:", panel_x + 20.0, py, 16.0, GRAY); py += dy;
    draw_text(&format!("Buy: {:.2} | Sell: {:.2}", a.buy_intent, a.sell_intent), panel_x + 20.0, py, 16.0, WHITE); py += dy;
    draw_text(&format!("Ask: {:.2} | Bid: {:.2}", a.ask_price, a.bid_price), panel_x + 20.0, py, 16.0, WHITE); py += dy;
    draw_text(&format!("Reproduce: {:.2}", a.reproduce_desire), panel_x + 20.0, py, 16.0, WHITE);
}

pub fn draw_generation_graph(times: &[u64], tick_to_mins: f32) {
    if times.is_empty() { return; }

    let panel_w = 400.0;
    let panel_h = 300.0;
    let panel_x = screen_width() / 2.0 - panel_w / 2.0;
    let panel_y = screen_height() / 2.0 - panel_h / 2.0;
    
    let margin = 40.0;
    let graph_w = panel_w - margin * 2.0;
    let graph_h = panel_h - margin * 2.0;
    let graph_x = panel_x + margin;
    let graph_y = panel_y + margin;

    // Draw background
    draw_rectangle(panel_x, panel_y, panel_w, panel_h, Color::new(0.05, 0.05, 0.05, 0.95));
    draw_rectangle_lines(panel_x, panel_y, panel_w, panel_h, 2.0, Color::new(0.0, 1.0, 0.8, 1.0));

    // Draw Title and Axes Labels
    draw_text("Generation Survival Time", panel_x + 100.0, panel_y + 25.0, 20.0, WHITE);
    draw_text("Generation", graph_x + graph_w / 2.0 - 40.0, graph_y + graph_h + 25.0, 16.0, GRAY);
    draw_text("Time", panel_x + 5.0, panel_y + 25.0, 16.0, GRAY);

    // Find max values for scaling
    let max_time = (*times.iter().max().unwrap_or(&1)).max(1);
    let max_gen = (times.len() - 1).max(1);

    // Draw Y-axis labels
    for i in 0..=5 {
        let val = max_time as f32 * (i as f32 / 5.0);
        let y = graph_y + graph_h - (graph_h * (i as f32 / 5.0));
        draw_line(graph_x - 5.0, y, graph_x, y, 1.0, GRAY);
        let time_str = format_time(val as u64, tick_to_mins);
        draw_text(&time_str, graph_x - 38.0, y + 4.0, 14.0, GRAY);
    }

    // Draw X-axis labels
    let x_label_step = (max_gen as f32 / 10.0).ceil().max(1.0) as usize;
    for i in (0..=max_gen).step_by(x_label_step) {
        let x = graph_x + (i as f32 / max_gen as f32) * graph_w;
        draw_line(x, graph_y + graph_h, x, graph_y + graph_h + 5.0, 1.0, GRAY);
        draw_text(&format!("{}", i + 1), x - 5.0, graph_y + graph_h + 20.0, 14.0, GRAY);
    }

    // Draw graph lines
    let mut last_x = 0.0;
    let mut last_y = 0.0;
    for (i, &time) in times.iter().enumerate() {
        let px = graph_x + (i as f32 / max_gen as f32) * graph_w;
        let py = graph_y + graph_h - (time as f32 / max_time as f32) * graph_h;
        
        draw_circle(px, py, 3.0, YELLOW);

        if i > 0 {
            draw_line(last_x, last_y, px, py, 2.0, Color::new(0.0, 1.0, 0.8, 1.0));
        }
        last_x = px;
        last_y = py;
    }
}