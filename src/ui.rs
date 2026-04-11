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
use crate::agent::{NUM_OUTPUTS, Person};
use crate::ui_logic::{self, UIColor};

impl From<UIColor> for Color {
    fn from(c: UIColor) -> Self {
        Color::new(c.r, c.g, c.b, c.a)
    }
}

pub fn apply_sort(agents: &mut Vec<(usize, Person)>, col: SortCol, desc: bool) {
    agents.sort_by(|a, b| {
        let (p1, p2) = (&a.1, &b.1);
        let res = match col {
            SortCol::Index => a.0.cmp(&b.0),
            SortCol::Age => p1.age.partial_cmp(&p2.age).unwrap_or(std::cmp::Ordering::Equal),
            SortCol::Health => p1.health.partial_cmp(&p2.health).unwrap_or(std::cmp::Ordering::Equal),
            SortCol::Food => p1.food.partial_cmp(&p2.food).unwrap_or(std::cmp::Ordering::Equal),
            SortCol::Wealth => p1.wealth.partial_cmp(&p2.wealth).unwrap_or(std::cmp::Ordering::Equal),
            SortCol::Gender => p1.gender.partial_cmp(&p2.gender).unwrap_or(std::cmp::Ordering::Equal),
            _ => std::cmp::Ordering::Equal,
        };
        if desc { res.reverse() } else { res }
    });
}

pub fn draw_metrics(
    pop_count: usize, compute_time: f32, speed: usize, ticks: u64, tick_to_mins: f32,
    fps: i32, avg_fps: f32, low_1_fps: f32, current_visual_mode: VisualMode, show_inspector: bool, show_generation_graph: bool,
    show_config_panel: bool, paused: bool, restart_msg: bool
) {
    draw_rectangle(10.0, 10.0, 260.0, 300.0, Color::new(0.0, 0.04, 0.04, 0.9));
    draw_rectangle_lines(10.0, 10.0, 260.0, 300.0, 1.0, Color::new(0.0, 1.0, 0.8, 1.0));
    
    let mut y = 30.0;
    let dy = 20.0;
    
    draw_text("REAL-TIME METRICS", 20.0, y, 16.0, Color::new(0.0, 1.0, 0.8, 1.0));
    y += dy;
    draw_text(&format!("Population: {}", pop_count), 20.0, y, 16.0, WHITE);
    y += dy;
    draw_text(&format!("Compute: {:.2}ms/loop", compute_time), 20.0, y, 16.0, WHITE);
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
        VisualMode::Default => "Default", VisualMode::Resources => "Resources", VisualMode::Age => "Age",
        VisualMode::Gender => "Gender", VisualMode::Pregnancy => "Pregnancy", VisualMode::MarketWealth => "Market Wealth",
        VisualMode::MarketFood => "Market Food", VisualMode::AskPrice => "Ask Price", VisualMode::BidPrice => "Bid Price",
        VisualMode::Infrastructure => "Infrastructure", VisualMode::DayNight => "Day/Night", VisualMode::Temperature => "Temperature",
        VisualMode::Tribes => "Identity / Tribes", VisualMode::Water => "Drinkable Water",
    };
    draw_text(&format!("Visuals [R]: {}", mode_str), 20.0, y, 16.0, WHITE);
    y += dy;
    draw_text(&format!("Inspector [TAB]: {}", if show_inspector { "OPEN" } else { "CLOSED" }), 20.0, y, 16.0, WHITE);
    y += dy;
    draw_text(&format!("Gen Graph [G]: {}", if show_generation_graph { "OPEN" } else { "CLOSED" }), 20.0, y, 16.0, WHITE);
    y += dy;
    draw_text(&format!("Config Panel [C]: {}", if show_config_panel { "OPEN" } else { "CLOSED" }), 20.0, y, 16.0, WHITE);
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
        (VisualMode::Default, "1. Default"), (VisualMode::Resources, "2. Resources"), (VisualMode::Age, "3. Age"),
        (VisualMode::Gender, "4. Gender"), (VisualMode::Pregnancy, "5. Pregnancy"), (VisualMode::MarketWealth, "6. Market Wealth"),
        (VisualMode::MarketFood, "7. Market Food"), (VisualMode::AskPrice, "8. Ask Price"), (VisualMode::BidPrice, "9. Bid Price"),
        (VisualMode::Infrastructure, "0. Infrastructure"), (VisualMode::Temperature, "T. Temperature"), (VisualMode::DayNight, "N. Day/Night"),
        (VisualMode::Tribes, "I. Identity / Tribes"), (VisualMode::Water, "W. Drinkable Water"),
    ];
    
    let mut vy = 55.0;
    for (mode, label) in modes.iter() {
        let is_hover = mx > 290.0 && mx < 470.0 && my > vy - 12.0 && my < vy + 4.0;
        let color = if *current_visual_mode == *mode { Color::new(0.0, 1.0, 0.8, 1.0) } else if is_hover { WHITE } else { GRAY };
        draw_text(label, 295.0, vy, 16.0, color);
        if left_clicked && is_hover { *current_visual_mode = *mode; }
        vy += 22.0;
    }
}

pub fn draw_inspector(
    mx: f32, my: f32, left_clicked: bool, mouse_wheel_y: f32,
    agents: &mut Vec<(usize, Person)>,
    sort_col: &mut SortCol, sort_desc: &mut bool, scroll: &mut usize,
    selected_agent: &mut Option<Person>,
    followed_agent_id: &mut Option<u32>,
    show_inspector: &mut bool,
    tick_to_mins: f32,
) {
    let panel_w = 600.0;
    let panel_h = screen_height() - 40.0;
    let panel_x = 10.0;
    let panel_y = 20.0;

    draw_rectangle(panel_x, panel_y, panel_w, panel_h, Color::new(0.0, 0.05, 0.05, 0.95));
    draw_rectangle_lines(panel_x, panel_y, panel_w, panel_h, 2.0, Color::new(0.0, 1.0, 0.8, 1.0));

    let mut y = panel_y + 30.0;
    draw_text("AGENT INSPECTOR", panel_x + 20.0, y, 24.0, Color::new(0.0, 1.0, 0.8, 1.0));
    
    let close_rect = Rect::new(panel_x + panel_w - 40.0, panel_y + 10.0, 30.0, 30.0);
    draw_rectangle(close_rect.x, close_rect.y, close_rect.w, close_rect.h, RED);
    draw_text("X", close_rect.x + 8.0, close_rect.y + 22.0, 24.0, WHITE);
    if left_clicked && close_rect.contains(vec2(mx, my)) { *show_inspector = false; }

    y += 40.0;

    let cols = [
        (SortCol::Index, "ID", 40.0), (SortCol::Age, "Age", 80.0), (SortCol::Health, "HP", 60.0),
        (SortCol::Food, "Food", 80.0), (SortCol::Wealth, "Wealth", 80.0), (SortCol::Gender, "Sex", 60.0),
    ];

    let mut cur_x = panel_x + 20.0;
    for (col, label, width) in cols.iter() {
        let header_rect = Rect::new(cur_x, y - 25.0, *width, 40.0);
        let is_hover = header_rect.contains(vec2(mx, my));
        let color = if *sort_col == *col { Color::new(0.0, 1.0, 0.8, 1.0) } else if is_hover { WHITE } else { GRAY };
        
        draw_text(label, cur_x, y, 16.0, color);
        if left_clicked && is_hover {
            if *sort_col == *col { *sort_desc = !*sort_desc; }
            else { 
                *sort_col = *col; 
                *sort_desc = true; 
            }
            apply_sort(agents, *sort_col, *sort_desc);
        }
        cur_x += width;
    }
    y += 40.0;

    if mx > panel_x && mx < panel_x + panel_w && my > panel_y && my < panel_y + panel_h {
        if mouse_wheel_y > 0.0 { *scroll = scroll.saturating_sub(1); }
        if mouse_wheel_y < 0.0 { *scroll = (*scroll + 1).min(agents.len().saturating_sub(1)); }
    }

    let items_to_show = ((panel_h - (y - panel_y)) / 25.0) as usize;
    for i in *scroll..(*scroll + items_to_show).min(agents.len()) {
        let (idx, a) = &agents[i];
        let row_rect = Rect::new(panel_x + 20.0, y - 15.0, panel_w - 40.0, 20.0);
        let is_hover = row_rect.contains(vec2(mx, my));
        let color = if is_hover { WHITE } else { Color::new(0.8, 0.8, 0.8, 1.0) };
        let mut cur_x = panel_x + 20.0;
        draw_text(&format!("{}", idx), cur_x, y, 14.0, color); cur_x += 40.0;
        draw_text(&format!("{}", format_time(a.age as u64, tick_to_mins)), cur_x, y, 14.0, color); cur_x += 80.0;
        draw_text(&format!("{:.0}", a.health), cur_x, y, 14.0, color); cur_x += 60.0;
        draw_text(&format!("{:.0}", a.food), cur_x, y, 14.0, color); cur_x += 80.0;
        draw_text(&format!("{:.0}", a.wealth), cur_x, y, 14.0, color); cur_x += 80.0;
        draw_text(if a.gender > 0.5 { "M" } else { "F" }, cur_x, y, 14.0, color); cur_x += 60.0;
        
        let locate_rect = Rect::new(cur_x, y - 12.0, 70.0, 16.0);
        let locate_hover = locate_rect.contains(vec2(mx, my));
        draw_rectangle(locate_rect.x, locate_rect.y, locate_rect.w, locate_rect.h, if locate_hover { YELLOW } else { Color::new(0.2, 0.2, 0.2, 1.0) });
        draw_text("[LOCATE]", locate_rect.x + 5.0, locate_rect.y + 12.0, 12.0, if locate_hover { BLACK } else { WHITE });
        
        if left_clicked && locate_hover { *followed_agent_id = Some(a.id); *show_inspector = false; }
        if left_clicked && is_hover && !locate_hover { *selected_agent = Some(*a); }
        y += 25.0;
    }

    if let Some(a) = selected_agent {
        draw_agent_detail_view(a, tick_to_mins);
    }
}

fn draw_agent_detail_view(a: &Person, tick_to_mins: f32) {
    let panel_w = 550.0;
    let panel_h = screen_height() - 40.0;
    let panel_x = screen_width() - panel_w - 20.0;
    let panel_y = 20.0;

    draw_rectangle(panel_x, panel_y, panel_w, panel_h, Color::new(0.0, 0.02, 0.02, 0.98));
    draw_rectangle_lines(panel_x, panel_y, panel_w, panel_h, 2.0, YELLOW);

    let mut py = panel_y + 30.0;
    draw_text(&format!("NEURAL PROFILE: AGENT #{}", a.id), panel_x + 20.0, py, 24.0, YELLOW);
    py += 40.0;

    draw_text("BEHAVIORAL SIMULATION (Situational Probing)", panel_x + 20.0, py, 18.0, GRAY); py += 30.0;

    let profile = ui_logic::calculate_behavioral_profile(a);
    for sit in profile {
        draw_text(sit.name, panel_x + 20.0, py, 16.0, WHITE);
        let bar_x = panel_x + 180.0;
        let props = [ 
            ("CBT", sit.combat, RED), 
            ("ALT", sit.altruism, GREEN), 
            ("IND", sit.industry, BLUE), 
            ("TRD", sit.trade, GOLD), 
            ("AGL", sit.agility, MAGENTA) 
        ];
        let mut bx = bar_x;
        for (label, val, color) in props {
            let h: f32 = val * 40.0;
            draw_rectangle(bx, py - h.max(2.0) + 15.0, 60.0, h.max(2.0), color);
            draw_text(label, bx + 2.0, py + 28.0, 10.0, GRAY);
            bx += 65.0;
        }
        py += 60.0;
    }

    py += 10.0;
    draw_text("BIOMETRICS:", panel_x + 20.0, py, 16.0, GRAY); py += 20.0;
    draw_text(&format!("Age: {} | Sex: {}", format_time(a.age as u64, tick_to_mins), if a.gender > 0.5 { "Male" } else { "Female" }), panel_x + 20.0, py, 16.0, WHITE); py += 20.0;
    draw_text(&format!("Health: {:.1} | Stamina: {:.1}", a.health, a.stamina), panel_x + 20.0, py, 16.0, WHITE); py += 20.0;
    draw_text(&format!("Wealth: ${:.2} | Food: {:.0}g", a.wealth, a.food), panel_x + 20.0, py, 16.0, WHITE); py += 35.0;

    let matrix_x = panel_x + 20.0;
    draw_text("NEURAL INFLUENCE MATRIX (W3)", matrix_x, py, 14.0, GRAY); py += 20.0;
    let cell_size = 7.0;
    for h in 0..a.hidden_count as usize {
        for o in 0..NUM_OUTPUTS {
            let weight = a.w3[h * NUM_OUTPUTS + o];
            let color = if weight > 0.0 { Color::new(0.0, (weight * 0.5).min(1.0), 1.0, 1.0) } else { Color::new(1.0, (weight.abs() * 0.5).min(1.0), 0.0, 1.0) };
            draw_rectangle(matrix_x + o as f32 * cell_size, py + h as f32 * cell_size, cell_size - 1.0, cell_size - 1.0, color);
        }
    }
    py += (a.hidden_count as f32 * cell_size) + 20.0;

    draw_text("INTERACTIVE SENSORY-BEHAVIORAL INFLUENCE (Linear Approximation)", matrix_x, py, 14.0, GRAY); py += 30.0;
    
    let mx = mouse_position().0;
    let my = mouse_position().1;
    let left_nodes_x = matrix_x + 120.0;
    let right_nodes_x = matrix_x + 400.0;
    let node_radius = 4.0;
    let influence = a.calculate_input_output_influence();
    
    let mut hover_node = None;
    let mut is_hover_output = false;

    let right_spacing = (panel_h - (py - panel_y) - 50.0) / (NUM_OUTPUTS as f32);
    for o in 0..NUM_OUTPUTS {
        let ny = py + o as f32 * right_spacing;
        let dist_sq = (mx - right_nodes_x) * (mx - right_nodes_x) + (my - ny) * (my - ny);
        let is_hover = dist_sq < (node_radius * 3.0) * (node_radius * 3.0);
        if is_hover { hover_node = Some(o); is_hover_output = true; }
        draw_circle(right_nodes_x, ny, node_radius, if is_hover { YELLOW } else { Color::new(0.3, 0.3, 0.3, 1.0) });
        draw_text(crate::agent::OUTPUT_LABELS[o], right_nodes_x + 15.0, ny + 5.0, 11.0, if is_hover { WHITE } else { GRAY });
    }

    let mut input_importance = vec![0.0f32; crate::agent::NUM_INPUTS];
    for i in 0..crate::agent::NUM_INPUTS {
        for o in 0..NUM_OUTPUTS { input_importance[i] += influence[i * NUM_OUTPUTS + o].abs(); }
    }
    let mut top_input_indices: Vec<usize> = (0..crate::agent::NUM_INPUTS).collect();
    top_input_indices.sort_by(|a, b| input_importance[*b].partial_cmp(&input_importance[*a]).unwrap());
    top_input_indices.truncate(30);
    top_input_indices.sort();

    let left_spacing = (panel_h - (py - panel_y) - 50.0) / (top_input_indices.len() as f32);
    for (idx, &i_idx) in top_input_indices.iter().enumerate() {
        let ny = py + idx as f32 * left_spacing;
        let dist_sq = (mx - left_nodes_x) * (mx - left_nodes_x) + (my - ny) * (my - ny);
        let is_hover = dist_sq < (node_radius * 3.0) * (node_radius * 3.0);
        if is_hover { hover_node = Some(i_idx); is_hover_output = false; }
        draw_circle(left_nodes_x, ny, node_radius, if is_hover { YELLOW } else { Color::new(0.3, 0.3, 0.3, 1.0) });
        draw_text(crate::agent::INPUT_LABELS[i_idx], left_nodes_x - 110.0, ny + 5.0, 11.0, if is_hover { WHITE } else { GRAY });
    }

    let connections = ui_logic::get_top_connections(&influence, crate::agent::NUM_INPUTS, NUM_OUTPUTS, hover_node, is_hover_output, if hover_node.is_some() { 20 } else { 40 });
    for conn in connections {
        if let Some(left_idx) = top_input_indices.iter().position(|&x| x == conn.from_idx) {
            let ly = py + left_idx as f32 * left_spacing;
            let ry = py + conn.to_idx as f32 * right_spacing;
            let alpha = (conn.weight.abs() * 2.0).min(1.0);
            let color = if conn.weight > 0.0 { Color::new(0.0, 0.8, 1.0, alpha) } else { Color::new(1.0, 0.4, 0.0, alpha) };
            draw_line(left_nodes_x, ly, right_nodes_x, ry, (conn.weight.abs() * 5.0).max(0.5), color);
            if hover_node.is_some() {
                let mid_x = (left_nodes_x + right_nodes_x) / 2.0;
                let mid_y = (ly + ry) / 2.0;
                draw_text(&format!("{:.2}", conn.weight), mid_x - 15.0, mid_y, 10.0, WHITE);
            }
        }
    }
}

pub fn draw_tracker(mx: f32, my: f32, left_clicked: bool, a: &Person, followed_id: &mut Option<u32>, show_inspector: &mut bool, tick_to_mins: f32) {
    let panel_w = 240.0;
    let panel_h = 200.0;
    let panel_x = 10.0;
    let panel_y = screen_height() - panel_h - 10.0;
    draw_rectangle(panel_x, panel_y, panel_w, panel_h, Color::new(0.0, 0.05, 0.05, 0.9));
    draw_rectangle_lines(panel_x, panel_y, panel_w, panel_h, 1.0, YELLOW);
    let mut py = panel_y + 25.0;
    let dy = 18.0;
    draw_text(&format!("TRACKING AGENT #{}", a.id), panel_x + 10.0, py, 16.0, YELLOW); py += dy + 5.0;
    draw_text(&format!("Age: {}", format_time(a.age as u64, tick_to_mins)), panel_x + 10.0, py, 14.0, WHITE); py += dy;
    draw_text(&format!("Health: {:.1}", a.health), panel_x + 10.0, py, 14.0, if a.health < 25.0 { RED } else { WHITE }); py += dy;
    draw_text(&format!("Wealth: ${:.0}", a.wealth), panel_x + 10.0, py, 14.0, GREEN); py += dy;
    draw_text(&format!("Food: {:.0}g", a.food), panel_x + 10.0, py, 14.0, WHITE); py += dy;
    draw_text(&format!("Water: {:.1}kg", a.water), panel_x + 10.0, py, 14.0, Color::new(0.4, 0.6, 1.0, 1.0)); py += dy;
    
    let stop_btn = Rect::new(panel_x + 10.0, py + 10.0, 100.0, 25.0);
    draw_rectangle(stop_btn.x, stop_btn.y, stop_btn.w, stop_btn.h, Color::new(0.3, 0.1, 0.1, 1.0));
    draw_text("STOP TRACK", stop_btn.x + 10.0, stop_btn.y + 18.0, 14.0, WHITE);
    if left_clicked && stop_btn.contains(vec2(mx, my)) { *followed_id = None; }

    let ins_btn = Rect::new(panel_x + 120.0, py + 10.0, 100.0, 25.0);
    draw_rectangle(ins_btn.x, ins_btn.y, ins_btn.w, ins_btn.h, Color::new(0.1, 0.3, 0.3, 1.0));
    draw_text("INSPECT", ins_btn.x + 20.0, ins_btn.y + 18.0, 14.0, WHITE);
    if left_clicked && ins_btn.contains(vec2(mx, my)) { *show_inspector = true; }
}

pub fn draw_generation_graph(times: &[u64], tick_to_mins: f32) {
    if times.is_empty() { return; }
    let panel_w = 400.0;
    let panel_h = 300.0;
    let panel_x = screen_width() / 2.0 - panel_w / 2.0;
    let panel_y = screen_height() / 2.0 - panel_h / 2.0;
    draw_rectangle(panel_x, panel_y, panel_w, panel_h, Color::new(0.05, 0.05, 0.05, 0.95));
    draw_rectangle_lines(panel_x, panel_y, panel_w, panel_h, 2.0, Color::new(0.0, 1.0, 0.8, 1.0));
    draw_text("Generation Survival Time", panel_x + 100.0, panel_y + 25.0, 20.0, WHITE);
    
    let max_ticks = (*times.iter().max().unwrap_or(&1)).max(1);
    let max_gen = (times.len() - 1).max(1);

    draw_line(panel_x + 40.0, panel_y + panel_h - 40.0, panel_x + panel_w - 40.0, panel_y + panel_h - 40.0, 1.0, GRAY);
    draw_line(panel_x + 40.0, panel_y + 40.0, panel_x + 40.0, panel_y + panel_h - 40.0, 1.0, GRAY);
    
    let (max_label, zero, r#gen) = ui_logic::get_graph_axes_labels(max_ticks, tick_to_mins);
    draw_text(&max_label, panel_x + 5.0, panel_y + 50.0, 14.0, GRAY);
    draw_text(&zero, panel_x + 25.0, panel_y + panel_h - 35.0, 14.0, GRAY);
    draw_text(&r#gen, panel_x + panel_w - 35.0, panel_y + panel_h - 25.0, 14.0, GRAY);

    let mut last_pts: Option<(f32, f32)> = None;
    for (i, &time) in times.iter().enumerate() {
        let px = panel_x + 40.0 + (i as f32 / max_gen as f32) * (panel_w - 80.0);
        let py = panel_y + panel_h - 40.0 - (time as f32 / max_ticks as f32) * (panel_h - 80.0);
        draw_circle(px, py, 3.0, YELLOW);
        if let Some(lp) = last_pts { draw_line(lp.0, lp.1, px, py, 2.0, Color::new(0.0, 1.0, 0.8, 1.0)); }
        last_pts = Some((px, py));
    }
}

pub fn draw_config_panel(
    mx: f32, my: f32, left_clicked: bool, _is_mouse_down: bool, _frame_time: f32,
    config: &mut crate::config::SimConfig,
    last_saved: &crate::config::SimConfig,
    scroll: &mut f32,
    search_query: &mut String,
    config_changed: &mut bool,
    pending_save: &mut bool,
    _active_button: &mut Option<String>,
    _hold_time: &mut f32,
) {
    let layout = ui_logic::calculate_config_layout(screen_width(), screen_height());
    draw_rectangle(layout.x, layout.y, layout.w, layout.h, Color::new(0.0, 0.05, 0.05, 0.98));
    draw_rectangle_lines(layout.x, layout.y, layout.w, layout.h, 2.0, Color::new(0.0, 1.0, 0.8, 1.0));

    let mut py = layout.y + 20.0;
    draw_text("CONFIGURATION", layout.x + 20.0, py, 24.0, Color::new(0.0, 1.0, 0.8, 1.0)); py += 35.0;

    draw_rectangle(layout.x + 20.0, py, layout.w - 40.0, 30.0, Color::new(0.1, 0.1, 0.1, 1.0));
    draw_rectangle_lines(layout.x + 20.0, py, layout.w - 40.0, 30.0, 1.0, GRAY);
    if search_query.is_empty() { draw_text("Search settings...", layout.x + 30.0, py + 20.0, 18.0, DARKGRAY); }
    else { draw_text(search_query, layout.x + 30.0, py + 20.0, 18.0, WHITE); }
    
    while let Some(c) = get_char_pressed() { if !c.is_control() { search_query.push(c); } }
    if is_key_pressed(KeyCode::Backspace) { search_query.pop(); }
    py += 45.0;

    let save_btn = Rect::new(layout.x + 20.0, py, 120.0, 35.0);
    let hover = save_btn.contains(vec2(mx, my));
    draw_rectangle(save_btn.x, save_btn.y, save_btn.w, save_btn.h, if hover { Color::new(0.0, 0.6, 0.5, 1.0) } else { Color::new(0.0, 0.4, 0.3, 1.0) });
    draw_text("SAVE JSON", save_btn.x + 15.0, save_btn.y + 24.0, 18.0, WHITE);
    if left_clicked && hover { *pending_save = true; }
    py += 50.0;

    let filtered = ui_logic::get_filtered_config_items(config, last_saved, search_query);
    let mut item_y = py + *scroll;
    for item in filtered {
        if item_y < py || item_y > layout.y + layout.h - 40.0 { item_y += 35.0; continue; }
        let text_color: Color = item.color.into();
        draw_text(&item.label, layout.x + 30.0, item_y + 20.0, 14.0, text_color);
        draw_text(&item.value_str, layout.x + 220.0, item_y + 20.0, 14.0, text_color);
        
        let inc_btn = Rect::new(layout.x + 330.0, item_y + 4.0, 30.0, 24.0);
        let dec_btn = Rect::new(layout.x + 300.0, item_y + 4.0, 30.0, 24.0);
        
        if left_clicked && inc_btn.contains(vec2(mx, my)) { update_val(config, &item.key, 1.0); *config_changed = true; }
        if left_clicked && dec_btn.contains(vec2(mx, my)) { update_val(config, &item.key, -1.0); *config_changed = true; }
        
        draw_rectangle(inc_btn.x, inc_btn.y, inc_btn.w, inc_btn.h, if inc_btn.contains(vec2(mx, my)) { GREEN } else { GRAY });
        draw_rectangle(dec_btn.x, dec_btn.y, dec_btn.w, dec_btn.h, if dec_btn.contains(vec2(mx, my)) { RED } else { GRAY });
        draw_text("+", inc_btn.x + 8.0, inc_btn.y + 18.0, 20.0, WHITE);
        draw_text("-", dec_btn.x + 10.0, dec_btn.y + 18.0, 20.0, WHITE);
        item_y += 35.0;
    }
}

fn update_val(c: &mut crate::config::SimConfig, key: &str, dir: f32) {
    match key {
        "world.regen_rate" => c.world.regen_rate = (c.world.regen_rate + 0.001 * dir).max(0.0),
        "world.max_tile_resource" => c.world.max_tile_resource = (c.world.max_tile_resource + 100.0 * dir).max(0.0),
        "world.max_tile_water" => c.world.max_tile_water = (c.world.max_tile_water + 100.0 * dir).max(0.0),
        "world.tick_to_mins" => c.world.tick_to_mins = (c.world.tick_to_mins + 1.0 * dir).max(0.1),
        
        "sim.agent_count" => c.sim.agent_count = (c.sim.agent_count as i32 + 100 * dir as i32).max(1) as u32,
        "sim.spawn_group_size" => c.sim.spawn_group_size = (c.sim.spawn_group_size as i32 + 10 * dir as i32).max(1) as u32,
        "sim.founder_count" => c.sim.founder_count = (c.sim.founder_count as i32 + 10 * dir as i32).max(1) as u32,
        "sim.load_saved_agents_on_start" => c.sim.load_saved_agents_on_start = (c.sim.load_saved_agents_on_start as i32 + dir as i32).clamp(0, 2) as u32,

        "bio.base_speed" => c.bio.base_speed = (c.bio.base_speed + 0.5 * dir).max(0.1),
        "bio.max_age" => c.bio.max_age = (c.bio.max_age + 10000.0 * dir).max(100.0),
        "bio.max_health" => c.bio.max_health = (c.bio.max_health + 10.0 * dir).max(1.0),
        "bio.max_stamina" => c.bio.max_stamina = (c.bio.max_stamina + 10.0 * dir).max(1.0),
        "bio.max_water" => c.bio.max_water = (c.bio.max_water + 10.0 * dir).max(1.0),
        "bio.puberty_age" => c.bio.puberty_age = (c.bio.puberty_age + 1000.0 * dir).max(0.0),
        "bio.gestation_period" => c.bio.gestation_period = (c.bio.gestation_period + 100.0 * dir).max(0.0),
        "bio.starvation_rate" => c.bio.starvation_rate = (c.bio.starvation_rate + 0.01 * dir).max(0.0),
        "bio.infant_speed_mult" => c.bio.infant_speed_mult = (c.bio.infant_speed_mult + 0.05 * dir).clamp(0.0, 1.0),
        "bio.infant_stamina_mult" => c.bio.infant_stamina_mult = (c.bio.infant_stamina_mult + 0.05 * dir).clamp(0.0, 1.0),

        "eco.baseline_cost" => c.eco.baseline_cost = (c.eco.baseline_cost + 0.01 * dir).max(0.0),
        "eco.reproduction_cost" => c.eco.reproduction_cost = (c.eco.reproduction_cost + 1.0 * dir).max(0.0),
        "eco.boat_cost" => c.eco.boat_cost = (c.eco.boat_cost + 10.0 * dir).max(0.0),
        "eco.water_transfer_amount" => c.eco.water_transfer_amount = (c.eco.water_transfer_amount + 0.1 * dir).max(0.0),
        "eco.base_spoilage_rate" => c.eco.base_spoilage_rate = (c.eco.base_spoilage_rate + 0.001 * dir).max(0.0),

        "genetics.mutation_rate" => c.genetics.mutation_rate = (c.genetics.mutation_rate + 0.01 * dir).clamp(0.0, 1.0),
        "genetics.mutation_strength" => c.genetics.mutation_strength = (c.genetics.mutation_strength + 0.01 * dir).max(0.0),
        "genetics.random_spawn_percentage" => c.genetics.random_spawn_percentage = (c.genetics.random_spawn_percentage + 0.05 * dir).clamp(0.0, 1.0),

        "infra.infra_cost" => c.infra.infra_cost = (c.infra.infra_cost + 5.0 * dir).max(0.0),
        "infra.road_speed_bonus" => c.infra.road_speed_bonus = (c.infra.road_speed_bonus + 0.1 * dir).max(1.0),
        "infra.housing_rest_bonus" => c.infra.housing_rest_bonus = (c.infra.housing_rest_bonus + 0.1 * dir).max(1.0),
        "infra.storage_rot_reduction" => c.infra.storage_rot_reduction = (c.infra.storage_rot_reduction + 0.05 * dir).clamp(0.0, 1.0),

        "combat.attacker_damage" => c.combat.attacker_damage = (c.combat.attacker_damage + 1.0 * dir).max(0.0),
        "combat.steal_amount" => c.combat.steal_amount = (c.combat.steal_amount + 1.0 * dir).max(0.0),
        _ => {}
    }
}
