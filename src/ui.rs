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
use crate::agent::{Person, NUM_OUTPUTS, NUM_INPUTS, INPUT_LABELS, OUTPUT_LABELS};
use crate::shared::{VisualMode, SortCol, format_time};
use crate::ui_logic;

pub fn draw_metrics(pop: usize, compute_time: f32, ticks_per_sec: f32, speed: usize, total_ticks: u64, tick_to_mins: f32, fps: i32, avg_fps: f32, low_1_fps: f32, visual_mode: VisualMode, show_inspector: bool, show_generation_graph: bool, show_config_panel: bool, paused: bool, restart_msg: bool) {
    let mut y = 20.0;
    let dy = 20.0;
    draw_rectangle(10.0, 10.0, 250.0, 310.0, Color::new(0.0, 0.0, 0.0, 0.7));
    draw_rectangle_lines(10.0, 10.0, 250.0, 310.0, 1.0, GRAY);
    
    draw_text(&format!("Population: {}", pop), 20.0, y + 15.0, 18.0, WHITE);
    y += dy + 10.0;
    draw_text(&format!("Compute: {:.2}ms/loop", compute_time), 20.0, y, 16.0, WHITE);
    y += dy;
    draw_text(&format!("Throughput: {:.0} ticks/s", ticks_per_sec), 20.0, y, 16.0, YELLOW);
    y += dy;
    draw_text(&format!("Speed: {} ticks/loop", speed), 20.0, y, 16.0, WHITE);
    y += dy;
    draw_text(&format!("Total Ticks: {}", total_ticks), 20.0, y, 16.0, WHITE);
    y += dy;
    draw_text(&format!("Time: {}", format_time(total_ticks, tick_to_mins)), 20.0, y, 16.0, WHITE);
    y += dy + 5.0;
    draw_text(&format!("FPS: {} (Avg: {:.0}, 1%: {:.0})", fps, avg_fps, low_1_fps), 20.0, y, 16.0, if fps < 30 { RED } else { GREEN });
    y += dy + 10.0;
    draw_text(&format!("Visual Mode [R]: {:?}", visual_mode), 20.0, y, 16.0, Color::new(0.0, 1.0, 0.8, 1.0));
    y += dy;
    draw_text(&format!("Inspector [TAB]: {}", if show_inspector { "OPEN" } else { "CLOSED" }), 20.0, y, 16.0, WHITE);
    y += dy;
    draw_text(&format!("Gen Graph [G]: {}", if show_generation_graph { "OPEN" } else { "CLOSED" }), 20.0, y, 16.0, WHITE);
    y += dy;
    draw_text(&format!("Config Panel [C]: {}", if show_config_panel { "OPEN" } else { "CLOSED" }), 20.0, y, 16.0, WHITE);
    y += dy;
    draw_text("Save Agents [S] | Full Save [F]", 20.0, y, 14.0, GRAY);

    if paused { draw_text("PAUSED", screen_width() / 2.0 - 50.0, 30.0, 30.0, RED); }
    if restart_msg {
        let text = "POPULATION CRITICAL, RESTARTING...";
        let text_dims = measure_text(text, None, 40, 1.0);
        draw_text(text, screen_width() / 2.0 - text_dims.width / 2.0, screen_height() / 2.0, 40.0, RED);
    }
}

pub fn draw_visuals_panel(mx: f32, my: f32, clicked: bool, current_mode: &mut VisualMode) {
    let panel_w = 200.0;
    let panel_h = 360.0;
    let panel_x = 280.0;
    let panel_y = 10.0;
    draw_rectangle(panel_x, panel_y, panel_w, panel_h, Color::new(0.0, 0.0, 0.0, 0.8));
    draw_rectangle_lines(panel_x, panel_y, panel_w, panel_h, 2.0, GRAY);
    let modes = [
        (VisualMode::Default, "Default Terrain"), (VisualMode::Resources, "Resources"), (VisualMode::Age, "Age Gradient"),
        (VisualMode::Gender, "Gender (M/F)"), (VisualMode::Pregnancy, "Pregnancy State"), (VisualMode::MarketWealth, "Market Wealth"),
        (VisualMode::MarketFood, "Market Food"), (VisualMode::AskPrice, "Avg Ask Price"), (VisualMode::BidPrice, "Avg Bid Price"),
        (VisualMode::Infrastructure, "Infrastructure"), (VisualMode::Temperature, "Temperature"), (VisualMode::DayNight, "Day/Night Cycle"),
        (VisualMode::Tribes, "Tribe Territory"), (VisualMode::Water, "Water Reserves"),
    ];
    let mut y = panel_y + 20.0;
    for (mode, label) in modes {
        let rect = Rect::new(panel_x + 10.0, y, panel_w - 20.0, 20.0);
        let hover = rect.contains(vec2(mx, my));
        let active = *current_mode == mode;
        draw_rectangle(rect.x, rect.y, rect.w, rect.h, if active { Color::new(0.0, 0.6, 0.5, 1.0) } else if hover { Color::new(0.2, 0.2, 0.2, 1.0) } else { BLANK });
        draw_text(label, rect.x + 5.0, rect.y + 15.0, 14.0, if active { WHITE } else { GRAY });
        if clicked && hover { *current_mode = mode; }
        y += 24.0;
    }
}

pub fn apply_sort(agents: &mut Vec<(usize, Person)>, col: SortCol, desc: bool) {
    agents.sort_by(|(_, a), (_, b)| {
        let cmp = match col {
            SortCol::Index => a.state.id.cmp(&b.state.id),
            SortCol::Age => a.state.age.partial_cmp(&b.state.age).unwrap(),
            SortCol::Health => a.state.health.partial_cmp(&b.state.health).unwrap(),
            SortCol::Food => a.state.food.partial_cmp(&b.state.food).unwrap(),
            SortCol::Wealth => a.state.wealth.partial_cmp(&b.state.wealth).unwrap(),
            SortCol::Gender => a.state.gender.partial_cmp(&b.state.gender).unwrap(),
            SortCol::Speed => a.state.speed.partial_cmp(&b.state.speed).unwrap(),
            SortCol::Heading => a.state.heading.partial_cmp(&b.state.heading).unwrap(),
            _ => std::cmp::Ordering::Equal,
        };
        if desc { cmp.reverse() } else { cmp }
    });
}

pub fn draw_influence_map(a: &Person, x: f32, y: f32, w: f32, h: f32, limit: usize) {
    draw_text("SENSORY-BEHAVIORAL INFLUENCE MAP", x, y - 10.0, 16.0, WHITE);
    draw_rectangle(x, y, w, h, Color::new(0.0, 0.05, 0.05, 0.3));

    let influence = a.calculate_input_output_influence();
    let top_connections = ui_logic::get_top_connections(&influence, NUM_INPUTS, NUM_OUTPUTS, None, false, limit);

    for conn in top_connections {
        let in_y = y + (conn.from_idx as f32 / NUM_INPUTS as f32) * h;
        let out_y = y + (conn.to_idx as f32 / NUM_OUTPUTS as f32) * h;
        let alpha = (conn.weight.abs() * 2.5).clamp(0.1, 1.0);
        let color = if conn.weight > 0.0 { Color::new(0.0, 1.0, 1.0, alpha) } else { Color::new(1.0, 0.0, 1.0, alpha) };
        draw_line(x, in_y, x + w, out_y, 1.5, color);
        
        draw_text(INPUT_LABELS[conn.from_idx], x - 100.0, in_y + 4.0, 10.0, if alpha > 0.6 { WHITE } else { GRAY });
        draw_text(OUTPUT_LABELS[conn.to_idx], x + w + 5.0, out_y + 4.0, 10.0, if alpha > 0.6 { WHITE } else { GRAY });
    }
}

pub fn draw_agent_profile_panel(a: &Person, tick_to_mins: f32) {
    let panel_w = 550.0;
    let panel_h = screen_height() - 40.0;
    let panel_x = screen_width() - panel_w - 20.0;
    let panel_y = 20.0;

    draw_rectangle(panel_x, panel_y, panel_w, panel_h, Color::new(0.0, 0.02, 0.02, 0.98));
    draw_rectangle_lines(panel_x, panel_y, panel_w, panel_h, 2.0, YELLOW);

    let mut py = panel_y + 30.0;
    draw_text(&format!("NEURAL PROFILE: AGENT #{}", a.state.id), panel_x + 20.0, py, 24.0, YELLOW);
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
            ("TRD", sit.trade, Color::new(1.0, 0.8, 0.0, 1.0)), 
            ("AGL", sit.agility, MAGENTA) 
        ];
        let mut bx = bar_x;
        for (label, val, color) in props {
            let h: f32 = (val * 40.0).clamp(2.0, 40.0);
            draw_rectangle(bx, py - h + 15.0, 60.0, h, color);
            draw_text(label, bx + 2.0, py + 28.0, 10.0, GRAY);
            bx += 65.0;
        }
        py += 60.0;
    }

    py += 10.0;
    draw_text("BIOMETRICS:", panel_x + 20.0, py, 16.0, GRAY); py += 20.0;
    draw_text(&format!("Age: {} | Sex: {}", format_time(a.state.age as u64, tick_to_mins), if a.state.gender > 0.5 { "Male" } else { "Female" }), panel_x + 20.0, py, 16.0, WHITE); py += 20.0;
    draw_text(&format!("Health: {:.1} | Stamina: {:.1}", a.state.health, a.state.stamina), panel_x + 20.0, py, 16.0, WHITE); py += 20.0;
    draw_text(&format!("Wealth: ${:.2} | Food: {:.0}g", a.state.wealth, a.state.food), panel_x + 20.0, py, 16.0, WHITE); py += 35.0;

    draw_text("WORKING MEMORY:", panel_x + 20.0, py, 14.0, GRAY); py += 15.0;
    let mem_cell_w = 20.0;
    let mem_cell_h = 10.0;
    for m in 0..24 {
        let val = a.state.mems[m];
        let color = if val > 0.0 { Color::new(0.0, 0.8, 1.0, (val * 0.8 + 0.2).min(1.0)) } else { Color::new(1.0, 0.4, 0.0, (val.abs() * 0.8 + 0.2).min(1.0)) };
        draw_rectangle(panel_x + 20.0 + m as f32 * (mem_cell_w + 2.0), py, mem_cell_w, mem_cell_h, color);
    }
    py += 25.0;

    draw_text("VOCAL SIGNALING:", panel_x + 20.0, py, 14.0, GRAY); py += 15.0;
    let comm_cell_w = 42.0;
    for c in 0..12 {
        let val = a.state.comms[c];
        let color = if val > 0.0 { Color::new(0.0, 1.0, 0.5, (val * 0.8 + 0.2).min(1.0)) } else { Color::new(1.0, 0.0, 0.2, (val.abs() * 0.8 + 0.2).min(1.0)) };
        draw_rectangle(panel_x + 20.0 + c as f32 * (comm_cell_w + 2.0), py, comm_cell_w, mem_cell_h, color);
    }
    py += 25.0;

    // --- New: Identity Sensing Display ---
    draw_text("IDENTITY SIGNATURE:", panel_x + 20.0, py, 14.0, GRAY); py += 15.0;
    let feat_w = 40.0;
    for i in 0..4 {
        let val = match i { 0 => a.state.id_f1, 1 => a.state.id_f2, 2 => a.state.id_f3, _ => a.state.id_f4 };
        let color = if val > 0.0 { Color::new(0.0, 0.8, 1.0, val.abs()) } else { Color::new(1.0, 0.4, 0.0, val.abs()) };
        draw_rectangle(panel_x + 20.0 + i as f32 * (feat_w + 5.0), py, feat_w, 10.0, color);
        draw_text(&format!("C{}", i+1), panel_x + 20.0 + i as f32 * (feat_w + 5.0), py + 22.0, 10.0, GRAY);
    }
    py += 35.0;

    // --- New: Spatial CNN & Plasticity Display ---
    draw_text("SPATIAL TOPOLOGY (CNN Features):", panel_x + 20.0, py, 14.0, GRAY); py += 15.0;
    for i in 0..8 {
        let val = a.state.spatial_features[i];
        let color = if val > 0.0 { Color::new(0.0, 1.0, 0.2, val.abs().min(1.0)) } else { Color::new(1.0, 0.0, 0.5, val.abs().min(1.0)) };
        draw_rectangle(panel_x + 20.0 + i as f32 * (feat_w / 1.5 + 5.0), py, feat_w / 1.5, 10.0, color);
        draw_text(&format!("S{}", i+1), panel_x + 20.0 + i as f32 * (feat_w / 1.5 + 5.0), py + 22.0, 10.0, GRAY);
    }
    py += 35.0;

    draw_text("IDENTITY SENSING (Target neighbor):", panel_x + 20.0, py, 14.0, GRAY); py += 15.0;
    if a.state.nearest_id_f1 != 0.0 || a.state.nearest_id_f2 != 0.0 {
        for i in 0..4 {
            let val = match i { 0 => a.state.nearest_id_f1, 1 => a.state.nearest_id_f2, 2 => a.state.nearest_id_f3, _ => a.state.nearest_id_f4 };
            let color = if val > 0.0 { Color::new(0.0, 1.0, 0.5, val.abs()) } else { Color::new(1.0, 0.0, 0.2, val.abs()) };
            draw_rectangle(panel_x + 20.0 + i as f32 * (feat_w + 5.0), py, feat_w, 10.0, color);
        }
        draw_text("MULTI-VARIABLE SIGNATURE DETECTED", panel_x + 20.0, py + 22.0, 12.0, WHITE);
    } else {
        draw_text("NO TARGET IN RANGE", panel_x + 20.0, py + 18.0, 14.0, DARKGRAY);
    }
    py += 45.0;

    draw_text("PERSONAL PLASTICITY (Hebbian Memory):", panel_x + 20.0, py, 14.0, GRAY); py += 15.0;
    let pw_w = (panel_w - 60.0) / 32.0;
    for i in 0..32 {
        let val = a.state.plastic_weights[i];
        let color = if val > 0.0 { Color::new(0.0, 0.8, 1.0, val.abs()) } else { Color::new(1.0, 0.5, 0.0, val.abs()) };
        draw_rectangle(panel_x + 20.0 + i as f32 * pw_w, py, pw_w - 1.0, 8.0, color);
    }
    py += 25.0;

    draw_influence_map(a, panel_x + 120.0, py + 20.0, 300.0, panel_h - (py - panel_y) - 60.0, 40);
}

pub fn draw_inspector(mx: f32, my: f32, left_clicked: bool, wheel: f32, agents: &mut Vec<(usize, Person)>, sort_col: &mut SortCol, sort_desc: &mut bool, scroll: &mut usize, selected: &mut Option<Person>, followed_id: &mut Option<u32>, show: &mut bool, _tick_to_mins: f32, is_live_mode: &mut bool) {
    let layout = ui_logic::calculate_inspector_layout(screen_width(), screen_height());
    draw_rectangle(layout.x, layout.y, layout.w, layout.h, Color::new(0.0, 0.05, 0.05, 0.95));
    draw_rectangle_lines(layout.x, layout.y, layout.w, layout.h, 2.0, Color::new(0.0, 1.0, 0.8, 1.0));
    draw_text("AGENT POPULATION INSPECTOR", layout.x + 20.0, layout.y + 35.0, 24.0, Color::new(0.0, 1.0, 0.8, 1.0));
    if is_key_pressed(KeyCode::Escape) || (left_clicked && mx < layout.x) { *show = false; }
    let col_w = layout.w / 8.0;
    let headers = [(SortCol::Index, "ID"), (SortCol::Age, "Age"), (SortCol::Health, "HP"), (SortCol::Food, "Food"), (SortCol::Wealth, "Wealth"), (SortCol::Gender, "G"), (SortCol::Speed, "Spd"), (SortCol::Heading, "Hdg")];
    let mut hx = layout.x + 10.0;
    for (col, label) in headers {
        let rect = Rect::new(hx, layout.y + 60.0, col_w - 5.0, 25.0);
        let hover = rect.contains(vec2(mx, my));
        draw_rectangle(rect.x, rect.y, rect.w, rect.h, if *sort_col == col { Color::new(0.0, 0.4, 0.3, 1.0) } else if hover { Color::new(0.1, 0.1, 0.1, 1.0) } else { BLACK });
        draw_text(label, rect.x + 5.0, rect.y + 18.0, 14.0, WHITE);
        if left_clicked && hover { if *sort_col == col { *sort_desc = !*sort_desc; } else { *sort_col = col; *sort_desc = true; } apply_sort(agents, *sort_col, *sort_desc); }
        hx += col_w;
    }
    let list_y = layout.y + 90.0;
    let row_h = 22.0;
    let visible_rows = ((layout.h - 550.0) / row_h) as usize; // Reduced to make space for Influence Map
    *scroll = (*scroll as f32 - wheel * 3.0).max(0.0) as usize;
    *scroll = (*scroll).min(agents.len().saturating_sub(visible_rows));
    for i in 0..visible_rows {
        let idx = *scroll + i;
        if idx >= agents.len() { break; }
        let (_, a) = &agents[idx];
        let ry = list_y + i as f32 * row_h;
        let rect = Rect::new(layout.x + 10.0, ry, layout.w - 20.0, row_h - 2.0);
        let hover = rect.contains(vec2(mx, my));
        let is_sel = selected.as_ref().map_or(false, |s| s.state.id == a.state.id);
        draw_rectangle(rect.x, rect.y, rect.w, rect.h, if is_sel { Color::new(0.0, 0.6, 0.5, 0.8) } else if hover { Color::new(0.1, 0.1, 0.1, 1.0) } else { Color::new(0.0, 0.02, 0.02, 1.0) });
        let mut cx = layout.x + 15.0;
        draw_text(&format!("{}", a.state.id), cx, ry + 15.0, 14.0, WHITE); cx += col_w;
        draw_text(&format!("{:.1}", a.state.age / 144.0), cx, ry + 15.0, 14.0, WHITE); cx += col_w;
        draw_text(&format!("{:.0}", a.state.health), cx, ry + 15.0, 14.0, if a.state.health < 20.0 { RED } else { WHITE }); cx += col_w;
        draw_text(&format!("{:.0}", a.state.food/1000.0), cx, ry + 15.0, 14.0, WHITE); cx += col_w;
        draw_text(&format!("{:.0}", a.state.wealth), cx, ry + 15.0, 14.0, GREEN); cx += col_w;
        draw_text(if a.state.gender > 0.5 { "M" } else { "F" }, cx, ry + 15.0, 14.0, if a.state.gender > 0.5 { BLUE } else { PINK }); cx += col_w;
        draw_text(&format!("{:.1}", a.state.speed), cx, ry + 15.0, 14.0, WHITE); cx += col_w;
        draw_text(&format!("{:.1}", a.state.heading), cx, ry + 15.0, 14.0, WHITE);
        if left_clicked && hover { *selected = Some(*a); }
    }
    if let &mut Some(ref a) = selected {
        let details_x = layout.x + 20.0;
        let dy = layout.y + layout.h - 440.0;
        draw_rectangle(layout.x + 10.0, dy - 10.0, layout.w - 20.0, 440.0, Color::new(0.0, 0.1, 0.1, 1.0));
        draw_text(&format!("SELECTED: AGENT #{} (Genetics: #{})", a.state.id, a.state.genetics_index), details_x, dy + 20.0, 18.0, YELLOW);
        
        let by = dy + 40.0;
        let locate_btn = Rect::new(details_x, by, 120.0, 30.0);
        let l_hover = locate_btn.contains(vec2(mx, my));
        draw_rectangle(locate_btn.x, locate_btn.y, locate_btn.w, locate_btn.h, if l_hover { Color::new(0.0, 0.8, 0.7, 1.0) } else { Color::new(0.0, 0.4, 0.3, 1.0) });
        draw_text("FOLLOW", locate_btn.x + 25.0, locate_btn.y + 20.0, 16.0, WHITE);
        if left_clicked && l_hover { *followed_id = Some(a.state.id); }

        let live_btn = Rect::new(details_x + 130.0, by, 120.0, 30.0);
        let live_hover = live_btn.contains(vec2(mx, my));
        draw_rectangle(live_btn.x, live_btn.y, live_btn.w, live_btn.h, if live_hover { Color::new(0.8, 0.3, 0.0, 1.0) } else { Color::new(0.5, 0.1, 0.0, 1.0) });
        draw_text("LIVE POV", live_btn.x + 25.0, live_btn.y + 20.0, 16.0, WHITE);
        if left_clicked && live_hover { 
            *followed_id = Some(a.state.id); 
            *show = false; // Close inspector
            if !*is_live_mode {
                *is_live_mode = true;
            }
        }
    }
}

pub fn draw_tracker(mx: f32, my: f32, left_clicked: bool, a: &Person, followed_id: &mut Option<u32>, show_inspector: &mut bool, tick_to_mins: f32, is_live_mode: &mut bool) {
    let panel_w = 260.0;
    let panel_h = 240.0;
    let panel_x = screen_width() - panel_w - 10.0;
    let panel_y = 330.0; // Anchored top-right below tactical minimap
    draw_rectangle(panel_x, panel_y, panel_w, panel_h, Color::new(0.0, 0.05, 0.05, 0.9));
    draw_rectangle_lines(panel_x, panel_y, panel_w, panel_h, 1.0, YELLOW);
    let mut py = panel_y + 25.0;
    let dy = 18.0;
    draw_text(&format!("TRACKING AGENT #{}", a.state.id), panel_x + 10.0, py, 16.0, YELLOW); py += dy + 5.0;
    draw_text(&format!("Age: {}", format_time(a.state.age as u64, tick_to_mins)), panel_x + 10.0, py, 14.0, WHITE); py += dy;
    draw_text(&format!("Health: {:.1}", a.state.health), panel_x + 10.0, py, 14.0, if a.state.health < 25.0 { RED } else { WHITE }); py += dy;
    draw_text(&format!("Wealth: ${:.0}", a.state.wealth), panel_x + 10.0, py, 14.0, GREEN); py += dy;
    draw_text(&format!("Food: {:.0}g", a.state.food), panel_x + 10.0, py, 14.0, WHITE); py += dy;
    draw_text(&format!("Water: {:.1}kg", a.state.water), panel_x + 10.0, py, 14.0, Color::new(0.4, 0.6, 1.0, 1.0)); py += dy;
    
    let stop_btn = Rect::new(panel_x + 10.0, py + 10.0, 110.0, 25.0);
    draw_rectangle(stop_btn.x, stop_btn.y, stop_btn.w, stop_btn.h, Color::new(0.3, 0.1, 0.1, 1.0));
    draw_text("STOP TRACK", stop_btn.x + 12.0, stop_btn.y + 18.0, 14.0, WHITE);
    if left_clicked && stop_btn.contains(vec2(mx, my)) { *followed_id = None; *is_live_mode = false; }

    let ins_btn = Rect::new(panel_x + 130.0, py + 10.0, 110.0, 25.0);
    draw_rectangle(ins_btn.x, ins_btn.y, ins_btn.w, ins_btn.h, Color::new(0.1, 0.3, 0.3, 1.0));
    draw_text("INSPECT", ins_btn.x + 25.0, ins_btn.y + 18.0, 14.0, WHITE);
    if left_clicked && ins_btn.contains(vec2(mx, my)) { *show_inspector = true; }

    let live_btn = Rect::new(panel_x + 10.0, py + 40.0, 230.0, 30.0);
    draw_rectangle(live_btn.x, live_btn.y, live_btn.w, live_btn.h, Color::new(0.8, 0.2, 0.0, 1.0));
    draw_text(if *is_live_mode { "EXIT LIVE POV [L]" } else { "ENTER LIVE POV [L]" }, live_btn.x + 50.0, live_btn.y + 20.0, 14.0, WHITE);
    if left_clicked && live_btn.contains(vec2(mx, my)) { 
        if !*is_live_mode { *show_inspector = false; } // Close inspector on entry
        *is_live_mode = !*is_live_mode;
    }
}

pub fn draw_quit_confirmation(mx: f32, my: f32, left_clicked: bool, is_quitting: &mut bool, should_exit: &mut bool) {
    let w = 400.0; let h = 150.0;
    let x = screen_width() / 2.0 - w / 2.0;
    let y = screen_height() / 2.0 - h / 2.0;
    draw_rectangle(x, y, w, h, Color::new(0.05, 0.05, 0.05, 0.95));
    draw_rectangle_lines(x, y, w, h, 2.0, RED);
    
    draw_text("EXIT SIMULATION?", x + 100.0, y + 40.0, 24.0, WHITE);
    draw_text("Any unsaved progress will be lost.", x + 85.0, y + 65.0, 14.0, GRAY);

    let btn_w = 120.0; let btn_h = 40.0;
    let yes_btn = Rect::new(x + 60.0, y + 90.0, btn_w, btn_h);
    let no_btn = Rect::new(x + 220.0, y + 90.0, btn_w, btn_h);

    let y_h = yes_btn.contains(vec2(mx, my));
    let n_h = no_btn.contains(vec2(mx, my));

    draw_rectangle(yes_btn.x, yes_btn.y, yes_btn.w, yes_btn.h, if y_h { RED } else { Color::new(0.3, 0.0, 0.0, 1.0) });
    draw_rectangle(no_btn.x, no_btn.y, no_btn.w, no_btn.h, if n_h { GRAY } else { DARKGRAY });

    draw_text("YES (Q)", yes_btn.x + 35.0, yes_btn.y + 25.0, 18.0, WHITE);
    draw_text("CANCEL", no_btn.x + 28.0, no_btn.y + 25.0, 18.0, WHITE);

    if left_clicked {
        if y_h { *should_exit = true; }
        if n_h { *is_quitting = false; }
    }
}

pub fn draw_live_pov(a: &Person, sim: &crate::simulation::SimulationManager, config: &crate::config::SimConfig) {
    let panel_w = screen_width() * 0.98;
    let panel_h = 600.0;
    let panel_x = screen_width() / 2.0 - panel_w / 2.0;
    let panel_y = screen_height() / 2.0 - panel_h / 2.0;

    draw_rectangle(panel_x, panel_y, panel_w, panel_h, Color::new(0.0, 0.01, 0.02, 0.99));
    draw_rectangle_lines(panel_x, panel_y, panel_w, panel_h, 2.0, RED);
    
    draw_text(&format!("RESEARCH CONSOLE: AGENT #{} (LINEAGE #{})", a.state.id, a.state.genetics_index), panel_x + 20.0, panel_y + 35.0, 26.0, RED);
    draw_text("Throttled: 8 TPS | Mode: COGNITIVE PIPELINE | [L] EXIT", panel_x + panel_w - 450.0, panel_y + 30.0, 16.0, YELLOW);

    // --- 1. Sensory Perception ---
    let grid_size = 210.0; let cell_size = grid_size / 5.0;
    let grid_x = panel_x + 40.0; let grid_y = panel_y + 100.0;
    draw_text("SENSORY GRID (5x5 FOV)", grid_x, grid_y - 20.0, 18.0, WHITE);
    
    let cos_h = a.state.heading.cos(); let sin_h = a.state.heading.sin();
    let agent_lat = ((a.state.y - config.world.map_height as f32 / 2.0).abs() / (config.world.map_height as f32 / 2.0)) * 1.570796;
    let lon_scale = 1.0 / (agent_lat.cos().max(0.15));

    for ly_screen in 0..5 {
        let ly_fwd = 3.0 - ly_screen as f32; // 3 ahead to 1 behind
        for lx_screen in 0..5 {
            let lx_lat = lx_screen as f32 - 2.0; // -2 left to 2 right
            
            let fwd = ly_fwd * 8.0;
            let lat = lx_lat * 8.0;
            
            let rot_x = (fwd * cos_h - lat * sin_h) * lon_scale;
            let rot_y = fwd * sin_h + lat * cos_h;

            let wx = (a.state.x + rot_x).rem_euclid(config.world.map_width as f32);
            let wy = (a.state.y + rot_y).clamp(0.0, config.world.map_height as f32 - 1.0);

            let cell = &sim.env.map_cells[(wy as usize) * config.world.map_width as usize + (wx as usize)];
            let cx = grid_x + lx_screen as f32 * cell_size; let cy = grid_y + ly_screen as f32 * cell_size;
            draw_rectangle(cx, cy, cell_size, cell_size, Color::new(cell.pheno_r*0.5+0.5, cell.pheno_g*0.5+0.5, cell.pheno_b*0.5+0.5, 1.0));
            draw_rectangle_lines(cx, cy, cell_size, cell_size, 1.0, Color::new(1.0, 1.0, 1.0, 0.1));
            let res = cell.res_value as f32 / 1000.0;
            if res > 5.0 { draw_circle(cx + cell_size/2.0, cy + cell_size/2.0, (res / config.world.max_tile_resource).sqrt() * (cell_size/2.2), GREEN); }
            if lx_lat == 0.0 && ly_fwd == 0.0 { draw_rectangle_lines(cx + 2.0, cy + 2.0, cell_size - 4.0, cell_size - 4.0, 2.0, YELLOW); }
        }
    }
    
    let mut leg_y = grid_y + grid_size + 30.0;
    draw_text("LEGEND:", grid_x, leg_y, 14.0, GRAY); leg_y += 20.0;
    draw_circle(grid_x + 7.0, leg_y - 5.0, 5.0, GREEN); draw_text("Detected Food", grid_x + 20.0, leg_y, 14.0, WHITE); leg_y += 20.0;
    draw_rectangle(grid_x + 2.0, leg_y - 10.0, 10.0, 10.0, Color::new(0.5, 0.2, 0.8, 1.0)); draw_text("Territory", grid_x + 20.0, leg_y, 14.0, WHITE);

    // --- 2. Sensory-Behavioral Influence Map ---
    let bip_x = grid_x + grid_size + 120.0;
    let bip_y = panel_y + 100.0;
    let bip_w = 420.0;
    let bip_h = panel_h - 160.0;
    draw_influence_map(a, bip_x, bip_y, bip_w, bip_h, 40);

    // --- 3. Hidden-Output Synaptic Matrix ---
    let mat_x = bip_x + bip_w + 140.0;
    let mat_y = panel_y + 100.0;
    let mat_w = panel_w - (mat_x - panel_x) - 180.0;
    let mat_h = panel_h - 160.0;
    draw_text("HIDDEN -> BEHAVIOR", mat_x, mat_y - 20.0, 18.0, WHITE);
    draw_rectangle(mat_x, mat_y, mat_w, mat_h, BLACK);

    let rows = 64; let cols = NUM_OUTPUTS;
    let cw = mat_w / cols as f32; let ch = mat_h / rows as f32;
    let mock_inputs = [0.5f32; NUM_INPUTS];
    let outputs = a.mental_simulation(&mock_inputs);

    for r in 0..rows {
        for c in 0..cols {
            let weight = a.genetics.w3[r * cols + c];
            let firing = outputs[c].abs();
            let alpha = (weight.abs() * 0.9).clamp(0.1, 1.0);
            let mut color = if weight > 0.0 { Color::new(0.0, 1.0, 1.0, alpha) } else { Color::new(1.0, 0.0, 1.0, alpha) };
            if firing > 0.4 { color.r = (color.r + 0.6).min(1.0); color.g = (color.g + 0.6).min(1.0); color.b = (color.b + 0.6).min(1.0); }
            draw_rectangle(mat_x + c as f32 * cw, mat_y + r as f32 * ch, cw - 0.5, ch - 0.5, color);
        }
    }
    
    let cats = [("SENSORS", 0.1), ("BIOLOGY", 0.3), ("SOCIAL", 0.5), ("ECON", 0.7), ("MEMORY", 0.9)];
    for (name, p) in cats { draw_text(name, mat_x - 60.0, mat_y + mat_h * p, 10.0, GRAY); }

    // --- 4. Intent Bars ---
    let bars_x = mat_x + mat_w + 30.0; let mut bars_y = mat_y;
    let intents = [("SPD", a.state.speed / config.bio.base_speed), ("AGGR", a.state.attack_intent), ("REPRO", a.state.reproduce_desire), ("ASK", (a.state.ask_price.ln() / 10.0).clamp(0.0, 1.0)), ("BID", (a.state.bid_price.ln() / 10.0).clamp(0.0, 1.0))];
    for (label, val) in intents {
        draw_text(label, bars_x, bars_y + 12.0, 12.0, WHITE);
        draw_rectangle(bars_x + 40.0, bars_y, 100.0, 12.0, BLACK);
        draw_rectangle(bars_x + 40.0, bars_y, 100.0 * val.clamp(0.0, 1.0), 12.0, if val > 0.5 { RED } else { Color::new(0.0, 1.0, 0.8, 1.0) });
        bars_y += 24.0;
    }
}

pub fn draw_generation_graph(times: &[u64], tick_to_mins: f32, mx: f32, my: f32, wheel: f32, scroll_x: &mut f32, zoom: &mut f32, avg_period: &mut usize) -> bool {
    if times.is_empty() { return false; }
    let panel_w = screen_width() * 0.8;
    let panel_h = 380.0;
    let panel_x = screen_width() / 2.0 - panel_w / 2.0;
    let panel_y = screen_height() / 2.0 - panel_h / 2.0;
    let panel_rect = Rect::new(panel_x, panel_y, panel_w, panel_h);
    let mouse_on_panel = panel_rect.contains(vec2(mx, my));
    let prev_mx = mx_prev();
    let delta_x = mx - prev_mx;

    draw_rectangle(panel_x, panel_y, panel_w, panel_h, Color::new(0.05, 0.05, 0.05, 0.95));
    draw_rectangle_lines(panel_x, panel_y, panel_w, panel_h, 2.0, Color::new(0.0, 1.0, 0.8, 1.0));
    draw_text("Evolutionary Timeline: Survival per Generation", panel_x + 20.0, panel_y + 25.0, 20.0, WHITE);
    draw_text(&format!("AVG PERIOD: {}", avg_period), panel_x + panel_w - 180.0, panel_y + 25.0, 16.0, YELLOW);
    let btn_minus = Rect::new(panel_x + panel_w - 60.0, panel_y + 10.0, 20.0, 20.0);
    let btn_plus = Rect::new(panel_x + panel_w - 35.0, panel_y + 10.0, 20.0, 20.0);
    draw_rectangle(btn_minus.x, btn_minus.y, btn_minus.w, btn_minus.h, if btn_minus.contains(vec2(mx, my)) { GRAY } else { DARKGRAY });
    draw_rectangle(btn_plus.x, btn_plus.y, btn_plus.w, btn_plus.h, if btn_plus.contains(vec2(mx, my)) { GRAY } else { DARKGRAY });
    draw_text("-", btn_minus.x + 6.0, btn_minus.y + 15.0, 18.0, WHITE);
    draw_text("+", btn_plus.x + 5.0, btn_plus.y + 15.0, 18.0, WHITE);
    if is_mouse_button_pressed(MouseButton::Left) { if btn_minus.contains(vec2(mx, my)) { *avg_period = avg_period.saturating_sub(5).max(1); } if btn_plus.contains(vec2(mx, my)) { *avg_period = (*avg_period + 5).min(200); } }
    draw_text("Scroll: Drag | Mouse Wheel: Zoom | Hover: Inspect", panel_x + 20.0, panel_y + panel_h - 10.0, 14.0, GRAY);
    let graph_rect = Rect::new(panel_x + 50.0, panel_y + 40.0, panel_w - 70.0, panel_h - 110.0);
    draw_rectangle(graph_rect.x, graph_rect.y, graph_rect.w, graph_rect.h, Color::new(0.0, 0.02, 0.02, 1.0));
    if mouse_on_panel { if wheel > 0.0 { *zoom *= 1.1; } else if wheel < 0.0 { *zoom *= 0.9; } if is_mouse_button_down(MouseButton::Left) && !btn_minus.contains(vec2(mx, my)) && !btn_plus.contains(vec2(mx, my)) { *scroll_x += delta_x; } }
    *zoom = zoom.clamp(1.0, 100.0);
    let max_ticks = (*times.iter().max().unwrap_or(&1)).max(1);
    let view_width = graph_rect.w * *zoom;
    let max_scroll = (view_width - graph_rect.w).max(0.0);
    *scroll_x = scroll_x.clamp(-max_scroll, 0.0);
    let sb_rect = Rect::new(graph_rect.x, graph_rect.y + graph_rect.h + 5.0, graph_rect.w, 10.0);
    draw_rectangle(sb_rect.x, sb_rect.y, sb_rect.w, sb_rect.h, BLACK);
    if max_scroll > 0.0 { let handle_w = (graph_rect.w / view_width) * sb_rect.w; let handle_x = sb_rect.x + ((-*scroll_x) / max_scroll) * (sb_rect.w - handle_w); draw_rectangle(handle_x, sb_rect.y, handle_w, sb_rect.h, GRAY); }
    let mut last_pts: Option<(f32, f32)> = None; let mut last_avg_pts: Option<(f32, f32)> = None; let mut hovered_data: Option<(usize, u64, f32, f32)> = None;
    for (i, &time) in times.iter().enumerate() {
        let x_norm = i as f32 / (times.len() as f32).max(1.0);
        let px = graph_rect.x + x_norm * view_width + *scroll_x;
        let py = graph_rect.y + graph_rect.h - (time as f32 / max_ticks as f32) * graph_rect.h;
        let start_idx = i.saturating_sub(*avg_period - 1);
        let avg_time: f64 = times[start_idx..=i].iter().map(|&t| t as f64).sum::<f64>() / (i - start_idx + 1) as f64;
        let py_avg = graph_rect.y + graph_rect.h - (avg_time as f32 / max_ticks as f32) * graph_rect.h;
        if px >= graph_rect.x && px <= graph_rect.x + graph_rect.w {
            if (mx - px).abs() < (5.0 * *zoom).min(10.0) && my >= graph_rect.y && my <= graph_rect.y + graph_rect.h { hovered_data = Some((i, time, px, py)); }
            draw_circle(px, py, 2.0, if hovered_data.as_ref().map_or(false, |h| h.0 == i) { YELLOW } else { Color::new(0.0, 0.8, 0.6, 0.4) });
            if let Some(lp) = last_pts { if lp.0 >= graph_rect.x && lp.0 <= graph_rect.x + graph_rect.w { draw_line(lp.0, lp.1, px, py, 1.0, Color::new(0.0, 1.0, 0.8, 0.3)); } }
            if let Some(la) = last_avg_pts { if la.0 >= graph_rect.x && la.0 <= graph_rect.x + graph_rect.w { draw_line(la.0, la.1, px, py_avg, 2.5, Color::new(1.0, 0.5, 0.0, 0.9)); } }
        }
        last_pts = Some((px, py)); last_avg_pts = Some((px, py_avg));
    }
    draw_line(graph_rect.x, graph_rect.y, graph_rect.x, graph_rect.y + graph_rect.h, 2.0, GRAY);
    draw_line(graph_rect.x, graph_rect.y + graph_rect.h, graph_rect.x + graph_rect.w, graph_rect.y + graph_rect.h, 2.0, GRAY);
    let (max_label, _, _) = ui_logic::get_graph_axes_labels(max_ticks, tick_to_mins);
    draw_text(&max_label, graph_rect.x - 45.0, graph_rect.y + 10.0, 14.0, GRAY);
    draw_text("0", graph_rect.x - 15.0, graph_rect.y + graph_rect.h, 14.0, GRAY);
    if let Some((r#gen, time, px, py)) = hovered_data {
        draw_circle(px, py, 5.0, YELLOW);
        let info = format!("Gen {}: {}", r#gen, format_time(time, tick_to_mins));
        let dims = measure_text(&info, None, 16, 1.0);
        let tx = (px + 10.0).min(screen_width() - dims.width - 10.0);
        draw_rectangle(tx - 5.0, py - 25.0, dims.width + 10.0, 25.0, Color::new(0.0, 0.1, 0.1, 0.9));
        draw_text(&info, tx, py - 8.0, 16.0, WHITE);
    }
    mouse_on_panel
}

fn mx_prev() -> f32 {
    static mut LAST_MX: f32 = 0.0;
    let cur = mouse_position().0;
    let old = unsafe { LAST_MX };
    unsafe { LAST_MX = cur; }
    old
}

pub fn draw_config_panel(
    mx: f32, my: f32, left_clicked: bool, _is_mouse_down: bool, _frame_time: f32,
    config: &mut crate::config::SimConfig,
    last_saved: &crate::config::SimConfig,
    scroll: &mut f32,
    search_query: &mut String,
    changed: &mut bool,
    _save_requested: &mut bool,
    active_button: &mut Option<String>,
) {
    let layout = ui_logic::calculate_config_layout(screen_width(), screen_height());
    draw_rectangle(layout.x, layout.y, layout.w, layout.h, Color::new(0.0, 0.05, 0.05, 0.95));
    draw_rectangle_lines(layout.x, layout.y, layout.w, layout.h, 2.0, GRAY);
    draw_text("SIMULATION CONFIGURATION", layout.x + 20.0, layout.y + 35.0, 20.0, Color::new(0.0, 1.0, 1.0, 1.0));
    draw_text("Search:", layout.x + 20.0, layout.y + 65.0, 16.0, WHITE);
    draw_rectangle(layout.x + 85.0, layout.y + 50.0, layout.w - 110.0, 20.0, BLACK);
    draw_text(search_query, layout.x + 90.0, layout.y + 65.0, 16.0, GREEN);
    while let Some(c) = get_char_pressed() { if c.is_alphanumeric() || c == '.' || c == '_' { search_query.push(c); } }
    if is_key_pressed(KeyCode::Backspace) { search_query.pop(); }
    let items = ui_logic::get_filtered_config_items(config, last_saved, search_query);
    let list_y = layout.y + 80.0;
    let row_h = 30.0;
    let visible_rows = ((layout.h - 100.0) / row_h) as usize;
    for i in 0..visible_rows {
        let idx = i + (*scroll as usize);
        if idx >= items.len() { break; }
        let item = &items[idx];
        let ry = list_y + i as f32 * row_h;
        let color = Color::new(item.color.r, item.color.g, item.color.b, 1.0);
        draw_text(&item.label, layout.x + 20.0, ry + 20.0, 14.0, color);
        draw_text(&item.value_str, layout.x + 180.0, ry + 20.0, 14.0, WHITE);
        let btn_m = Rect::new(layout.x + 250.0, ry + 5.0, 20.0, 20.0);
        let btn_p = Rect::new(layout.x + 280.0, ry + 5.0, 20.0, 20.0);
        let m_h = btn_m.contains(vec2(mx, my)); let p_h = btn_p.contains(vec2(mx, my));
        draw_rectangle(btn_m.x, btn_m.y, btn_m.w, btn_m.h, if m_h { RED } else { DARKGRAY });
        draw_rectangle(btn_p.x, btn_p.y, btn_p.w, btn_p.h, if p_h { GREEN } else { DARKGRAY });
        draw_text("-", btn_m.x + 7.0, btn_m.y + 15.0, 18.0, WHITE);
        draw_text("+", btn_p.x + 5.0, btn_p.y + 15.0, 18.0, WHITE);
        if left_clicked && (m_h || p_h) {
            *changed = true;
            let dir = if p_h { 1.0 } else { -1.0 };
            ui_logic::update_config_value(config, &item.key, dir);
            *active_button = Some(item.key.clone());
        }
    }
}
