/*
 * Grand Sim Pro: UI Logic Module
 * Pure logic for situational probing, configuration filtering, and layout.
 * This file MUST NOT import macroquad to remain testable in headless environments.
 */

use crate::agent::Person;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UIColor { pub r: f32, pub g: f32, pub b: f32, pub a: f32 }

impl UIColor {
    pub const WHITE: UIColor = UIColor { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const GRAY: UIColor = UIColor { r: 0.5, g: 0.5, b: 0.5, a: 1.0 };
    pub const CYAN: UIColor = UIColor { r: 0.0, g: 1.0, b: 0.8, a: 1.0 };
    pub const YELLOW: UIColor = UIColor { r: 1.0, g: 1.0, b: 0.0, a: 1.0 };
    pub const RED: UIColor = UIColor { r: 1.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const GREEN: UIColor = UIColor { r: 0.0, g: 1.0, b: 0.0, a: 1.0 };
    pub const BLUE: UIColor = UIColor { r: 0.2, g: 0.4, b: 1.0, a: 1.0 };
    pub const GOLD: UIColor = UIColor { r: 1.0, g: 0.8, b: 0.0, a: 1.0 };
    pub const MAGENTA: UIColor = UIColor { r: 1.0, g: 0.0, b: 1.0, a: 1.0 };
}

pub struct BehavioralSituation {
    pub name: &'static str,
    pub combat: f32,
    pub altruism: f32,
    pub industry: f32,
    pub trade: f32,
    pub agility: f32,
}

/// Calculates how an agent would behave in various situational archetypes.
pub fn calculate_behavioral_profile(a: &Person) -> Vec<BehavioralSituation> {
    let situations = [
        ("Crowded / Strange", [ (24, 1.0), (40, 1.0), (0, 1.0) ]),
        ("Starving / Desperate", [ (15, 0.0), (16, 0.0), (0, 1.0) ]),
        ("Prosperous / Safe", [ (33, 1.0), (15, 1.0), (0, 1.0) ])
    ];

    situations.iter().map(|(name, inputs_setup)| {
        let mut test_inputs = [0.0f32; 160];
        for (idx, val) in inputs_setup { test_inputs[*idx as usize] = *val; }
        let outputs = a.mental_simulation(&test_inputs);
        
        BehavioralSituation {
            name,
            combat: outputs[4].max(0.0),
            altruism: (outputs[2].max(0.0) + outputs[6].max(0.0)) / 2.0,
            industry: (outputs[26].max(0.0) + outputs[27].max(0.0) + outputs[28].max(0.0)) / 3.0,
            trade: (outputs[19].max(0.0) + outputs[20].max(0.0)) / 2.0,
            agility: (outputs[1].abs() + outputs[0].abs()) / 2.0,
        }
    }).collect()
}

pub struct ConfigItemInfo {
    pub label: String,
    pub is_changed: bool,
    pub value_str: String,
    pub color: UIColor,
}

/// Pure logic for determining which config items to show and their visual state.
pub fn get_filtered_config_items(
    config: &crate::config::SimConfig,
    last_saved: &crate::config::SimConfig,
    query: &str,
) -> Vec<ConfigItemInfo> {
    let q = query.to_lowercase();
    let mut results = Vec::new();
    
    // Helper to add config items (Refactored for testability)
    fn push_item(res: &mut Vec<ConfigItemInfo>, label: &str, val: f32, last: f32, q: &str) {
        if q.is_empty() || label.to_lowercase().contains(q) {
            let changed = (val - last).abs() > 0.000001;
            res.push(ConfigItemInfo {
                label: label.to_string(),
                is_changed: changed,
                value_str: format!("{:.4}", val),
                color: if changed { UIColor::YELLOW } else { UIColor::WHITE },
            });
        }
    }

    push_item(&mut results, "Regen Rate", config.world.regen_rate, last_saved.world.regen_rate, &q);
    push_item(&mut results, "Max Tile Resource", config.world.max_tile_resource, last_saved.world.max_tile_resource, &q);
    push_item(&mut results, "Agent Count", config.sim.agent_count as f32, last_saved.sim.agent_count as f32, &q);
    push_item(&mut results, "Max Age", config.bio.max_age, last_saved.bio.max_age, &q);
    push_item(&mut results, "Mutation Rate", config.genetics.mutation_rate, last_saved.genetics.mutation_rate, &q);
    
    results
}

pub struct PanelLayout {
    pub x: f32, pub y: f32, pub w: f32, pub h: f32,
}

pub fn calculate_inspector_layout(_screen_w: f32, screen_h: f32) -> PanelLayout {
    PanelLayout { x: 10.0, y: 20.0, w: 600.0, h: screen_h - 40.0 }
}

pub fn calculate_config_layout(screen_w: f32, screen_h: f32) -> PanelLayout {
    let panel_w = 400.0;
    let panel_h = (screen_h - 40.0).min(900.0);
    PanelLayout { x: screen_w - panel_w - 20.0, y: (screen_h - panel_h) / 2.0, w: panel_w, h: panel_h }
}

pub fn get_graph_axes_labels(max_time: u64) -> (String, String, String) {
    (format!("{}", max_time), "0".to_string(), "Gen".to_string())
}
