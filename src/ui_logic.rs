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
    pub key: String, // Internal key for identification
}

/// Pure logic for determining which config items to show and their visual state.
pub fn get_filtered_config_items(
    config: &crate::config::SimConfig,
    last_saved: &crate::config::SimConfig,
    query: &str,
) -> Vec<ConfigItemInfo> {
    let q = query.to_lowercase();
    let mut results = Vec::new();
    
    fn push_item(res: &mut Vec<ConfigItemInfo>, label: &str, key: &str, val: f32, last: f32, q: &str) {
        if q.is_empty() || label.to_lowercase().contains(q) || key.to_lowercase().contains(q) {
            let changed = (val - last).abs() > 0.000001;
            res.push(ConfigItemInfo {
                label: label.to_string(),
                is_changed: changed,
                value_str: if val.fract() == 0.0 { format!("{:.0}", val) } else { format!("{:.4}", val) },
                color: if changed { UIColor::YELLOW } else { UIColor::WHITE },
                key: key.to_string(),
            });
        }
    }

    // World
    push_item(&mut results, "W: Regen Rate", "world.regen_rate", config.world.regen_rate, last_saved.world.regen_rate, &q);
    push_item(&mut results, "W: Max Resource", "world.max_tile_resource", config.world.max_tile_resource, last_saved.world.max_tile_resource, &q);
    push_item(&mut results, "W: Max Water", "world.max_tile_water", config.world.max_tile_water, last_saved.world.max_tile_water, &q);
    push_item(&mut results, "W: Tick to Mins", "world.tick_to_mins", config.world.tick_to_mins, last_saved.world.tick_to_mins, &q);

    // Sim
    push_item(&mut results, "S: Agent Count", "sim.agent_count", config.sim.agent_count as f32, last_saved.sim.agent_count as f32, &q);
    push_item(&mut results, "S: Spawn Group", "sim.spawn_group_size", config.sim.spawn_group_size as f32, last_saved.sim.spawn_group_size as f32, &q);
    push_item(&mut results, "S: Founders", "sim.founder_count", config.sim.founder_count as f32, last_saved.sim.founder_count as f32, &q);
    push_item(&mut results, "S: Load Mode (0..2)", "sim.load_saved_agents_on_start", config.sim.load_saved_agents_on_start as f32, last_saved.sim.load_saved_agents_on_start as f32, &q);

    // Bio
    push_item(&mut results, "B: Base Speed", "bio.base_speed", config.bio.base_speed, last_saved.bio.base_speed, &q);
    push_item(&mut results, "B: Max Age", "bio.max_age", config.bio.max_age, last_saved.bio.max_age, &q);
    push_item(&mut results, "B: Max HP", "bio.max_health", config.bio.max_health, last_saved.bio.max_health, &q);
    push_item(&mut results, "B: Max Stamina", "bio.max_stamina", config.bio.max_stamina, last_saved.bio.max_stamina, &q);
    push_item(&mut results, "B: Max Water", "bio.max_water", config.bio.max_water, last_saved.bio.max_water, &q);
    push_item(&mut results, "B: Puberty Age", "bio.puberty_age", config.bio.puberty_age, last_saved.bio.puberty_age, &q);
    push_item(&mut results, "B: Gestation", "bio.gestation_period", config.bio.gestation_period, last_saved.bio.gestation_period, &q);
    push_item(&mut results, "B: Starvation Rate", "bio.starvation_rate", config.bio.starvation_rate, last_saved.bio.starvation_rate, &q);
    push_item(&mut results, "B: Infant Spd", "bio.infant_speed_mult", config.bio.infant_speed_mult, last_saved.bio.infant_speed_mult, &q);
    push_item(&mut results, "B: Infant Stam", "bio.infant_stamina_mult", config.bio.infant_stamina_mult, last_saved.bio.infant_stamina_mult, &q);

    // Eco
    push_item(&mut results, "E: Baseline Cost", "eco.baseline_cost", config.eco.baseline_cost, last_saved.eco.baseline_cost, &q);
    push_item(&mut results, "E: Repro Cost", "eco.reproduction_cost", config.eco.reproduction_cost, last_saved.eco.reproduction_cost, &q);
    push_item(&mut results, "E: Boat Cost", "eco.boat_cost", config.eco.boat_cost, last_saved.eco.boat_cost, &q);
    push_item(&mut results, "E: Water Transf", "eco.water_transfer_amount", config.eco.water_transfer_amount, last_saved.eco.water_transfer_amount, &q);
    push_item(&mut results, "E: Spoilage Rate", "eco.base_spoilage_rate", config.eco.base_spoilage_rate, last_saved.eco.base_spoilage_rate, &q);

    // Genetics
    push_item(&mut results, "G: Mutation Rate", "genetics.mutation_rate", config.genetics.mutation_rate, last_saved.genetics.mutation_rate, &q);
    push_item(&mut results, "G: Mut Strength", "genetics.mutation_strength", config.genetics.mutation_strength, last_saved.genetics.mutation_strength, &q);
    push_item(&mut results, "G: Random Spawn %", "genetics.random_spawn_percentage", config.genetics.random_spawn_percentage, last_saved.genetics.random_spawn_percentage, &q);

    // Infra
    push_item(&mut results, "I: Infra Cost", "infra.infra_cost", config.infra.infra_cost, last_saved.infra.infra_cost, &q);
    push_item(&mut results, "I: Road Bonus", "infra.road_speed_bonus", config.infra.road_speed_bonus, last_saved.infra.road_speed_bonus, &q);
    push_item(&mut results, "I: Housing Bonus", "infra.housing_rest_bonus", config.infra.housing_rest_bonus, last_saved.infra.housing_rest_bonus, &q);
    push_item(&mut results, "I: Storage Bonus", "infra.storage_rot_reduction", config.infra.storage_rot_reduction, last_saved.infra.storage_rot_reduction, &q);

    // Combat
    push_item(&mut results, "C: Attacker Dmg", "combat.attacker_damage", config.combat.attacker_damage, last_saved.combat.attacker_damage, &q);
    push_item(&mut results, "C: Steal Amt", "combat.steal_amount", config.combat.steal_amount, last_saved.combat.steal_amount, &q);
    
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

pub fn get_graph_axes_labels(max_ticks: u64, tick_to_mins: f32) -> (String, String, String) {
    let total_mins = max_ticks as f32 * tick_to_mins;
    let years = (total_mins / 525600.0) as i32;
    let months = ((total_mins % 525600.0) / 43800.0) as i32;
    
    let label = if years > 0 {
        format!("{}y {}m", years, months)
    } else {
        format!("{}m", months)
    };
    
    (label, "0".to_string(), "Gen".to_string())
}

#[derive(Debug, Clone)]
pub struct NeuralConnection {
    pub from_idx: usize,
    pub to_idx: usize,
    pub weight: f32,
}

pub fn get_top_connections(
    weights: &[f32], 
    from_count: usize, 
    to_count: usize, 
    focus_idx: Option<usize>, 
    is_focus_output: bool,
    limit: usize
) -> Vec<NeuralConnection> {
    let mut connections = Vec::new();
    if let Some(focus) = focus_idx {
        if is_focus_output {
            for h in 0..from_count {
                let w = weights[h * to_count + focus];
                connections.push(NeuralConnection { from_idx: h, to_idx: focus, weight: w });
            }
        } else {
            for o in 0..to_count {
                let w = weights[focus * to_count + o];
                connections.push(NeuralConnection { from_idx: focus, to_idx: o, weight: w });
            }
        }
    } else {
        for h in 0..from_count {
            for o in 0..to_count {
                let w = weights[h * to_count + o];
                connections.push(NeuralConnection { from_idx: h, to_idx: o, weight: w });
            }
        }
    }
    connections.sort_by(|a, b| b.weight.abs().partial_cmp(&a.weight.abs()).unwrap());
    connections.truncate(limit);
    connections
}
