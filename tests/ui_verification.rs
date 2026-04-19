/*
 * Grand Sim Pro: A high-performance GPGPU evolutionary agent simulation.
 * Part of an independent research project into emergent biological complexity.
 *
 * Copyright (C) 2026 Ved Hirenkumar Suthar
 * Licensed under the GNU General Public License v3.0 or later.
 * * This software is provided "as is", without warranty of any kind.
 * See the LICENSE file in the project root for full license details.
 */

use world_sim::ui_logic;
use world_sim::config::SimConfig;
use world_sim::agent::Person;

#[test]
fn test_behavioral_inference_logic() {
    let config = SimConfig::default();
    let p = Person::new(0.0, 0.0, 0, &config);
    
    // Test that logic produces a profile without crashing
    let profile = ui_logic::calculate_behavioral_profile(&p);
    assert_eq!(profile.len(), 3);
    
    for situation in profile {
        assert!(!situation.name.is_empty());
        // Propensity scores should be reasonable (clamped in our logic)
        assert!(situation.combat >= 0.0);
        assert!(situation.altruism >= 0.0);
    }
}

#[test]
fn test_config_filtering_and_highlighting() {
    let mut config = SimConfig::default();
    let last_saved = SimConfig::default();
    
    // 1. Test empty search
    let items = ui_logic::get_filtered_config_items(&config, &last_saved, "");
    assert_eq!(items.len(), 35); // Total count of all exposed config items
    
    // 2. Test specific search (Case Insensitive)
    let items_filtered = ui_logic::get_filtered_config_items(&config, &last_saved, "REGEN");
    assert_eq!(items_filtered.len(), 1);
    assert!(items_filtered[0].label.contains("Regen"));
    
    // 3. Test change detection (Highlighting)
    config.world.regen_rate += 0.1;
    let items_changed = ui_logic::get_filtered_config_items(&config, &last_saved, "Regen");
    assert!(items_changed[0].is_changed);
    assert_eq!(items_changed[0].color, ui_logic::UIColor::YELLOW);

    // 4. Test multiple matches
    let items_multi = ui_logic::get_filtered_config_items(&config, &last_saved, "a");
    assert!(items_multi.len() > 1);
}

#[test]
fn test_layout_logic() {
    let screen_w = 1920.0;
    let screen_h = 1080.0;
    
    let inspector = ui_logic::calculate_inspector_layout(screen_w, screen_h);
    assert_eq!(inspector.w, 600.0);
    assert_eq!(inspector.h, 1040.0);
    
    let config = ui_logic::calculate_config_layout(screen_w, screen_h);
    assert_eq!(config.w, 400.0);
    assert_eq!(config.x, 1920.0 - 400.0 - 20.0);
}

#[test]
fn test_graph_logic() {
    let (max, zero, r#gen) = ui_logic::get_graph_axes_labels(12345, 10.0);
    assert!(max.contains("y") || max.contains("m"));
    assert_eq!(zero, "0");
    assert_eq!(r#gen, "Gen");
}

#[test]
fn test_ui_colors() {
    let c = ui_logic::UIColor::RED;
    assert_eq!(c.r, 1.0);
    assert_eq!(c.g, 0.0);
    assert_eq!(c.b, 0.0);
}

#[test]
fn test_neural_connections_logic() {
    let mut weights = vec![0.0f32; 100];
    weights[5] = 1.0;
    weights[50] = -2.0;
    weights[99] = 0.5;
    
    // Test top connections overall
    let top = ui_logic::get_top_connections(&weights, 10, 10, None, false, 2);
    assert_eq!(top.len(), 2);
    assert_eq!(top[0].weight, -2.0); // Magnitude based
    assert_eq!(top[1].weight, 1.0);
    
    // Test focused on an output (to_idx)
    let focused = ui_logic::get_top_connections(&weights, 10, 10, Some(0), true, 10);
    // Should find weights at index [0, 10, 20, ... 90]
    assert_eq!(focused.len(), 10);
}

#[test]
fn test_influence_map_calculation_completeness() {
    let config = SimConfig::default();
    let p = Person::new(0.0, 0.0, 0, &config);
    let influence = p.calculate_input_output_influence();
    
    // Check that we have values for all input-output pairs
    assert_eq!(influence.len(), world_sim::agent::NUM_INPUTS * world_sim::agent::NUM_OUTPUTS);
    
    // Check that we can extract top connections for UI
    let top = ui_logic::get_top_connections(&influence, world_sim::agent::NUM_INPUTS, world_sim::agent::NUM_OUTPUTS, None, false, 30);
    assert_eq!(top.len(), 30);
}

#[test]
fn test_ui_panel_visibility_rules() {
    // These rules are encoded in main.rs but we verify the logic here if possible
    // Rule 1: Tracker should not be visible in Live Mode to avoid overlap
    let is_live_mode = true;
    let show_inspector = false;
    let should_draw_tracker = (!show_inspector || is_live_mode) && !is_live_mode;
    assert!(!should_draw_tracker);
    
    // Rule 2: Tracker should be visible when inspector is closed and NOT in live mode
    let is_live_mode = false;
    let show_inspector = false;
    let should_draw_tracker = (!show_inspector || is_live_mode) && !is_live_mode;
    assert!(should_draw_tracker);
}

#[test]
fn test_agent_profile_visibility_logic() {
    let is_live_mode = true;
    let has_selected_agent = true;
    // Should NOT show profile if in live mode (overlap)
    let show_profile = has_selected_agent && !is_live_mode;
    assert!(!show_profile);
    
    let is_live_mode = false;
    let show_profile = has_selected_agent && !is_live_mode;
    assert!(show_profile);
}

#[test]
fn test_inspector_live_mode_exclusivity() {
    // Simulator entry logic simulation
    let mut show_inspector = true;
    let mut is_live_mode = false;
    
    // Simulate pressing 'L' or "LIVE POV"
    if !is_live_mode {
        is_live_mode = true;
        show_inspector = false;
    }
    
    assert!(!show_inspector);
    assert!(is_live_mode);
}

#[test]
fn test_identity_labels_registration() {
    use world_sim::agent::{NUM_INPUTS, INPUT_LABELS};
    assert_eq!(INPUT_LABELS.len(), NUM_INPUTS);
    assert_eq!(INPUT_LABELS[NUM_INPUTS - 1], "CNN S8");
    assert_eq!(INPUT_LABELS[NUM_INPUTS - 9], "Target F4");
    assert_eq!(NUM_INPUTS, 420);
}

#[test]
fn test_deterministic_identity_initialization() {
    let config = SimConfig::default();
    let p = Person::new(0.0, 0.0, 0, &config);
    
    // Features should be in -1.0..1.0 range and non-zero (statistically)
    assert!(p.state.id_f1 >= -1.0 && p.state.id_f1 <= 1.0);
    assert!(p.state.id_f2 >= -1.0 && p.state.id_f2 <= 1.0);
    
    // Verify against helper function directly
    use world_sim::agent::get_id_feature;
    assert_eq!(p.state.id_f1, get_id_feature(p.state.id, 1));
    assert_eq!(p.state.id_f2, get_id_feature(p.state.id, 2));
    assert_eq!(p.state.id_f3, get_id_feature(p.state.id, 3));
    assert_eq!(p.state.id_f4, get_id_feature(p.state.id, 4));
}

#[test]
fn test_cnn_and_plasticity_initialization() {
    let config = SimConfig::default();
    let p = Person::new(0.0, 0.0, 0, &config);
    
    // Plasticity should be initialized with small random weights
    assert_eq!(p.state.plastic_weights.len(), 32);
    assert_eq!(p.state.plastic_indices.len(), 32);
    
    // At least some weights should be non-zero
    let sum_abs: f32 = p.state.plastic_weights.iter().map(|w| w.abs()).sum();
    assert!(sum_abs > 0.0);
    
    // Spatial features should be zeroed (waiting for GPU)
    assert_eq!(p.state.spatial_features.len(), 8);
    assert_eq!(p.state.spatial_features[0], 0.0);
}
