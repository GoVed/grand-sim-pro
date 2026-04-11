
use world_sim::ui_logic;
use world_sim::config::SimConfig;
use world_sim::agent::Person;

#[test]
fn test_behavioral_inference_logic() {
    let config = SimConfig::default();
    let p = Person::new(0.0, 0.0, &config);
    
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
    assert_eq!(items.len(), 5); // Regen, Max Food, Agent Count, Max Age, Mutation
    
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
