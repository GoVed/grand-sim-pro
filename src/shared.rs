/*
 * Grand Sim Pro: A high-performance GPGPU evolutionary agent simulation.
 * Part of an independent research project into emergent biological complexity.
 *
 * Copyright (C) 2026 Ved Hirenkumar Suthar
 * Licensed under the GNU General Public License v3.0 or later.
 * * This software is provided "as is", without warranty of any kind.
 * See the LICENSE file in the project root for full license details.
 */

use crate::simulation::SimulationManager;
use crate::config::SimConfig;
use serde::{Serialize, Deserialize};

// Translates 10-minute ticks into realistic Years and Months
pub fn format_time(ticks: u64, tick_to_mins: f32) -> String {
    let total_mins = ticks as f64 * tick_to_mins as f64;
    let total_days = total_mins / (60.0 * 24.0);
    let years = (total_days / 365.0).floor() as u32;
    let months = ((total_days % 365.0) / 30.0).floor() as u32;
    if years > 0 {
        format!("{}y {}m", years, months)
    } else {
        format!("{}m", months)
    }
}

#[derive(PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum VisualMode { Default, Resources, Age, Gender, Pregnancy, MarketWealth, MarketFood, AskPrice, BidPrice, Infrastructure, DayNight, Temperature, Tribes, Water }

#[derive(PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum SortCol { Index, Age, Health, Food, Wealth, Gender, Speed, Heading, State, Outputs }

pub struct AgentRenderData {
    pub x: f32, pub y: f32, pub health: f32, pub food: f32,
    pub age: f32, pub wealth: f32, pub gender: f32, pub is_pregnant: f32,
    pub pheno_r: f32, pub pheno_g: f32, pub pheno_b: f32,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct SharedData {
    pub sim: SimulationManager,
    pub config: SimConfig,
    pub last_saved_config: SimConfig,
    pub is_paused: bool,
    pub restart_message_active: bool,
    pub ticks_per_loop: usize,
    pub total_ticks: u64,
    pub last_compute_time_micros: u128,
    pub ticks_per_second: f32,
    pub generation_survival_times: Vec<u64>,
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct FullState {
    pub shared: SharedData,
}

pub struct ProgressReport {
    pub message: String,
    pub progress: f32, // 0.0 to 1.0
}

use flate2::write::GzEncoder;
use flate2::read::GzDecoder;
use flate2::Compression;
use std::sync::mpsc::Sender;

pub async fn save_everything(shared: &SharedData, name: &str, is_last_run: bool, progress: Option<Sender<ProgressReport>>) {
    let report = |msg: &str, p: f32| {
        if let Some(ref tx) = progress {
            let _ = tx.send(ProgressReport { message: msg.to_string(), progress: p });
        }
    };

    report("Preparing Save...", 0.1);
    let base_path = if is_last_run { format!("{}", name) } else { format!("saves/{}", name) };
    let _ = std::fs::create_dir_all(&base_path);
    
    // Save state using Bincode + Gzip
    report("Compressing State...", 0.3);
    let state = FullState { shared: shared.clone() };
    let file = std::fs::File::create(format!("{}/state.bin.gz", base_path));
    if let Ok(file) = file {
        let mut encoder = GzEncoder::new(file, Compression::default());
        let _ = bincode::serialize_into(&mut encoder, &state);
        let _ = encoder.finish();
    }
    
    report("Copying Telemetry...", 0.7);
    // Save telemetry if it exists
    if std::path::Path::new("telemetry.csv").exists() {
        let _ = std::fs::copy("telemetry.csv", format!("{}/telemetry.csv", base_path));
    }
    
    report("Extracting Top Brains...", 0.8);
    // Save weights
    let weights_path = format!("{}/weights", base_path);
    let _ = std::fs::create_dir_all(&weights_path);
    let mut living_indices: Vec<_> = shared.sim.states.iter().enumerate()
        .filter(|(_, s)| s.health > 0.0)
        .map(|(i, _)| i)
        .collect();
    
    living_indices.sort_by(|&a_idx, &b_idx| {
        let s_a = &shared.sim.states[a_idx];
        let s_b = &shared.sim.states[b_idx];
        (s_b.wealth + s_b.food).partial_cmp(&(s_a.wealth + s_a.food)).unwrap_or(std::cmp::Ordering::Equal)
    });

    use crate::agent::Person;
    for i in 0..living_indices.len().min(shared.config.sim.founder_count as usize) {
        let idx = living_indices[i];
        let s = &shared.sim.states[idx];
        let p = Person { state: *s, genetics: shared.sim.genetics[s.genetics_index as usize] };
        let weights = p.extract_weights();
        if let Ok(json) = serde_json::to_string_pretty(&weights) {
            let _ = std::fs::write(format!("{}/agent_{}.json", weights_path, i), json);
        }
    }
    report("Done!", 1.0);
}

pub fn load_everything(path: &std::path::Path, progress: Sender<ProgressReport>) -> Option<FullState> {
    let report = |msg: &str, p: f32| {
        let _ = progress.send(ProgressReport { message: msg.to_string(), progress: p });
    };

    report("Opening Save File...", 0.1);
    let bin_path = path.join("state.bin.gz");
    let json_path = path.join("state.json");
    
    if bin_path.exists() {
        report("Decompressing Binary State...", 0.3);
        if let Ok(file) = std::fs::File::open(bin_path) {
            let decoder = GzDecoder::new(file);
            match bincode::deserialize_from(decoder) {
                Ok(state) => {
                    report("Restoring Telemetry...", 0.8);
                    let _ = std::fs::copy(path.join("telemetry.csv"), "telemetry.csv");
                    report("Done!", 1.0);
                    return Some(state);
                }
                Err(e) => {
                    report(&format!("Binary Load Error: {}", e), 0.0);
                    return None;
                }
            }
        }
    } else if json_path.exists() {
        report("Parsing Legacy JSON State...", 0.3);
        if let Ok(data) = std::fs::read_to_string(json_path) {
            match serde_json::from_str::<FullState>(&data) {
                Ok(state) => {
                    report("Restoring Telemetry...", 0.8);
                    let _ = std::fs::copy(path.join("telemetry.csv"), "telemetry.csv");
                    report("Done!", 1.0);
                    return Some(state);
                }
                Err(e) => {
                    report(&format!("JSON Load Error: {}", e), 0.0);
                    return None;
                }
            }
        }
    }
    
    None
}