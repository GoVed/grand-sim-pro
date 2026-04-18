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
    pub cumulative_ticks: u64,
    pub last_telemetry_tick: u64,
    pub cumulative_births: u64,
    pub cumulative_deaths: u64,
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

use std::sync::mpsc::Sender;
use std::io::{Read, Write};

struct ProgressWriter<W: Write> {
    inner: W,
    total: u64,
    written: u64,
    tx: Sender<ProgressReport>,
    msg: String,
    base: f32,
    scale: f32,
}

impl<W: Write> Write for ProgressWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.written += n as u64;
        if self.total > 0 {
            let p = self.base + (self.written as f32 / self.total as f32) * self.scale;
            let _ = self.tx.send(ProgressReport { message: self.msg.clone(), progress: p.min(self.base + self.scale) });
        }
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> { self.inner.flush() }
}

struct ProgressReader<R: Read> {
    inner: R,
    total: u64,
    read: u64,
    tx: Sender<ProgressReport>,
    msg: String,
    base: f32,
    scale: f32,
}

impl<R: Read> Read for ProgressReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.read += n as u64;
        if self.total > 0 {
            let p = self.base + (self.read as f32 / self.total as f32) * self.scale;
            let _ = self.tx.send(ProgressReport { message: self.msg.clone(), progress: p.min(self.base + self.scale) });
        }
        Ok(n)
    }
}

pub async fn save_everything(shared: &SharedData, name: &str, is_last_run: bool, progress: Option<Sender<ProgressReport>>) {
    let report = |msg: &str, p: f32| {
        if let Some(ref tx) = progress {
            let _ = tx.send(ProgressReport { message: msg.to_string(), progress: p });
        }
    };

    report("Preparing Save...", 0.05);
    let base_path = if is_last_run { format!("{}", name) } else { format!("saves/{}", name) };
    let _ = std::fs::create_dir_all(&base_path);
    
    // Save state using Bincode + Zstd (Multithreaded)
    let state = FullState { shared: shared.clone() };
    if let Ok(file) = std::fs::File::create(format!("{}/state.bin.zst", base_path)) {
        let uncompressed_size = bincode::serialized_size(&state).unwrap_or(0);
        
        let mut encoder = zstd::Encoder::new(file, 3).expect("Failed to create Zstd encoder");
        // Use all available cores for compression
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) as u32;
        encoder.multithread(cores).expect("Failed to enable Zstd multithreading");
        
        if let Some(ref tx) = progress {
            let mut writer = ProgressWriter {
                inner: encoder,
                total: uncompressed_size,
                written: 0,
                tx: tx.clone(),
                msg: "Compressing (Parallel) & Saving State...".to_string(),
                base: 0.1,
                scale: 0.6,
            };
            let _ = bincode::serialize_into(&mut writer, &state);
            let _ = writer.inner.finish(); // Finish the Zstd Encoder
        } else {
            let _ = bincode::serialize_into(&mut encoder, &state);
            let _ = encoder.finish();
        }
    }
    
    report("Copying Telemetry...", 0.75);
    if std::path::Path::new("telemetry.csv").exists() {
        let _ = std::fs::copy("telemetry.csv", format!("{}/telemetry.csv", base_path));
    }
    
    report("Extracting Top Brains...", 0.85);
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

    report("Opening Save File...", 0.05);
    let zst_path = path.join("state.bin.zst");
    let gz_path = path.join("state.bin.gz");
    let json_path = path.join("state.json");
    
    if zst_path.exists() {
        if let Ok(file) = std::fs::File::open(&zst_path) {
            let compressed_size = file.metadata().map(|m| m.len()).unwrap_or(0);
            let reader = ProgressReader {
                inner: file,
                total: compressed_size,
                read: 0,
                tx: progress.clone(),
                msg: "Decompressing Binary (Parallel) State...".to_string(),
                base: 0.1,
                scale: 0.7,
            };
            let mut decoder = zstd::Decoder::new(reader).expect("Failed to create Zstd decoder");
            match bincode::deserialize_from(&mut decoder) {
                Ok(state) => {
                    report("Restoring Telemetry...", 0.9);
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
    } else if gz_path.exists() {
        // Fallback for Gzip if anyone has old saves
        report("Decompressing Legacy Gzip State...", 0.1);
        if let Ok(file) = std::fs::File::open(&gz_path) {
            let mut decoder = flate2::read::GzDecoder::new(file);
            match bincode::deserialize_from(&mut decoder) {
                Ok(state) => {
                    report("Restoring Telemetry...", 0.9);
                    let _ = std::fs::copy(path.join("telemetry.csv"), "telemetry.csv");
                    report("Done!", 1.0);
                    return Some(state);
                }
                Err(e) => {
                    report(&format!("Gzip Load Error: {}", e), 0.0);
                    return None;
                }
            }
        }
    } else if json_path.exists() {
        report("Parsing Legacy JSON State...", 0.3);
        if let Ok(data) = std::fs::read_to_string(json_path) {
            match serde_json::from_str::<FullState>(&data) {
                Ok(state) => {
                    report("Restoring Telemetry...", 0.9);
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