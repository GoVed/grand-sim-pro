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

#[derive(PartialEq, Clone, Copy)]
pub enum VisualMode { Default, Resources, Age, Gender, Pregnancy, MarketWealth, MarketFood, AskPrice, BidPrice, Infrastructure, DayNight, Temperature, Tribes, Water }

#[derive(PartialEq, Clone, Copy)]
pub enum SortCol { Index, Age, Health, Food, Wealth, Gender, Speed, Heading, State, Outputs }

pub struct AgentRenderData {
    pub x: f32, pub y: f32, pub health: f32, pub food: f32,
    pub age: f32, pub wealth: f32, pub gender: f32, pub is_pregnant: f32,
    pub pheno_r: f32, pub pheno_g: f32, pub pheno_b: f32,
}

pub struct SharedData {
    pub sim: SimulationManager,
    pub config: SimConfig,
    pub is_paused: bool,
    pub restart_message_active: bool,
    pub ticks_per_loop: usize,
    pub total_ticks: u64,
    pub last_compute_time_ms: u128,
    pub generation_survival_times: Vec<u64>,
}