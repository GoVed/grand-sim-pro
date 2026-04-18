/*
 * Grand Sim Pro: A high-performance GPGPU evolutionary agent simulation.
 * Part of an independent research project into emergent biological complexity.
 *
 * Copyright (C) 2026 Ved Hirenkumar Suthar
 * Licensed under the GNU General Public License v3.0 or later.
 * * This software is provided "as is", without warranty of any kind.
 * See the LICENSE file in the project root for full license details.
 */

use std::fs::OpenOptions;
use std::io::{Write, BufWriter};
use std::path::Path;
use crate::agent::AgentState;
use crate::environment::CellState;
use crate::config::SimConfig;

pub struct TelemetryExporter {
    file_path: String,
    initialized: bool,
}

impl TelemetryExporter {
    pub fn new(file_path: &str) -> Self {
        Self {
            file_path: file_path.to_string(),
            initialized: false,
        }
    }

    fn initialize(&mut self) -> std::io::Result<()> {
        if self.initialized { return Ok(()); }
        
        let path = Path::new(&self.file_path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let exists = path.exists();
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .append(true)
            .open(&self.file_path)?;
        
        let mut writer = BufWriter::new(file);
        
        if !exists {
            // CSV Header
            writeln!(writer, "Generation,Tick,Population,AvgAge,AvgHealth,AvgWealth,AvgFood,AvgStamina,AvgWater,InfraRoads,InfraHousing,InfraFarms,InfraStorage,TotalBirths,TotalDeaths,AvgAggression,AvgAltruism,AvgAskPrice,AvgBidPrice,TotalMarketFood,TotalMarketWealth,PhenoVariance")?;
        }
        
        self.initialized = true;
        Ok(())
    }

    pub fn export_optimized(
        &mut self,
        config: &SimConfig,
        states: &[AgentState],
        cells: &[CellState],
        cumulative_ticks: u64,
        cumulative_births: u64,
        cumulative_deaths: u64,
        generation: u32
    ) -> std::io::Result<()> {
        if config.telemetry.enabled == 0 { return Ok(()); }
        self.initialize()?;

        let file = OpenOptions::new().append(true).open(&self.file_path)?;
        let mut writer = BufWriter::new(file);

        let living_states: Vec<_> = states.iter().filter(|s| s.health > 0.0).collect();
        let pop_count = living_states.len();

        let (avg_age, avg_health, avg_wealth, avg_food, avg_stamina, avg_water, avg_aggression, avg_altruism, avg_ask, avg_bid, pheno_var) = if pop_count > 0 {
            let mut sum_age = 0.0; let mut sum_health = 0.0; let mut sum_wealth = 0.0;
            let mut sum_food = 0.0; let mut sum_stamina = 0.0; let mut sum_water = 0.0;
            let mut sum_aggr = 0.0; let mut sum_repro = 0.0;
            let mut sum_ask = 0.0; let mut sum_bid = 0.0;

            for s in &living_states {
                sum_age += s.age; sum_health += s.health; sum_wealth += s.wealth;
                sum_food += s.food; sum_stamina += s.stamina; sum_water += s.water;
                sum_aggr += s.attack_intent; sum_repro += s.reproduce_desire;
                sum_ask += s.ask_price; sum_bid += s.bid_price;
            }

            let mut mean_r = 0.0; let mut mean_g = 0.0; let mut mean_b = 0.0;
            for s in &living_states { mean_r += s.pheno_r; mean_g += s.pheno_g; mean_b += s.pheno_b; }
            mean_r /= pop_count as f32; mean_g /= pop_count as f32; mean_b /= pop_count as f32;
            
            let mut var = 0.0;
            for s in &living_states {
                var += (s.pheno_r - mean_r).powi(2) + (s.pheno_g - mean_g).powi(2) + (s.pheno_b - mean_b).powi(2);
            }
            var /= pop_count as f32;
            
            let p_f32 = pop_count as f32;
            (
                sum_age / p_f32, sum_health / p_f32, sum_wealth / p_f32,
                sum_food / p_f32, sum_stamina / p_f32, sum_water / p_f32,
                sum_aggr / p_f32, sum_repro / p_f32, sum_ask / p_f32, sum_bid / p_f32,
                if var.is_nan() { 0.0 } else { var }
            )
        } else {
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        };

        let mut infra_roads = 0.0;
        let mut infra_housing = 0.0;
        let mut infra_farms = 0.0;
        let mut infra_storage = 0.0;
        let mut total_market_food = 0.0;
        let mut total_market_wealth = 0.0;

        for cell in cells {
            infra_roads += cell.infra_roads as f32 / 1000.0;
            infra_housing += cell.infra_housing as f32 / 1000.0;
            infra_farms += cell.infra_farms as f32 / 1000.0;
            infra_storage += cell.infra_storage as f32 / 1000.0;
            total_market_food += cell.market_food as f32 / 1000.0;
            total_market_wealth += cell.market_wealth as f32 / 1000.0;
        }

        let sanitize = |v: f32| if v.is_nan() || v.is_infinite() { 0.0 } else { v };

        writeln!(
            writer,
            "{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{},{},{:.6},{:.6},{:.6},{:.6},{:.0},{:.0},{:.6}",
            generation, cumulative_ticks, pop_count, 
            sanitize(avg_age), sanitize(avg_health), sanitize(avg_wealth), sanitize(avg_food), sanitize(avg_stamina), sanitize(avg_water),
            infra_roads, infra_housing, infra_farms, infra_storage, cumulative_births, cumulative_deaths,
            sanitize(avg_aggression), sanitize(avg_altruism), sanitize(avg_ask), sanitize(avg_bid), total_market_food, total_market_wealth, sanitize(pheno_var)
        )?;

        Ok(())
    }

    pub fn reset_initialization(&mut self) {
        self.initialized = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SimConfig;

    #[test]
    fn test_telemetry_export() {
        let config = SimConfig::default();
        let mut states = vec![AgentState::default(); 10];
        for (i, s) in states.iter_mut().enumerate() { 
            s.health = 100.0;
            s.attack_intent = i as f32 * 0.1;
            s.reproduce_desire = 0.5;
        } 
        let cells = vec![CellState::default(); 100];
        let mut exporter = TelemetryExporter::new("test_telemetry.csv");
        
        let result = exporter.export_optimized(&config, &states, &cells, 100, 10, 0, 0);
        assert!(result.is_ok());
        
        let content = std::fs::read_to_string("test_telemetry.csv").unwrap();
        assert!(content.contains("Generation,Tick,Population"));
        assert!(content.contains("0,100,10"));
        // Aggression should be (0+0.1+0.2+0.3+0.4+0.5+0.6+0.7+0.8+0.9)/10 = 0.45
        assert!(content.contains(",0.4500,"));
        
        // Cleanup
        let _ = std::fs::remove_file("test_telemetry.csv");
    }
}
