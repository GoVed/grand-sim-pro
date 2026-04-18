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
use crate::simulation::SimulationManager;
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

    pub fn export(&mut self, sim: &SimulationManager, config: &SimConfig, total_ticks: u64, generation: u32) -> std::io::Result<()> {
        if config.telemetry.enabled == 0 { return Ok(()); }
        self.initialize()?;

        let file = OpenOptions::new().append(true).open(&self.file_path)?;
        let mut writer = BufWriter::new(file);

        let living_states: Vec<_> = sim.states.iter().filter(|s| s.health > 0.0).collect();
        let pop_count = living_states.len();

        let (avg_age, avg_health, avg_wealth, avg_food, avg_stamina, avg_water, avg_aggression, avg_altruism, avg_ask, avg_bid, pheno_var) = if pop_count > 0 {
            let sum_age: f32 = living_states.iter().map(|s| s.age).sum();
            let sum_health: f32 = living_states.iter().map(|s| s.health).sum();
            let sum_wealth: f32 = living_states.iter().map(|s| s.wealth).sum();
            let sum_food: f32 = living_states.iter().map(|s| s.food).sum();
            let sum_stamina: f32 = living_states.iter().map(|s| s.stamina).sum();
            let sum_water: f32 = living_states.iter().map(|s| s.water).sum();
            let sum_aggr: f32 = living_states.iter().map(|s| s.attack_intent).sum();
            let sum_repro_desire: f32 = living_states.iter().map(|s| s.reproduce_desire).sum(); // Proxy for altruism/sociality
            let sum_ask: f32 = living_states.iter().map(|s| s.ask_price).sum();
            let sum_bid: f32 = living_states.iter().map(|s| s.bid_price).sum();

            // Calculate Phenotype Variance (R+G+B variance)
            let mut mean_r = 0.0; let mut mean_g = 0.0; let mut mean_b = 0.0;
            for s in &living_states { mean_r += s.pheno_r; mean_g += s.pheno_g; mean_b += s.pheno_b; }
            mean_r /= pop_count as f32; mean_g /= pop_count as f32; mean_b /= pop_count as f32;
            
            let mut var = 0.0;
            for s in &living_states {
                var += (s.pheno_r - mean_r).powi(2) + (s.pheno_g - mean_g).powi(2) + (s.pheno_b - mean_b).powi(2);
            }
            var /= pop_count as f32;
            
            (
                sum_age / pop_count as f32,
                sum_health / pop_count as f32,
                sum_wealth / pop_count as f32,
                sum_food / pop_count as f32,
                sum_stamina / pop_count as f32,
                sum_water / pop_count as f32,
                sum_aggr / pop_count as f32,
                sum_repro_desire / pop_count as f32,
                sum_ask / pop_count as f32,
                sum_bid / pop_count as f32,
                var
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

        for cell in &sim.env.map_cells {
            infra_roads += cell.infra_roads as f32 / 1000.0;
            infra_housing += cell.infra_housing as f32 / 1000.0;
            infra_farms += cell.infra_farms as f32 / 1000.0;
            infra_storage += cell.infra_storage as f32 / 1000.0;
            total_market_food += cell.market_food as f32 / 1000.0;
            total_market_wealth += cell.market_wealth as f32 / 1000.0;
        }

        writeln!(
            writer,
            "{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{},{},{:.4},{:.4},{:.4},{:.4},{:.0},{:.0},{:.4}",
            generation,
            total_ticks,
            pop_count,
            avg_age,
            avg_health,
            avg_wealth,
            avg_food,
            avg_stamina,
            avg_water,
            infra_roads,
            infra_housing,
            infra_farms,
            infra_storage,
            sim.total_births,
            sim.total_deaths,
            avg_aggression,
            avg_altruism,
            avg_ask,
            avg_bid,
            total_market_food,
            total_market_wealth,
            pheno_var
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
    use crate::simulation::SimulationManager;

    #[test]
    fn test_telemetry_export() {
        let config = SimConfig::default();
        let sim = SimulationManager::new(100, 100, 12345, 10, &config, Vec::new());
        let mut exporter = TelemetryExporter::new("test_telemetry.csv");
        
        let result = exporter.export(&sim, &config, 100, 0);
        assert!(result.is_ok());
        
        let content = std::fs::read_to_string("test_telemetry.csv").unwrap();
        assert!(content.contains("Generation,Tick,Population"));
        assert!(content.contains("PhenoVariance"));
        assert!(content.contains("0,100,10"));
        
        // Cleanup
        let _ = std::fs::remove_file("test_telemetry.csv");
    }
}
