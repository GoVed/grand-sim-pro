/*
 * Grand Sim Pro: A high-performance GPGPU evolutionary agent simulation.
 * Part of an independent research project into emergent biological complexity.
 *
 * Copyright (C) 2026 Ved Hirenkumar Suthar
 * Licensed under the GNU General Public License v3.0 or later.
 * * This software is provided "as is", without warranty of any kind.
 * See the LICENSE file in the project root for full license details.
 */

use std::fs::{File, OpenOptions};
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

        let file = File::create(&self.file_path)?;
        let mut writer = BufWriter::new(file);
        
        // CSV Header
        writeln!(writer, "Generation,Tick,Population,AvgAge,AvgHealth,AvgWealth,AvgFood,AvgStamina,AvgWater,InfraRoads,InfraHousing,InfraFarms,InfraStorage,TotalBirths,TotalDeaths")?;
        
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

        let (avg_age, avg_health, avg_wealth, avg_food, avg_stamina, avg_water) = if pop_count > 0 {
            let sum_age: f32 = living_states.iter().map(|s| s.age).sum();
            let sum_health: f32 = living_states.iter().map(|s| s.health).sum();
            let sum_wealth: f32 = living_states.iter().map(|s| s.wealth).sum();
            let sum_food: f32 = living_states.iter().map(|s| s.food).sum();
            let sum_stamina: f32 = living_states.iter().map(|s| s.stamina).sum();
            let sum_water: f32 = living_states.iter().map(|s| s.water).sum();
            
            (
                sum_age / pop_count as f32,
                sum_health / pop_count as f32,
                sum_wealth / pop_count as f32,
                sum_food / pop_count as f32,
                sum_stamina / pop_count as f32,
                sum_water / pop_count as f32,
            )
        } else {
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        };

        let mut infra_roads = 0.0;
        let mut infra_housing = 0.0;
        let mut infra_farms = 0.0;
        let mut infra_storage = 0.0;

        for cell in &sim.env.map_cells {
            infra_roads += cell.infra_roads as f32 / 1000.0;
            infra_housing += cell.infra_housing as f32 / 1000.0;
            infra_farms += cell.infra_farms as f32 / 1000.0;
            infra_storage += cell.infra_storage as f32 / 1000.0;
        }

        // Note: TotalBirths and TotalDeaths are harder to track per-interval without adding state to SimManager.
        // For now, we leave them as 0 or calculate them if possible.
        // Pending births can be a proxy for recent birth activity.
        let pending_births = sim.pending_births.len();

        writeln!(
            writer,
            "{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{},{}",
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
            pending_births,
            0 // Death tracking not yet implemented
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
        assert!(content.contains("0,100,10"));
        
        // Cleanup
        let _ = std::fs::remove_file("test_telemetry.csv");
    }
}
