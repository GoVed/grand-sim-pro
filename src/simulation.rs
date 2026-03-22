use crate::agent::Person;
use crate::environment::Environment;

pub struct SimulationManager {
    pub env: Environment,
    pub agents: Vec<Person>,
}

impl SimulationManager {
    pub fn new(width: u32, height: u32, seed: u32, count: u32) -> Self {
        let env = Environment::new(width, height, seed);
        let agents = (0..count).map(|_| Person::new((width/2) as f32, (height/2) as f32)).collect();
        Self { env, agents }
    }
}