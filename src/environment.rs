use noise::{NoiseFn, Perlin, Fbm};

pub struct Environment {
    pub map_data: Vec<u8>,
    pub height_map: Vec<f32>,
    pub map_resources: Vec<f32>,
}

impl Environment {
    pub fn new(width: u32, height: u32, seed: u32, config: &crate::config::SimConfig) -> Self {
        let fbm = Fbm::<Perlin>::new(seed);
        
        let mut map_data = Vec::with_capacity((width * height * 4) as usize);
        let mut height_map = Vec::with_capacity((width * height) as usize);
        let mut map_resources = Vec::with_capacity((width * height) as usize);

        // Calculate the radius for our 4D mapping to match the original noise scale
        let radius_x = width as f64 / (150.0 * 2.0 * std::f64::consts::PI);
        let radius_y = height as f64 / (150.0 * 2.0 * std::f64::consts::PI);

        for y in 0..height {
            for x in 0..width {
                // Convert X and Y to angles
                let angle_x = (x as f64 / width as f64) * 2.0 * std::f64::consts::PI;
                let angle_y = (y as f64 / height as f64) * 2.0 * std::f64::consts::PI;
                
                // Sample 4D noise for seamless wrapping
                let val = fbm.get([
                    angle_x.cos() * radius_x, angle_x.sin() * radius_x,
                    angle_y.cos() * radius_y, angle_y.sin() * radius_y
                ]);

                let mut color = match val {
                    v if v < -0.2 => [10, 50, 150, 255],
                    v if v < 0.0  => [30, 100, 200, 255],
                    v if v < 0.1  => [240, 230, 140, 255],
                    v if v < 0.4  => [34, 139, 34, 255],
                    _             => [100, 100, 100, 255],
                };

                // Add topological contour lines for height visualization
                if val >= 0.0 && (val % 0.1).abs() < 0.015 {
                    color = [0, 0, 0, 180]; // Dark contour line
                }

                map_data.extend_from_slice(&color);
                height_map.push(val as f32);
                
                // Initialize land based on max configured economic scale
                let base_res = if val >= 0.0 { config.max_tile_resource } else { 0.0 };
                map_resources.push(base_res);
            }
        }
        Self { map_data, height_map, map_resources }
    }
}