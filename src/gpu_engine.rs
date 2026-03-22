use wgpu::util::DeviceExt;
use pollster::block_on;

pub struct GpuEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    agent_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    resource_buffer: wgpu::Buffer,
    staging_resource_buffer: wgpu::Buffer,
    agent_count: u32,
}

impl GpuEngine {
    pub fn new(agents: &[crate::agent::Person], map_heights: &[f32], map_resources: &[f32], config: &crate::config::SimConfig) -> Self {
        block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::PRIMARY,
                ..Default::default()
            });
            let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
            let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.unwrap();

            let agent_bytes = bytemuck::cast_slice(agents);
            let agent_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Agent Buffer"),
                contents: agent_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            });

            let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Buffer"),
                size: agent_bytes.len() as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let height_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Height Buffer"),
                contents: bytemuck::cast_slice(map_heights),
                usage: wgpu::BufferUsages::STORAGE,
            });
            
            let resource_bytes = bytemuck::cast_slice(map_resources);
            let resource_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Resource Buffer"),
                contents: resource_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            });

            let staging_resource_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Resource Buffer"),
                size: resource_bytes.len() as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            
            let config_bytes = bytemuck::bytes_of(config);
            let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Config Buffer"),
                contents: config_bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let shader = device.create_shader_module(wgpu::include_wgsl!("sim.wgsl"));
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Sim Pipeline"), layout: None, module: &shader, entry_point: "main",
                compilation_options: Default::default(), cache: None,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: agent_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: height_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: resource_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: config_buffer.as_entire_binding() },
                ],
            });

            Self { device, queue, pipeline, bind_group, agent_buffer, staging_buffer, resource_buffer, staging_resource_buffer, agent_count: agents.len() as u32 }
        })
    }

    pub fn compute_ticks(&self, ticks: usize) {
        let chunk_size = 250;
        let mut remaining = ticks;
        
        while remaining > 0 {
            let current_ticks = remaining.min(chunk_size);
            remaining -= current_ticks;
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            for _ in 0..current_ticks {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                cpass.set_pipeline(&self.pipeline);
                cpass.set_bind_group(0, &self.bind_group, &[]);
                cpass.dispatch_workgroups((self.agent_count as f32 / 64.0).ceil() as u32, 1, 1);
            }
            self.queue.submit(Some(encoder.finish()));
        }
    }

    pub fn update_agents(&self, agents: &[crate::agent::Person]) {
        self.queue.write_buffer(&self.agent_buffer, 0, bytemuck::cast_slice(agents));
    }

    pub fn update_resources(&self, resources: &[f32]) {
        self.queue.write_buffer(&self.resource_buffer, 0, bytemuck::cast_slice(resources));
    }

    pub fn fetch_state(&self) -> (Vec<crate::agent::Person>, Vec<f32>) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&self.agent_buffer, 0, &self.staging_buffer, 0, self.staging_buffer.size());
        encoder.copy_buffer_to_buffer(&self.resource_buffer, 0, &self.staging_resource_buffer, 0, self.staging_resource_buffer.size());
        self.queue.submit(Some(encoder.finish()));

        let slice = self.staging_buffer.slice(..);
        let res_slice = self.staging_resource_buffer.slice(..);
        
        let (tx, rx) = std::sync::mpsc::channel();
        let tx1 = tx.clone();
        
        slice.map_async(wgpu::MapMode::Read, move |v| tx1.send(v).unwrap());
        res_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let agents: Vec<crate::agent::Person> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.staging_buffer.unmap();
        
        let res_data = res_slice.get_mapped_range();
        let resources: Vec<f32> = bytemuck::cast_slice(&res_data).to_vec();
        drop(res_data);
        self.staging_resource_buffer.unmap();
        
        (agents, resources)
    }
}