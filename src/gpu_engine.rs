use wgpu::util::DeviceExt;
use pollster::block_on;

pub struct GpuEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    agent_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    agent_count: u32,
}

impl GpuEngine {
    pub fn new(agents: &[crate::agent::Person], map_mask: &[f32]) -> Self {
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

            let mask_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Mask Buffer"),
                contents: bytemuck::cast_slice(map_mask),
                usage: wgpu::BufferUsages::STORAGE,
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
                    wgpu::BindGroupEntry { binding: 1, resource: mask_buffer.as_entire_binding() },
                ],
            });

            Self { device, queue, pipeline, bind_group, agent_buffer, staging_buffer, agent_count: agents.len() as u32 }
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

    pub fn fetch_agents(&self) -> Vec<crate::agent::Person> {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&self.agent_buffer, 0, &self.staging_buffer, 0, self.staging_buffer.size());
        self.queue.submit(Some(encoder.finish()));

        let slice = self.staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let agents: Vec<crate::agent::Person> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.staging_buffer.unmap();
        agents
    }
}