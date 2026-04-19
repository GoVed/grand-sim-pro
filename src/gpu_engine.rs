/*
 * Grand Sim Pro: A high-performance GPGPU evolutionary agent simulation.
 * Part of an independent research project into emergent biological complexity.
 *
 * Copyright (C) 2026 Ved Hirenkumar Suthar
 * Licensed under the GNU General Public License v3.0 or later.
 * * This software is provided "as is", without warranty of any kind.
 * See the LICENSE file in the project root for full license details.
 */

use wgpu::util::DeviceExt;
use pollster::block_on;
use std::sync::Mutex;

pub struct GpuEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    clear_pipeline: wgpu::ComputePipeline,
    world_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    world_bind_group: wgpu::BindGroup,
    render_bind_group: wgpu::BindGroup,
    agent_state_buffer: wgpu::Buffer,
    genetics_buffer: wgpu::Buffer,
    staging_state_buffer: wgpu::Buffer,
    staging_cell_buffer: wgpu::Buffer,
    cell_buffer: wgpu::Buffer,
    render_buffer: wgpu::Buffer,
    staging_render_buffer: wgpu::Buffer,
    config_buffer: wgpu::Buffer,
    height_buffer: wgpu::Buffer,
    agent_count: u32,
    map_width: u32,
    map_height: u32,
    internal_tick: Mutex<u32>,
    staging_lock: Mutex<()>,
}

impl GpuEngine {
    pub fn new(states: &[crate::agent::AgentState], genetics: &[crate::agent::Genetics], map_heights: &[f32], map_cells: &[crate::environment::CellState], config: &crate::config::SimConfig) -> Self {
        block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::PRIMARY,
                ..Default::default()
            });
            let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
            let (device, queue) = adapter.request_device(
                &wgpu::DeviceDescriptor {
                    required_limits: adapter.limits(),
                    ..Default::default()
                },
                None,
            ).await.unwrap();

            let agent_state_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Agent State Buffer"),
                contents: bytemuck::cast_slice(states),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            });

            let genetics_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Genetics Buffer"),
                contents: bytemuck::cast_slice(genetics),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            });

            let staging_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging State Buffer"),
                size: (states.len() * std::mem::size_of::<crate::agent::AgentState>()) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let staging_cell_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Cell Buffer"),
                size: (map_cells.len() * std::mem::size_of::<crate::environment::CellState>()) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let height_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Height Buffer"),
                contents: bytemuck::cast_slice(map_heights),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            
            let cell_bytes = bytemuck::cast_slice(map_cells);
            let cell_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Cell Buffer"),
                contents: cell_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            });

            let render_buffer_size = (config.world.map_width * config.world.map_height * 4) as wgpu::BufferAddress;
            let render_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Render Buffer"),
                size: render_buffer_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            
            let staging_render_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Render Buffer"),
                size: render_buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            
            let config_bytes = bytemuck::bytes_of(config);
            let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Config Buffer"),
                contents: config_bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            let shader_src = format!(
                "{}\n{}\n{}\n{}\n{}",
                include_str!("shaders/types.wgsl"),
                include_str!("shaders/bindings.wgsl"),
                include_str!("shaders/sim.wgsl"),
                include_str!("shaders/world.wgsl"),
                include_str!("shaders/render.wgsl")
            );
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Sim Shader"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(shader_src)),
            });
            
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Sim Pipeline"), layout: None, module: &shader, entry_point: "main",
                compilation_options: Default::default(), cache: None,
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Shared Layout"),
                bind_group_layouts: &[&pipeline.get_bind_group_layout(0)],
                push_constant_ranges: &[],
            });

            let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Clear Pipeline"), layout: Some(&pipeline_layout), module: &shader, entry_point: "clear_main",
                compilation_options: Default::default(), cache: None,
            });

            let world_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("World Pipeline"), layout: None, module: &shader, entry_point: "world_main",
                compilation_options: Default::default(), cache: None,
            });
            
            let render_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Render Pipeline"), layout: None, module: &shader, entry_point: "render_main",
                compilation_options: Default::default(), cache: None,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Sim Bind Group"), layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: agent_state_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: height_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: cell_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: config_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: genetics_buffer.as_entire_binding() },
                ],
            });

            let world_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("World Bind Group"), layout: &world_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 1, resource: height_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: cell_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: config_buffer.as_entire_binding() },
                ],
            });

            let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Render Bind Group"), layout: &render_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 1, resource: height_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: cell_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: config_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: render_buffer.as_entire_binding() },
                ],
            });

            Self {
                device, queue, pipeline, clear_pipeline, world_pipeline, render_pipeline, bind_group, world_bind_group, render_bind_group,
                agent_state_buffer, genetics_buffer, staging_state_buffer,
                staging_cell_buffer, cell_buffer, render_buffer, staging_render_buffer,
                config_buffer, height_buffer, agent_count: states.len() as u32,
                map_width: config.world.map_width,
                map_height: config.world.map_height,
                internal_tick: Mutex::new(0),
                staging_lock: Mutex::new(()),
            }
        })
    }

    pub fn wait_idle(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }

    pub fn compute_ticks(&self, ticks: u32) {
        if ticks == 0 { return; }
        
        // Chunk dispatches to prevent GPU hangs/TDR (Timeout Detection and Recovery)
        // Drivers often kill contexts that take > 2 seconds.
        let max_ticks_per_submission = 250;
        let mut remaining = ticks;

        let mut internal_tick = self.internal_tick.lock().unwrap();

        while remaining > 0 {
            let current_batch = remaining.min(max_ticks_per_submission);
            remaining -= current_batch;

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                for _ in 0..current_batch {
                    // 0. Clear Population (Fast, Agent Count Threads)
                    cpass.set_pipeline(&self.clear_pipeline);
                    cpass.set_bind_group(0, &self.bind_group, &[]);
                    cpass.dispatch_workgroups((self.agent_count as f32 / 64.0).ceil() as u32, 1, 1);

                    // 1. Agent Simulation
                    cpass.set_pipeline(&self.pipeline);
                    cpass.set_bind_group(0, &self.bind_group, &[]);
                    cpass.dispatch_workgroups((self.agent_count as f32 / 64.0).ceil() as u32, 1, 1);

                    // 2. World Environment Update (Slow, Every 100 Ticks)
                    if *internal_tick % 100 == 0 {
                        cpass.set_pipeline(&self.world_pipeline);
                        cpass.set_bind_group(0, &self.world_bind_group, &[]);
                        // Use 2D dispatch to avoid exceeding 65535 limit on a single dimension
                        cpass.dispatch_workgroups((self.map_width as f32 / 8.0).ceil() as u32, (self.map_height as f32 / 8.0).ceil() as u32, 1);
                    }

                    *internal_tick = internal_tick.wrapping_add(1);
                }
            }
            self.queue.submit(Some(encoder.finish()));
            
            // In high-load batches, occasionally poll to let the driver stay alive
            if remaining > 0 {
                self.device.poll(wgpu::Maintain::Poll);
            }
        }
    }

    pub fn update_agents(&self, states: &[crate::agent::AgentState], genetics: &[crate::agent::Genetics]) {
        self.queue.write_buffer(&self.agent_state_buffer, 0, bytemuck::cast_slice(states));
        self.queue.write_buffer(&self.genetics_buffer, 0, bytemuck::cast_slice(genetics));
    }

    pub fn update_config(&self, config: &crate::config::SimConfig) {
        self.queue.write_buffer(&self.config_buffer, 0, bytemuck::bytes_of(config));
    }

    pub fn update_heights(&self, heights: &[f32]) {
        self.queue.write_buffer(&self.height_buffer, 0, bytemuck::cast_slice(heights));
    }

    pub fn update_cells(&self, cells: &[crate::environment::CellState]) {
        self.queue.write_buffer(&self.cell_buffer, 0, bytemuck::cast_slice(cells));
    }

    pub fn fetch_agents(&self) -> Vec<crate::agent::AgentState> {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&self.agent_state_buffer, 0, &self.staging_state_buffer, 0, self.staging_state_buffer.size());
        self.queue.submit(Some(encoder.finish()));

        let state_slice = self.staging_state_buffer.slice(..);

        let (tx_s, rx_s) = std::sync::mpsc::channel();

        state_slice.map_async(wgpu::MapMode::Read, move |v: Result<(), wgpu::BufferAsyncError>| tx_s.send(v).unwrap());

        self.device.poll(wgpu::Maintain::Wait);
        rx_s.recv().unwrap().unwrap();

        let state_data = state_slice.get_mapped_range();

        let states: &[crate::agent::AgentState] = bytemuck::cast_slice(&state_data);

        let states_vec = states.to_vec();

        drop(state_data);
        self.staging_state_buffer.unmap();

        states_vec
    }
    pub fn fetch_cells(&self) -> Vec<crate::environment::CellState> {
        let _guard = self.staging_lock.lock().unwrap();
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&self.cell_buffer, 0, &self.staging_cell_buffer, 0, self.staging_cell_buffer.size());
        self.queue.submit(Some(encoder.finish()));

        let slice = self.staging_cell_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v: Result<(), wgpu::BufferAsyncError>| tx.send(v).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let cells: &[crate::environment::CellState] = bytemuck::cast_slice(&data);
        let cells_vec = cells.to_vec();
        drop(data);
        self.staging_cell_buffer.unmap();
        cells_vec
    }

    pub fn fetch_render(&self, width: u32, height: u32) -> Vec<u8> {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&self.render_pipeline);
            cpass.set_bind_group(0, &self.render_bind_group, &[]);
            cpass.dispatch_workgroups((width as f32 / 16.0).ceil() as u32, (height as f32 / 16.0).ceil() as u32, 1);
        }
        encoder.copy_buffer_to_buffer(&self.render_buffer, 0, &self.staging_render_buffer, 0, self.staging_render_buffer.size());
        self.queue.submit(Some(encoder.finish()));

        let slice = self.staging_render_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v: Result<(), wgpu::BufferAsyncError>| tx.send(v).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let pixels: Vec<u8> = data.to_vec();
        drop(data);
        self.staging_render_buffer.unmap();
        pixels
    }
}
