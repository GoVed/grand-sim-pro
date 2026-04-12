@group(0) @binding(0) var<storage, read_write> agents: array<AgentState>;
@group(0) @binding(1) var<storage, read> map_heights: array<f32>;
@group(0) @binding(2) var<storage, read_write> map_cells: array<CellState>;
@group(0) @binding(3) var<uniform> cfg: SimConfig;
@group(0) @binding(4) var<storage, read_write> render_buffer: array<u32>;
@group(0) @binding(5) var<storage, read_write> genetics: array<Genetics>;

