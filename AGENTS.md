# Agent Neural Architecture Documentation

## Neural Structure Synchronization Mandate
**CRITICAL:** Whenever the neural network structure (inputs, layers, CNN kernels, or outputs) is modified in `src/agent.rs` or `src/shaders/sim.wgsl`, you **MUST** update the architectural visualization.

### How to synchronize:
1.  Ensure `tests/neural_visualization.rs` correctly reflects the new indices and labels.
2.  Run the visualization utility:
    ```bash
    cargo test --test neural_visualization
    ```
3.  Verify the generated image at `test_screenshots/neural_structure.png`.
4.  Commit the updated image along with the code changes to ensure documentation never drifts from the implementation.

This diagram is the primary source of truth for understanding the high-level cognitive pipeline of the agents.
