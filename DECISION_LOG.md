# Grand Sim Pro - Cognitive Architectural Decisions

This log tracks major changes to the agent cognitive model, ensuring that every expansion is grounded in biological or psychological research rather than arbitrary scaling.

## 1. Memory Expansion (8 -> 24 Channels)
- **Scientific Basis:** **Miller's Law (1956)** & **Working Memory Capacity**.
- **Reasoning:** Human working memory is traditionally estimated at $7 \pm 2$ "chunks." In a neural context, a "chunk" (like a spatial location or a social identity) is not a single scalar but a vector. 24 channels allow the agent to store roughly 8 concepts as 3D vectors (e.g., $X, Y, \text{Importance}$).
- **Impact:** Allows for emergent "Persistence of Intent." Agents can now remember a destination, a recent threat, and an internal state simultaneously, moving from reactive "reflex" behavior to proactive "planned" behavior.

## 2. Communication Expansion (4 -> 12 Channels)
- **Scientific Basis:** **Phonemic Complexity & Multi-modal Signaling**.
- **Reasoning:** Basic animal calls (alarm, mating) use 1-3 degrees of freedom. Higher primates and humans use complex vocal tracts capable of modulating pitch, volume, timber, and specific phonemes. 12 channels provide a 12-dimensional signal space.
- **Impact:** Enables "Semantic Specialization." Evolution can now allocate specific channels for resource types, tribal identifiers, and emotional gradients without signal "crowding."

## 3. Hidden Layer Expansion (64 -> 128 Nodes)
- **Scientific Basis:** **Neural Scaling Laws for Task Complexity**.
- **Reasoning:** As the input space (Sensory) and output space (Cognitive) expand, the "Bottleneck" (hidden layer) must scale to maintain representational capacity. 128 nodes provide a $4\times$ increase in total synaptic potential ($128^2$ vs $64^2$ connections in $W_2$).
- **Impact:** Higher "Non-linear Resolution." Agents can now learn more complex "If-This-And-That-But-Not-Then" rules, necessary for managing 24 memory registers.

## 4. Competitive Intent Prioritization
- **Scientific Basis:** **Action Selection & Winner-Take-All Neural Dynamics**.
- **Reasoning:** Real organisms face metabolic and physical trade-offs. You cannot run at full speed while carefully placing a brick.
- **Impact:** Agents must now "choose" their focus. If the neural intent for Building ($I_{build}$) exceeds the intent for Moving ($I_{move}$), the agent stops. This creates realistic "work cycles" and prevents "dotted" infrastructure.

## 5. Neural Influence Normalization
- **Scientific Basis:** **Variance Scaling in Deep Networks (Xavier/He Initialization Principles)**.
- **Reasoning:** In a deep linear approximation of influence ($W_1 \times W_2 \times W_3$), the magnitude of the product scales with the width of the hidden layers ($H$). Without normalization, expanding to 128 nodes caused a $2\times$ increase in visual "noise" and weight magnitude.
- **Impact:** By dividing the influence product by $H$ at each layer step, we maintain a stable sensory-behavioral map that remains interpretable regardless of brain width.

