Automated geometric and topological analysis engine to test the differentiability of natural language embeddings using Spherical Linear Interpolation (Slerp) and FAISS neighbourhood density metrics.

# Metaphorical Void Analyser: Latent Manifold Differentiability Engine

This repository provides a validation pipeline to stress-test a foundational assumption in modern Natural Language Processing: Are semantic vector spaces truly continuously differentiable manifolds, or are they discrete, anisotropic graphs punctuated by non-semantic vacuums ("empty space")?

By tracking structural transitions along the geodesic path between literal statements (e.g., "The machine ceased to function due to broken gears") and their metaphorical equivalents ("The bureaucratic system broke down entirely"), this engine maps out the underlying topology of pre-trained language models.

# Core Methodology & Pipeline

```mermaid
graph TD;
    A[Literal Anchor v0] -->|Extract Vector| B(all-MiniLM-L6-v2)
    C[Metaphorical Anchor v1] -->|Extract Vector| B
    B --> D[Normalise to Unit Sphere]
    D --> E[Compute Geodesic via Slerp t=0.1...0.9]
    E --> F[Generate 10 Intermediate Tokens/Vectors]
    F --> G[k-NN Index Search over Vector DB]
    G --> H{Evaluate Neighborhood Metrics}
    H -->|Smooth Semantics| I[Continuous Manifold Confirmed]
    H -->|Erratic/Nonsensical Hits| J[Discrete Topology Discovered]
```

# Expected Key Diagnostic Analytics

The program automatically renders dual-axis structural charts using matplotlib to evaluate the manifold's integrity:

Manifold Uniformity Check: Monitors drops in cosine similarity over the trajectory parameter t. Sharp dips identify unindexed, non-semantic "voids.
"Smoothness Check: Validates whether the semantic drift matches a constant-velocity parallel transport or displays chaotic, non-linear jumps.

# Failure Mode Analysis

```mermaid
graph LR;
    A[Semantic Void Discovered] --> B[LLM Alignment & Guardrail Failure]
    A --> C[Industrial Automation Exploit]
    A --> D[Financial Market Instability]
    
    B --> B1[Adversarial Jailbreaks via Slerp Deflection]
    C --> C1[Unexpected Actions in Unseen Edge-Cases]
    D --> D1[Liquidity and Risk Cascades from Textual Shocks]
```
