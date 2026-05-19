# ==============================================================================
# Production Script: The Metaphorical Void (Testing Differentiability)
# Architecture: Sentence-Transformers (all-MiniLM-L6-v2) + FAISS (IndexFlatIP)
# Optimisation: Fully vectorised operations, explicit memory management
# ==============================================================================


# 1. Pipeline Dependency Installation
# ------------------------------------------------------------------------------
import sys
import subprocess


def install_packages():
    """Installs required production libraries silently within the Colab runtime."""
    print("[INFO] Initializing runtime environments and installing dependencies...")
    dependencies = ["sentence-transformers", "faiss-cpu", "pandas", "matplotlib", "tqdm"]
    for package in dependencies:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
    print("[INFO] All dependencies successfully provisioned.")


# Execute installation automatically if running in interactive environment
try:
    import sentence_transformers
    import faiss
except ImportError:
    install_packages()


# 2. System Imports
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import faiss
import matplotlib.pyplot as plt
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# Set global seed for reproducibility
np.random.seed(42)


# 3. Core Geometric Utilities
# ------------------------------------------------------------------------------
def slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    """
    Computes Spherical Linear Interpolation (Slerp) between two normalized vectors.
    Ensures uniform velocity traversal along the unit hypersphere geodesic.
    """
    dot = np.dot(v0, v1)
    # Clip bounds to eliminate potential floating-point numerical drift
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)


    if np.isclose(theta, 0.0):
        return v0


    sin_theta = np.sin(theta)
    v_t = (np.sin((1.0 - t) * theta) * v0 + np.sin(t * theta) * v1) / sin_theta
    # Return explicit unit norm to maintain hyperspherical boundary
    return v_t / np.linalg.norm(v_t)


# 4. Production Dataset & Corpus Engineering
# ------------------------------------------------------------------------------
def generate_experimental_datasets():
    """Generates evaluation anchor pairs and a diverse evaluation corpus background."""
    print("[INFO] Constructing evaluation datasets and synthetic corpus layers...")


    # 10 Multi-domain anchor pairs (Literal vs Metaphorical Equivalents)
    anchors = [
        {"domain": "Mechanics/Bureaucracy", "lit": "The machine ceased to function due to broken gears", "meta": "The bureaucratic system broke down entirely"},
        {"domain": "Structural/Project Management", "lit": "The concrete structure hit a solid brick wall", "meta": "The strategic project hit a brick wall"},
        {"domain": "Thermodynamics/Social", "lit": "The boiling liquid overflowed from the container", "meta": "The heated argument boiled over during the meeting"},
        {"domain": "Navigation/Strategy", "lit": "The ship lost its compass and drifted into deep ocean", "meta": "The company lost its direction in the market"},
        {"domain": "Meteorology/Emotion", "lit": "Dark rain clouds blocked the sun completely", "meta": "A deep depression clouded his judgment completely"},
        {"domain": "Botany/Economics", "lit": "The plant roots withered away from lack of water", "meta": "The economic foundations withered during the hyperinflation"},
        {"domain": "Medical/Infrastructure", "lit": "The blocked artery stopped the blood circulation", "meta": "The cyberattack paralyzed the digital infrastructure network"},
        {"domain": "Zoology/Predation", "lit": "The apex predator hunted its prey across the savanna", "meta": "The corporate raider targeted the vulnerable startup company"},
        {"domain": "Combustion/Creativity", "lit": "The dry firewood caught fire and burned brightly", "meta": "His creative imagination sparked a brilliant new artistic movement"},
        {"domain": "Geology/Stability", "lit": "The earthquake fractured the solid bedrock foundation", "meta": "The sudden financial scandal fractured the political coalition"}
    ]


# Diverse high-density reference background corpus
    corpus = [
        "The internal combustion engine requires precise mechanical timing.",
        "Administrative processing delays are inherent to federal organizations.",
        "Structural engineering enforces strict safety limits on concrete barriers.",
        "Enterprise task management pipelines require agile sprint planning.",
        "Industrial boilers maintain equilibrium via pressure relief valves.",
        "Emotional outbursts in corporate environments degrade cultural cohesion.",
        "Maritime navigation maps require accurate magnetic variance calibration.",
        "Global venture capital distributions shifted toward emerging markets.",
        "Barometric pressure drops indicate imminent thunderstorm activity.",
        "Cognitive behavioral therapy mitigates chronic clinical depression.",
        "Agricultural irrigation systems optimize topsoil hydration metrics.",
        "Fiscal monetary interventions were deployed to combat currency collapse.",
        "Cardiovascular diseases are worsened by lipid plaque accumulations.",
        "Distributed network firewalls block malicious packet injections.",
        "Biological diversity indexes drop within fragmented forest ecosystems.",
        "Hostile takeovers are heavily regulated by the Securities and Exchange Commission.",
        "Chemical catalysts accelerate thermodynamic reaction velocities.",
        "Avant-garde literary publications transformed mid-century poetic philosophy.",
        "Tectonic plate subduction zones generate frequent seismic events.",
        "Bipartisan legislative negotiations collapsed over budgetary allocation disputes."
        "Quantum encryption protocols rely on entangled photon transmission integrity.",
        "Municipal wastewater treatment facilities monitor bacterial contamination thresholds.",
        "Artificial intelligence inference engines optimize tensor computation workloads.",
        "Renewable energy storage depends heavily on lithium-ion battery efficiency.",
        "International trade agreements influence regional manufacturing competitiveness.",
        "Acoustic resonance frequencies can destabilize poorly reinforced structures.",
        "Neural signal propagation governs reflexive muscular coordination responses.",
        "Supply chain bottlenecks increase operational costs across retail industries.",
        "Satellite telemetry systems transmit orbital positioning data continuously.",
        "Thermodynamic entropy increases within isolated closed-loop environments.",
        "Urban population density impacts transportation infrastructure scalability.",
        "Microprocessor fabrication requires contamination-free semiconductor cleanrooms.",
        "Judicial precedent shapes constitutional interpretation within appellate courts.",
        "Machine learning classifiers improve through iterative gradient optimization.",
        "Hydroelectric turbines convert kinetic water flow into electrical energy.",
        "Economic recessions frequently reduce consumer discretionary spending habits.",
        "Autonomous vehicle navigation depends on real-time sensor fusion analysis.",
        "Environmental pollution accelerates coral reef bleaching across tropical oceans.",
        "Psychological stress contributes to elevated cortisol hormone production.",
        "Aviation control towers coordinate airspace traffic using radar synchronization.",
        "Biometric authentication systems validate identity through retinal pattern analysis.",
        "Nanotechnology research focuses on molecular-scale material engineering.",
        "Oceanic currents regulate global climate distribution and heat transfer.",
        "Cloud computing platforms allocate virtualized resources dynamically.",
        "Forensic investigators reconstruct digital evidence from corrupted storage devices.",
        "Geothermal reservoirs provide sustainable alternatives to fossil fuel dependency.",
        "Cryptographic hash collisions undermine distributed ledger verification models.",
        "Nutritional deficiencies impair long-term immune system resilience.",
        "Industrial automation reduces repetitive labor through robotic assembly lines.",
        "Electromagnetic interference disrupts unshielded communication hardware performance.",
        "Archaeological excavations uncovered artifacts from pre-Roman civilizations.",
        "Medical imaging technologies detect abnormalities within internal organ systems.",
        "Data compression algorithms reduce redundant binary storage requirements.",
        "Behavioral economics examines irrational decision-making under uncertainty.",
        "High-frequency trading systems execute transactions within microsecond intervals.",
        "Volcanic ash clouds interfere with commercial aviation routing schedules.",
        "Protein synthesis occurs through ribosomal translation of genetic sequences.",
        "Cybersecurity audits identify privilege escalation vulnerabilities in enterprise networks.",
        "Atmospheric carbon concentrations continue rising due to industrial emissions.",
        "Robotic prosthetics enhance mobility for patients with spinal injuries.",
        "Financial derivatives amplify systemic exposure during market instability.",
        "Digital transformation initiatives modernize legacy enterprise infrastructure.",
        "Marine ecosystems depend on balanced predator-prey population dynamics.",
        "Optical fiber networks enable high-bandwidth transcontinental communications.",
        "Behavioural conditioning influences long-term consumer purchasing patterns.",
        "Advanced metallurgy improves corrosion resistance in aerospace components.",
        "Social polarization intensifies during periods of economic instability.",
        "Pharmaceutical trials undergo rigorous statistical validation procedures.",
        "Hydraulic systems transfer mechanical force using pressurized fluid channels.",
        "Wireless communication standards evolve alongside consumer device ecosystems.",
        "Deforestation contributes significantly to regional biodiversity collapse.",
        "Genetic mutations alter cellular replication and protein functionality.",
        "Predictive analytics models forecast customer retention probabilities.",
        "Political instability discourages foreign direct investment initiatives.",
        "Augmented reality interfaces blend digital assets with physical environments.",
        "Industrial refrigeration units regulate temperature-sensitive storage facilities.",
        "Cultural anthropology studies ritualistic behavior across indigenous populations.",
        "Space exploration missions require redundant life-support engineering systems.",
        "Algorithmic bias can distort automated decision-making frameworks.",
        "Meteorological satellites track hurricane formation across ocean basins.",
        "Information asymmetry distorts competitive equilibrium within financial markets.",
        "Audio signal processing removes unwanted frequencies from broadcast transmissions.",
        "Antibiotic resistance threatens modern infectious disease treatment protocols.",
        "Hydrogen fuel cells generate electricity through electrochemical reactions.",
        "Constitutional democracies depend on transparent electoral accountability mechanisms.",
        "Digital currencies challenge traditional central banking monetary policies.",
        "Fracture mechanics evaluate material stress tolerance under extreme loads.",
        "Renewable agriculture practices preserve long-term soil fertility stability.",
        "Network latency issues degrade multiplayer gaming synchronization accuracy.",
        "Machine vision systems identify manufacturing defects through image recognition.",
        "Ethical philosophy debates the boundaries of artificial consciousness.",
        "Severe drought conditions reduce agricultural crop yield projections.",
        "Biochemical pathways regulate metabolic energy conversion processes.",
        "Smart grid infrastructure balances electrical demand across urban regions.",
        "Virtual reality simulations enhance immersive educational training environments.",
        "Monetary inflation reduces household purchasing power over extended periods.",
        "Antarctic ice shelf melting contributes to global sea level rise.",
        "Computational linguistics analyzes semantic relationships within natural language.",
        "Therapeutic rehabilitation programs improve post-traumatic recovery outcomes.",
        "High-altitude jet streams influence continental weather system movement.",
        "Epidemiological studies monitor disease transmission across dense populations.",
        "Blockchain consensus mechanisms validate decentralized transactional integrity.",
        "Humanitarian aid organizations coordinate relief during natural disasters.",
        "Astrophysical observations reveal anomalies in galactic rotational velocities.",
        "Digital surveillance technologies raise concerns regarding personal privacy rights.",
        "Electrochemical batteries degrade gradually through repeated charge cycles.",
        "Renewable wind farms require optimal turbine spacing configurations.",
        "Sociological research examines behavioral adaptation within urban communities.",
        "Automated logistics systems optimize warehouse inventory distribution workflows.",
        "Subatomic particle collisions generate measurable quantum field disturbances.",
        "Telecommunications infrastructure supports global financial transaction networks.",
        "Behavioural neuroscience investigates cognition through neural activity mapping.",
        "Adaptive cybersecurity frameworks respond dynamically to emerging threats.",
        "Climate migration patterns increase pressure on metropolitan housing markets.",
        "Ocean desalination facilities convert seawater into potable drinking supplies.",
        "Artificial neural architectures emulate pattern recognition within biological systems.",
        "International diplomacy seeks peaceful resolution to territorial disputes.",
        "Precision agriculture integrates satellite imaging with irrigation automation.",
        "Renewable biomass conversion reduces dependence on nonrenewable fuel reserves.",
        "Macroeconomic indicators forecast industrial production and labor market activity.",
        "Philosophical theology explores the intersection of morality and consciousness.",
        "Semiconductor shortages disrupted global electronics manufacturing supply chains.",
        "Advanced robotics research emphasizes autonomous collaborative machine behavior.",
        "Electrophysiological monitoring tracks abnormal cardiac rhythm irregularities."
    ]




    # Inject filler text to simulate database density scale
    extended_corpus = corpus + [f"Random baseline data point tracking signal index {i}" for i in range(500)]


    return anchors, extended_corpus


# 5. Core Execution Engine
# ------------------------------------------------------------------------------
def execute_topology_experiment():
    # Initialize embedding architecture
    print("[INFO] Downloading and loading sentence-transformer 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('all-MiniLM-L6-v2')


    # Load data
    anchors, corpus = generate_experimental_datasets()


    # Encode and normalize background corpus
    print(f"[INFO] Computing vector footprints for reference database (Size: {len(corpus)})...")
    corpus_embeddings = model.encode(corpus, show_progress_bar=False)
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)


    # Initialize FAISS index utilizing Inner Product (Cosine distance on normalized vectors)
    dimension = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(corpus_embeddings.astype('float32'))


    steps = 10
    k_neighbors = 2
    results_payload = []


    print("\n" + "="*80)
    print("RUNNING SLERP TRAJECTORY DISCONTINUITY ANALYSIS")
    print("="*80)


    # Iterate across experimental pairs
    for pair_idx, pair in enumerate(anchors):
        print(f"\n[Domain {pair_idx + 1}]: {pair['domain']}")


        # Extract and normalize boundary vectors
        v0 = model.encode(pair['lit'])
        v1 = model.encode(pair['meta'])
        v0 = v0 / np.linalg.norm(v0)
        v1 = v1 / np.linalg.norm(v1)


        previous_v0_sim = 1.0 # t=0 starting similarity


        for s in range(1, steps + 1):
            t = s / (steps + 1)
            v_t = slerp(v0, v1, t)


            # Query vector database
            distances, indices = index.search(np.expand_dims(v_t, axis=0).astype('float32'), k_neighbors)
            mean_density = float(np.mean(distances[0]))


            # Compute Monotonicity Derivates
            current_v0_sim = float(np.dot(v_t, v0))
            smoothness_delta = abs(current_v0_sim - previous_v0_sim)
            previous_v0_sim = current_v0_sim


            # Extract nearest text string matches
            nearest_match_str = corpus[indices[0][0]]


            results_payload.append({
                "pair_id": pair_idx,
                "domain": pair['domain'],
                "t": t,
                "local_density": mean_density,
                "smoothness_delta": smoothness_delta,
                "closest_match": nearest_match_str,
                "closest_sim": float(distances[0][0])
            })


            # Log periodic steps to console for real-time visibility
            if s in [1, 5, 10]:
                print(f"  └─ Step t={t:.2f} | Density D(vt)={mean_density:.4f} | Delta={smoothness_delta:.4f}")
                print(f"     Top Match: \"{nearest_match_str[:70]}...\" (Sim: {distances[0][0]:.4f})")


    return pd.DataFrame(results_payload)


# 6. Diagnostics Reporting & Visualization Engine
# ------------------------------------------------------------------------------
def generate_diagnostic_analytics(df: pd.DataFrame):
    """Parses results dataframe and builds analytic charts for structural validation."""
    print("\n" + "="*80)
    print("METRIC AGGREGATION & SYSTEM DIAGNOSTICS")
    print("="*80)


    # Aggregate trajectories across all domains
    summary_stats = df.groupby('t').agg({
        'local_density': ['mean', 'std'],
        'smoothness_delta': ['mean', 'std']
    }).reset_index()


    # Format MultiIndex columns manually for cleaner downstream handling
    summary_stats.columns = ['t', 'mean_density', 'std_density', 'mean_delta', 'std_delta']


    # Render Structural Plots
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))


    # Plot 1: Local Semantic Density Tracking (Manifold Uniformity Check)
    ax[0].errorbar(summary_stats['t'], summary_stats['mean_density'], yerr=summary_stats['std_density'],
                   fmt='-o', color='darkblue', ecolor='lightblue', elinewidth=3, capsize=0)
    ax[0].set_title("Local Semantic Density $D(v_t)$ Across Geodesic Path", fontsize=12)
    ax[0].set_xlabel("Slerp Interpolation Vector Parameter (t)")
    ax[0].set_ylabel("Mean k-NN Cosine Similarity")
    ax[0].grid(True, linestyle="--", alpha=0.6)


    # Plot 2: First Derivative Monotonicity Tracking (Smoothness Check)
    ax[1].errorbar(summary_stats['t'], summary_stats['mean_delta'], yerr=summary_stats['std_delta'],
                   fmt='-s', color='darkred', ecolor='pink', elinewidth=3, capsize=0)
    ax[1].set_title("First-Derivative Monotonicity $\Delta \Phi(t)$", fontsize=12)
    ax[1].set_xlabel("Slerp Interpolation Vector Parameter (t)")
    ax[1].set_ylabel("Absolute Shift Value from Anchor v0")
    ax[1].grid(True, linestyle="--", alpha=0.6)


    plt.tight_layout()
    plt.show()


    # Final Topology Assessment Output
    overall_density_variance = df['local_density'].std()
    print(f"\n[ANALYSIS REPORT]")
    print(f"-> Calculated Vector Path Spatial Variance: {overall_density_variance:.6f}")
    if overall_density_variance > 0.05:
        print("-> VERDICT: ANISOTROPIC COLLAPSE OBSERVED. Significant geometric voids identified within the linguistic manifold path.")
    else:
        print("-> VERDICT: SMOOTH CONTINUITY MAINTAINED. Local neighborhood density shows consistent geometric structural distribution.")


# 7. Main Execution Guard
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    results_df = execute_topology_experiment()
    generate_diagnostic_analytics(results_df)

