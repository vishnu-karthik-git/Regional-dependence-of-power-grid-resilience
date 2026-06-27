# Regional-dependence-of-power-grid-resilience
## High Voltage Grid Resilience Analysis with SimBench

A power systems analysis project that inspects SimBench high voltage (HV) grids, runs powerflow simulations across load/production snapshots, quantifies grid resilience, and compares urban vs. mixed HV topologies.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
  - [1. Grid Inspection & Snapshot Analysis](#1-grid-inspection--snapshot-analysis)
  - [2. Powerflow Calculations](#2-powerflow-calculations)
  - [3. Resilience Quantification](#3-resilience-quantification)
  - [4. Graph-Theoretic Centrality Measures](#4-graph-theoretic-centrality-measures)
  - [5. Comparison of HV Grids](#5-comparison-of-hv-grids)
- [Results](#results)
- [Discussion](#discussion)
- [References](#references)

---

## Project Overview

This project analyzes the **resilience of high voltage (HV) transmission grids** using the [SimBench](https://simbench.de/en/) benchmark dataset. The analysis proceeds in three phases:

1. **Grid Inspection & Temporal Analysis** — Explore multiple SimBench HV grids (no-switch variants), select one, and visualize how load and generation snapshots evolve over a day, week, and full year.
2. **Powerflow-Based Resilience Assessment** — Run AC powerflow calculations for every (or a representative subset of) snapshots; derive resilience indicators such as line loading, voltage deviations, and system reserve margins.
3. **Grid Comparison** — Repeat the analysis on a second HV grid from a different region (e.g., urban vs. mixed) and compare structural and operational differences.

---

## Project Structure

```
hv-grid-resilience/
│
├── data/                        # Raw SimBench grid data (auto-downloaded)
│
├── notebooks/
│   ├── 01_grid_inspection.ipynb         # Inspect & compare HV grids
│   ├── 02_snapshot_analysis.ipynb       # Load/generation time series plots
│   ├── 03_powerflow_calculations.ipynb  # AC powerflow over all snapshots
│   ├── 04_resilience_metrics.ipynb      # Resilience quantification
│   ├── 05_centrality_measures.ipynb     # Graph-theoretic centrality analysis
│   └── 06_grid_comparison.ipynb        # Urban vs. mixed HV grid comparison
│
├── results/
│   ├── powerflow/               # Stored powerflow results (CSV/pickle)
│   ├── centrality/              # Centrality measure outputs
│   └── figures/                 # All generated plots (PNG/SVG)
│
├── src/
│   ├── grid_utils.py            # Grid loading and topology helpers
│   ├── powerflow.py             # Powerflow batch runner & result storage
│   ├── resilience.py            # Resilience metric calculations
│   ├── centrality.py            # Graph centrality wrappers (NetworkX)
│   └── plotting.py              # Reusable plotting functions
│
├── requirements.txt
└── README.md
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `pandapower` | Power system modelling & AC powerflow |
| `simbench` | SimBench grid loader |
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualisation |
| `networkx` | Graph-theoretic centrality measures |
| `jupyter` | Interactive notebooks |

Python 3.9+ is recommended.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/hv-grid-resilience.git
cd hv-grid-resilience

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

Run notebooks in order, or execute the source scripts directly:

```bash
# Launch Jupyter
jupyter notebook notebooks/

# Or run a specific analysis script
python src/powerflow.py --grid "1-HV-urban--2-no_sw" --output results/powerflow/
```

To run powerflow on a representative subset (faster, useful for large grids):

```python
# In src/powerflow.py — set SUBSET_MODE = True and SUBSET_SIZE = 96 (e.g., one week)
SUBSET_MODE = True
SUBSET_SIZE = 672   # 7 days × 96 snapshots/day
```

---

## Methodology

### 1. Grid Inspection & Snapshot Analysis

SimBench provides several HV grids distinguished by region and switch configuration. Only **`no_sw` variants** (no explicit switch modelling) are considered here, including:

- `1-HV-urban--2-no_sw`
- `1-HV-mixed--2-no_sw`
- `1-HV-rural--2-no_sw`

Key structural properties compared across grids:

- Number of buses, lines, and transformers
- Nominal voltage levels
- Network topology (radial vs. meshed)
- Total installed load and generation capacity

The selected grid's **35,040 snapshots** (15-minute resolution, full year) are analysed temporally:

- **Daily profile** — typical intraday load/generation patterns
- **Weekly profile** — weekday vs. weekend variation
- **Annual profile** — seasonal trends and peak periods

### 2. Powerflow Calculations

AC powerflow is computed for every snapshot (or a representative subset) using `pandapower`:

```python
import pandapower as pp
import simbench as sb

net = sb.get_simbench_net("1-HV-urban--2-no_sw")
sb.apply_const_controllers(net, profiles)
pp.runpp(net)
```

Results stored per snapshot:

- Bus voltages (p.u.)
- Line loading (% of thermal rating)
- Transformer loading (%)
- Active/reactive power flows

### 3. Resilience Quantification

Resilience indicators derived from powerflow results:

| Metric | Description | Ideal Value |
|---|---|---|
| **Max line loading** | Peak % utilisation across all lines | < 80% |
| **Voltage deviation** | Max \|V − 1.0\| across buses (p.u.) | < 0.05 p.u. |
| **System reserve margin** | 1 − (max line loading / 100) | Higher = more resilient |
| **N−1 feasibility** | Fraction of snapshots where all N−1 contingencies are feasible | → 1.0 |
| **Overload frequency** | Fraction of snapshots with any line > 100% loading | → 0.0 |

The **system reserve margin** is the primary scalar resilience indicator: it captures how far the most stressed component is from its operating limit, averaged or assessed at peak across all snapshots.

### 4. Graph-Theoretic Centrality Measures

The HV grid is modelled as an undirected weighted graph (nodes = buses, edges = lines/transformers, weights = line impedance or admittance). The following centrality measures are computed using `networkx`:

| Measure | Interpretation in Power Systems |
|---|---|
| **Degree centrality** | Highly connected buses (potential single points of failure) |
| **Betweenness centrality** | Buses/lines critical for power transit paths |
| **Closeness centrality** | Buses that are electrically "close" to the rest of the network |
| **Eigenvector centrality** | Buses connected to other well-connected buses |

**Suitability as resilience quantifiers:**

- Betweenness centrality correlates well with line loading at high-stress nodes, making it a useful *static* proxy for resilience without requiring powerflow.
- However, centrality measures are topology-only — they ignore impedances, generation dispatch, and operating limits. Powerflow-derived metrics are therefore more physically accurate but computationally heavier.
- Centrality measures are best used as *screening tools* to identify vulnerable nodes before detailed powerflow analysis.

### 5. Comparison of HV Grids

A second grid (e.g., `1-HV-mixed--2-no_sw`) is analysed with the same pipeline. Comparison dimensions:

- **Structural** — topology, connectivity, redundancy paths
- **Operational** — loading distributions, voltage profiles, peak stress snapshots
- **Resilience** — reserve margins, overload frequencies, centrality distributions
- **Temporal** — differences in seasonal and daily patterns linked to regional demand/generation mix

---

## Results

Key outputs stored in `results/`:

- `results/powerflow/` — Per-snapshot line loadings and bus voltages (parquet/CSV)
- `results/centrality/` — Centrality scores per bus (CSV)
- `results/figures/` — Plots including:
  - Grid topology visualisations
  - Daily / weekly / annual load-generation profiles
  - Line loading heatmaps (time × line)
  - Voltage deviation distributions
  - Centrality score bar charts
  - Urban vs. mixed grid comparison figures

---

## Discussion

- **SimBench HV grids** share the same nominal voltage (110 kV) but differ substantially in topology: urban grids tend to be more meshed (higher redundancy), while rural grids are more radial (higher vulnerability to single-line failures).
- **Snapshot analysis** reveals strong diurnal and seasonal patterns driven by residential and industrial demand cycles, as well as renewable generation variability.
- **Powerflow results** show that the system is most stressed during winter evening peaks and periods of low renewable output, when reserve margins are smallest.
- **Centrality vs. powerflow metrics**: Betweenness centrality is a reasonable proxy for identifying critical assets, but it overestimates the importance of topologically central nodes that carry little actual power flow due to impedance characteristics. Powerflow-based reserve margins remain the gold standard for operational resilience assessment.

---

## References

- SimBench: [https://simbench.de/en/](https://simbench.de/en/)
- Meinecke, S. et al. (2020). *SimBench — A Benchmark Dataset of Electric Power Systems to Compare Powerflow Analysis Algorithms*. Energies.
- pandapower: [https://pandapower.readthedocs.io/](https://pandapower.readthedocs.io/)
- NetworkX: [https://networkx.org/](https://networkx.org/)
- Zimmerman, R.D. et al. *MATPOWER: Steady-State Operations, Planning, and Analysis Tools for Power Systems Research and Education*. IEEE Transactions on Power Systems.
