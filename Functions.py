# Data handling
import pandas as pd
import numpy as np
import math

# Power system modelling and SimBench benchmark grids
import pandapower as pp
import simbench as sb

# Plotting and graph analysis
import matplotlib.pyplot as plt
import igraph as ig
import json

# Suppress non-critical warnings from pandapower/simbench for cleaner output
import warnings
warnings.filterwarnings('ignore')

def import_sb_grid(sb_code):
    """
    Load a SimBench benchmark grid by its code and plot its topology.

    Parameters
    ----------
    sb_code : str
        A valid SimBench grid code (e.g. "1-HV-urban--0-no_sw").

    Returns
    -------
    pandapowerNet or None
        The loaded pandapower network, or None if the code is invalid.
    """
    if sb_code in sb.collect_all_simbench_codes():
        net_simbench = sb.get_simbench_net(sb_code)
        print(net_simbench)
        pp.plotting.simple_plot(net_simbench)
        return net_simbench
    else:
        print('Please check the name of the example')

def get_profiles(net, step, id):
    """
    Retrieve SimBench load/generation profiles for the grid and plot the
    aggregated total load vs. total generation over the year.

    Parameters
    ----------
    net : pandapowerNet
    step : int
        Downsampling factor applied to the 15-minute-resolution profiles:
        step=1 -> 15 min (raw), step=4 -> hourly, step=96 -> daily, step=672 -> weekly.

    Returns
    -------
    dict
        The full set of SimBench profiles (all element/column combinations).
    """
    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)

    print("Available profile keys:")
    for key, df_p in profiles.items():
        print(f"  {str(key):35s} shape={df_p.shape}")

    total_load = profiles[('load', 'p_mw')].sum(axis=1)
    total_gen  = profiles[('sgen', 'p_mw')].sum(axis=1)

    # Downsample
    total_load_filtered = total_load.iloc[::step]
    total_gen_filtered = total_gen.iloc[::step]

    # Create x-axis in hours
    hours = np.arange(len(total_load_filtered)) * (step * 15 / 60)
    plt.plot(hours, total_load_filtered,
    label="Total Load", color="steelblue", linewidth=0.6)
    plt.plot(hours, total_gen_filtered,
    label="Total Generation", color="darkorange", linewidth=0.6)
    plt.title(f"Full Year ({step*15} min resolution)")
    plt.xlabel("Hours")
    plt.ylabel("MW")
    
    # Force axis from 0 to 8760 hours
    plt.xlim(0, 8760)
    # Optional: major ticks every 1000 h
    plt.xticks(np.arange(0, 8761, 1000))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/load_generation_profiles_step{step}_{id}.png", dpi=150)
    plt.show()

    return profiles

def apply_snapshot(net, profiles, t):
    """Write snapshot t's load/generation/storage values directly into net's element tables."""

    # Loads
    if ("load", "p_mw") in profiles and len(net.load) > 0:
        p = profiles[("load", "p_mw")]
        if t < len(p):
            net.load["p_mw"] = p.iloc[t].values
    if ("load", "q_mvar") in profiles and len(net.load) > 0:
        q = profiles[("load", "q_mvar")]
        if t < len(q):
            net.load["q_mvar"] = q.iloc[t].values

    # Static generators (renewables, typically)
    if ("sgen", "p_mw") in profiles and len(net.sgen) > 0:
        sg = profiles[("sgen", "p_mw")]
        if t < len(sg):
            net.sgen["p_mw"] = sg.iloc[t].values

    # Conventional generators (powerplants)
    if ("gen", "p_mw") in profiles and len(net.gen) > 0:
        g = profiles[("gen", "p_mw")]
        if t < len(g):
            net.gen["p_mw"] = g.iloc[t].values

    # Storage
    if ("storage", "p_mw") in profiles and len(net.storage) > 0:
        st = profiles[("storage", "p_mw")]
        if t < len(st):
            net.storage["p_mw"] = st.iloc[t].values

    return net

def run_powerflow_over_period(net, profiles, id, start_t=0, end_t=672, step=4):
    """
    Run an AC power flow for each snapshot in [start_t, end_t) and collect
    resilience-relevant results.

    Parameters
    ----------
    net : pandapowerNet
    profiles : dict
        Output of get_profiles(), used to update `net` for each snapshot.
    start_t, end_t, step : int
        Range and step size (in profile timesteps) to iterate over.

    Returns
    -------
    df : pd.DataFrame
        One summary row per snapshot (worst line/bus, voltages, etc.), indexed
        by timestep `t`. Saved to Line_Loading.csv.
    line_loading_history : pd.DataFrame
        loading_percent for EVERY line at every converged snapshot
        (rows=t, columns=line index). Used to find the System Reserve/Burden.
    bus_power_history : pd.DataFrame
        p_mw for EVERY bus at every converged snapshot
        (rows=t, columns=bus index). Used to find the System Reserve/Burden.
    """
    records = []
    line_loading_history_records = []
    bus_power_history_records = []
    bus_voltage_history_records = []
    timesteps = range(start_t, end_t, step)
    n = len(timesteps)

    for i, t in enumerate(timesteps):
        if i % 50 == 0:
            print(f"  Progress: {i}/{n} (t={t})")

        apply_snapshot(net, profiles, t)

        try:
            pp.runpp(net, distributed_slack=False, enforce_q_lims=True, numba=False)

            ll = net.res_line["loading_percent"]
            vm = net.res_bus["vm_pu"]

            # Keep the full per-line / per-bus snapshot for the reserve & burden analysis
            line_loading_history_records.append(ll.rename(t))

            # res_bus.p_mw: positive = net consumption, negative = net injection
            bus_p = net.res_bus["p_mw"]
            bus_power_history_records.append(bus_p.rename(t))
            bus_voltage_history_records.append(vm.rename(t))

            # Worst-loaded line at this snapshot
            worst_idx  = ll.idxmax()
            worst_line = net.line.loc[worst_idx]

            max_load_bus_idx  = bus_p.idxmax()   # highest net consumer
            max_load_bus_p_mw = bus_p.max()
            min_gen_bus_idx   = bus_p.idxmin()   # highest net generator
            min_gen_bus_p_mw  = bus_p.min()

            max_load_bus_name = net.bus.loc[max_load_bus_idx, "name"] if "name" in net.bus.columns else f"bus_{max_load_bus_idx}"
            min_gen_bus_name  = net.bus.loc[min_gen_bus_idx,  "name"] if "name" in net.bus.columns else f"bus_{min_gen_bus_idx}"

            records.append({
                "t"                     : t,
                "max_line_loading"      : np.round(ll.max(), 2),
                "mean_line_loading"     : np.round(ll.mean(), 2),
                "min_line_loading"      : np.round(ll.min(), 2),
                "system_reserve_index"  : np.round(1 - ll.max() / 100, 2),   # headroom on the worst line this snapshot (1=safe, 0=at limit)
                "n_lines_over_80"       : int((ll > 80).sum()),
                "n_lines_overloaded"    : int((ll > 100).sum()),
                "min_voltage_pu"        : np.round(vm.min(), 2),
                "max_voltage_pu"        : np.round(vm.max(), 2),
                "max_voltage_deviation" : (vm - 1.0).abs().max(),
                "max_line_idx"          : worst_idx,
                "max_line_name"         : worst_line.get("name", f"line_{worst_idx}"),
                "max_line_from_bus"     : worst_line["from_bus"],
                "max_line_to_bus"       : worst_line["to_bus"],
                # Highest-load bus
                "max_load_bus_idx"      : max_load_bus_idx,
                "max_load_bus_name"     : max_load_bus_name,
                "max_load_bus_p_mw"     : np.round(max_load_bus_p_mw, 2),
                # Highest-generation bus
                "max_gen_bus_idx"       : min_gen_bus_idx,
                "max_gen_bus_name"      : min_gen_bus_name,
                "max_gen_bus_p_mw"      : np.round(abs(min_gen_bus_p_mw), 2),  # stored positive for readability
                "converged"             : True,
            })

        except pp.LoadflowNotConverged:
            # Snapshot failed to converge (e.g. an extreme load/generation combination) - flagged, not fatal
            records.append({"t": t, "converged": False})

    df = pd.DataFrame(records).set_index("t")
    print(f"Done. {df['converged'].sum()} converged, {(~df['converged']).sum()} failed.")
    df.to_csv(f'outputs/Line_Loading_{id}.csv', sep =';')

    line_loading_history = pd.DataFrame(line_loading_history_records)
    line_loading_history.to_csv(f'outputs/Line_History_{id}.csv', sep =';')
    bus_power_history = pd.DataFrame(bus_power_history_records)
    bus_power_history.to_csv(f'outputs/Bus_Power_{id}.csv', sep =';')
    bus_voltage_history = pd.DataFrame(bus_voltage_history_records)
    bus_voltage_history.to_csv(f'outputs/Bus_Voltage_{id}.csv', sep=';')

    return df, line_loading_history, bus_power_history, bus_voltage_history

def plot_line_loading_over_time(df, id = 2, title="Line Loading Over Time"):
    """
    Three-panel overview of the simulation period:
    max/mean line loading, system reserve index, and number of stressed lines.
    """
    df_ok = df[df["converged"] == True]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Panel 1 - max and mean line loading
    axes[0].plot(df_ok.index, df_ok["max_line_loading"],  label="Max",  color="red",       linewidth=1.2)
    axes[0].plot(df_ok.index, df_ok["mean_line_loading"], label="Mean", color="steelblue", linewidth=1.0, alpha=0.7)
    axes[0].fill_between(df_ok.index, df_ok["min_line_loading"], df_ok["max_line_loading"],
                         alpha=0.1, color="steelblue", label="Min-Max range")
    axes[0].axhline(80,  color="orange", linestyle="--", linewidth=0.8, label="Warning 80%")
    axes[0].axhline(100, color="red",    linestyle="--", linewidth=0.8, label="Limit 100%")
    axes[0].set_ylabel("Line loading (%)")
    axes[0].legend(fontsize=8)
    axes[0].set_title(title)

    # Panel 2 - system reserve index (snapshot-level headroom on the worst line)
    # axes[1].plot(df_ok.index, df_ok["system_reserve_index"], color="green", linewidth=1.2)
    # axes[1].axhline(0.2, color="orange", linestyle="--", linewidth=0.8, label="Low reserve (0.2)")
    # axes[1].set_ylabel("System Reserve Index\n(1 - max_loading/100)")
    # axes[1].set_ylim(0, 1)
    # axes[1].legend(fontsize=8)

    # Panel 3 - number of stressed lines
    axes[1].fill_between(df_ok.index, df_ok["n_lines_over_80"], alpha=0.5,
                         color="orange", label="Lines > 80%")
    axes[1].fill_between(df_ok.index, df_ok["n_lines_overloaded"], alpha=0.7,
                         color="red",    label="Lines > 100%")
    axes[1].set_ylabel("Number of lines")
    axes[1].set_xlabel("Snapshot index (t)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"outputs/line_loading_over_time_{id}.png", dpi=150)
    plt.show()

def get_connected_lines(net, bus_idx):
    """Return the line indices connected to a given bus (as either from_bus or to_bus)."""
    mask = (net.line['from_bus'] == bus_idx) | (net.line['to_bus'] == bus_idx)
    return net.line.index[mask].tolist()

def find_system_reserve(line_loading_history, bus_voltage_history, net, top_n=20):
    line_mean_loading = line_loading_history.mean(axis=0).sort_values()
    bus_voltage_deviation = (bus_voltage_history - 1.0).abs().mean(axis=0).sort_values()

    reserve_lines = line_mean_loading.head(top_n)
    reserve_buses = bus_voltage_deviation.head(top_n)

    print(f"Top {top_n} most UNDER-USED lines (System Reserve):")
    for line_idx, mean_load in reserve_lines.items():
        name = net.line.loc[line_idx, "name"] if "name" in net.line.columns else f"line_{line_idx}"
        from_bus = net.line.loc[line_idx, "from_bus"]
        to_bus = net.line.loc[line_idx, "to_bus"]
        print(f"  {name} (idx {line_idx}): mean loading = {mean_load:.2f}%  |  connects bus {from_bus} <-> bus {to_bus}")

    print(f"\nTop {top_n} most STABLE buses (System Reserve, by voltage):")
    for bus_idx, dev in reserve_buses.items():
        name = net.bus.loc[bus_idx, "name"] if "name" in net.bus.columns else f"bus_{bus_idx}"
        connected_lines = get_connected_lines(net, bus_idx)
        print(f"  {name} (idx {bus_idx}): mean |vm - 1.0| = {dev:.4f} pu  |  connected lines: {connected_lines}")

    return reserve_lines, reserve_buses


def find_system_burden(line_loading_history, bus_voltage_history, net, top_n=20, v_band=(0.95, 1.05)):
    line_max_loading = line_loading_history.max(axis=0).sort_values(ascending=False)
    burden_lines = line_max_loading.head(top_n)

    v_min, v_max = v_band
    bus_voltage_violations = ((bus_voltage_history < v_min) | (bus_voltage_history > v_max)).sum(axis=0).sort_values(ascending=False)
    burden_buses = bus_voltage_violations.head(top_n)

    print(f"\nTop {top_n} highest-loaded lines (System Burden, by peak loading):")
    for line_idx, max_load in burden_lines.items():
        name = net.line.loc[line_idx, "name"] if "name" in net.line.columns else f"line_{line_idx}"
        from_bus = net.line.loc[line_idx, "from_bus"]
        to_bus = net.line.loc[line_idx, "to_bus"]
        print(f"  {name} (idx {line_idx}): peak loading = {max_load:.2f}%  |  connects bus {from_bus} <-> bus {to_bus}")

    print(f"\nTop {top_n} most BURDENED buses (voltage outside {v_band} pu):")
    for bus_idx, count in burden_buses.items():
        name = net.bus.loc[bus_idx, "name"] if "name" in net.bus.columns else f"bus_{bus_idx}"
        pct_of_time = 100 * count / len(bus_voltage_history)
        connected_lines = get_connected_lines(net, bus_idx)
        print(f"  {name} (idx {bus_idx}): outside band in {count} snapshots ({pct_of_time:.1f}% of the period)  |  connected lines: {connected_lines}")

    return burden_lines, burden_buses


def plot_reserve_and_burden(reserve_lines, burden_lines, id):
    """Bar chart comparing the most under-used lines (reserve) to the most stressed lines (burden)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh([str(i) for i in reserve_lines.index[::-1]], reserve_lines.values[::-1], color="seagreen")
    axes[0].set_xlabel("Mean loading (%)")
    axes[0].set_title("Most Under-Used Lines\n(System Reserve)")

    axes[1].barh([str(i) for i in burden_lines.index[::-1]], burden_lines.values[::-1], color="firebrick")
    axes[1].set_xlabel("Max loading (%)")
    axes[1].set_title("Most Burdened Lines\n(System Burden)")

    plt.tight_layout()
    plt.savefig(f"outputs/system_reserve_and_burden_{id}.png", dpi=150)
    plt.show()

def plot_mean_vs_max(line_ids, line_loading_history, net, title, filename, id):
    """
    Dumbbell plot: MEAN loading (typical usage) vs MAX loading (worst-case spike)
    for a given set of lines - reveals whether the average is a reliable picture
    of risk, or hides occasional spikes.
    """
    means = line_loading_history[line_ids].mean(axis=0)
    maxes = line_loading_history[line_ids].max(axis=0)

    names = [net.line.loc[i, "name"] if "name" in net.line.columns else f"line_{i}" for i in line_ids]
    y_pos = np.arange(len(line_ids))

    plt.figure(figsize=(10, 6))
    plt.hlines(y=y_pos, xmin=means.values, xmax=maxes.values, color="grey", alpha=0.6, linewidth=2)
    plt.scatter(means.values, y_pos, color="steelblue", s=80, label="Mean loading", zorder=3)
    plt.scatter(maxes.values, y_pos, color="firebrick", s=80, label="Max loading", zorder=3)

    plt.axvline(80, color="orange", linestyle="--", linewidth=0.8, label="Warning 80%")
    plt.axvline(100, color="red", linestyle="--", linewidth=0.8, label="Limit 100%")

    plt.yticks(y_pos, names)
    plt.xlabel("Line loading (%)")
    plt.xlim((0,100))
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"outputs/{filename}_{id}.png", dpi=150)
    plt.show()

def plot_voltage_deviation_mean_vs_max(bus_ids, bus_voltage_history, net, title, filename, id, xlim=None):
    dev = (bus_voltage_history[bus_ids] - 1.0).abs()
    means, maxes = dev.mean(axis=0), dev.max(axis=0)
    names = [net.bus.loc[i, "name"] if "name" in net.bus.columns else f"bus_{i}" for i in bus_ids]
    y_pos = np.arange(len(bus_ids))

    plt.figure(figsize=(10, 6))
    plt.hlines(y=y_pos, xmin=means.values, xmax=maxes.values, color="grey", alpha=0.6, linewidth=2)
    plt.scatter(means.values, y_pos, color="steelblue", s=80, label="Mean |Δv|", zorder=3)
    plt.scatter(maxes.values, y_pos, color="firebrick", s=80, label="Max |Δv|", zorder=3)
    plt.yticks(y_pos, names)
    plt.xlabel("Voltage deviation from 1.0 pu")
    plt.title(title)
    if xlim is not None:
        plt.xlim(xlim)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"outputs/{filename}_{id}.png", dpi=150)
    plt.show()

# Node/edge extraction works for both pandapower 2.x.x and 3.x.x
def get_node_list(net):
    """
    Extract bus coordinates as a DataFrame, compatible with both pandapower
    2.x (net.bus_geodata) and 3.x (net.bus['geo'] as GeoJSON strings).
    """
    if "bus_geodata" in net and len(net.bus_geodata) > 0:
        # pandapower 2.x.x
        node_list = pd.DataFrame({
            'NodeIndex': net.bus_geodata.index,
            'x': net.bus_geodata['x'],
            'y': net.bus_geodata['y']
        })
    else:
        # pandapower 3.x.x
        coords = net.bus["geo"].apply(
            lambda g: json.loads(g)["coordinates"] if pd.notna(g) else [None, None]
        )
        node_list = pd.DataFrame({
            'NodeIndex': net.bus.index,
            'x': coords.apply(lambda c: c[0]),
            'y': coords.apply(lambda c: c[1])
        })
    # node_list.to_csv('nodes.csv')
    return node_list


def get_arc_list(net):
    """Build an edge list (from_bus -> to_bus) from the network's lines."""
    arc_list = pd.DataFrame({'SourceID': net.line['from_bus'], 'TargetID': net.line['to_bus']})
    return arc_list

def make_graph_net(node_list, arc_list, id):
    """
    Build an undirected igraph.Graph from a node list (with coordinates) and
    an edge list of bus IDs, then plot it using the geographic layout.
    """
    # Reset node_list to a clean 0..N-1 positional index, keep original bus IDs as a column
    node_list = node_list.reset_index(drop=True)

    # Map original pandapower bus IDs to igraph vertex positions
    bus_id_to_vertex_pos = {bus_id: pos for pos, bus_id in enumerate(node_list['NodeIndex'])}

    graph_simbench = ig.Graph(directed=False)
    graph_simbench.add_vertices(len(node_list))  # vertices 0..N-1, matching node_list row order

    edges = []
    for _, row in arc_list.iterrows():
        source_id = row['SourceID']
        target_id = row['TargetID']
        if source_id in bus_id_to_vertex_pos and target_id in bus_id_to_vertex_pos:
            edges.append((bus_id_to_vertex_pos[source_id], bus_id_to_vertex_pos[target_id]))
        else:
            print(f"Skipping edge ({source_id}, {target_id}) - bus not in node_list")

    graph_simbench.add_edges(edges)  # bulk add is faster than looping add_edge calls
    layout = [(lon, lat * -1) for lon, lat in zip(node_list['x'], node_list['y'])]
    ig.plot(graph_simbench, layout=layout, vertex_size=7,vertex_label_size=4, vertex_label=node_list.index.tolist(), target=f"outputs/network_{id}.pdf")
    return graph_simbench, layout

def calculate_betweenness_centrality(graph_simbench, node_list, id):
    """Compute, plot, and append betweenness centrality for each bus."""
    pal = ig.drawing.colors.ClusterColoringPalette(10000)
    vals_between = graph_simbench.betweenness()
    graphCol_between = [pal.get(int(x)) for x in vals_between if not math.isnan(x)]
    ig.plot(graph_simbench, vertex_color=graphCol_between, vertex_size=7,
            vertex_label=node_list.index.tolist(), vertex_label_size=4, target=f'outputs/Betweenness_{id}.pdf')

    node_list['betweenness'] = np.round(vals_between,2)
    print(node_list)

    plt.figure(figsize=(15, 10))
    plt.bar(node_list['NodeIndex'], node_list['betweenness'], color='steelblue')
    plt.title('Betweenness Centrality')
    plt.xlabel('number node')
    plt.ylabel('betweenness')
    plt.savefig(f'outputs/Betweenness_{id}.png')
    plt.show()


def calculate_closeness_centrality(graph_simbench, node_list, id):
    """Compute, plot, and append closeness centrality for each bus."""
    pal = ig.drawing.colors.ClusterColoringPalette(10000)
    vals_close = graph_simbench.closeness()
    graphCol_close = [pal.get(int(x)) for x in vals_close if not math.isnan(x)]
    ig.plot(graph_simbench, vertex_color=graphCol_close, vertex_size=7,
           vertex_label=node_list.index.tolist(), vertex_label_size=4, target=f'outputs/Closeness_{id}.pdf')

    node_list['closeness'] = np.round(vals_close,2)
    print(node_list)

    plt.figure(figsize=(15, 10))
    plt.bar(node_list['NodeIndex'], node_list['closeness'], color='darkorange')
    plt.title('Closeness Centrality')
    plt.xlabel('number node')
    plt.ylabel('closeness')
    plt.savefig(f'outputs/Closeness_{id}.png')
    plt.show()

# def calculate_eigenvector_centrality(graph_simbench, node_list, id):
#     """Compute, plot, and append eigenvector centrality for each bus."""
#     pal = ig.drawing.colors.ClusterColoringPalette(10000)
#     vals_eigen = graph_simbench.eigenvector_centrality()
#     graphCol_eigen = [pal.get(int(x)) for x in vals_eigen if not math.isnan(x)]
#     # ig.plot(graph_simbench, vertex_color=graphCol_eigen, vertex_size=7,
#     #         vertex_label=node_list.index.tolist(), vertex_label_size=4, layout=layout, target=f'Eigenvector_{id}.pdf')

#     node_list['eigenvector'] = np.round(vals_eigen,2)
#     print(node_list)

#     node_list.to_csv(f'Centrality_{id}.csv', sep = ';')
#     plt.figure(figsize=(15, 10))
#     plt.bar(node_list['NodeIndex'], node_list['eigenvector'], color='seagreen')
#     plt.title('Eigenvector Centrality')
#     plt.xlabel('number node')
#     plt.ylabel('eigenvector')
#     plt.savefig(f'Eigenvector_{id}.png')
#     plt.show()

def build_betweenness_loading_table(burden_lines, node_list, net, top_n_betweenness=10):
    """
    For each highly-loaded line, check whether either endpoint bus is among
    the top-N highest-betweenness buses - tests real physical connectivity
    between loading hotspots and structurally important buses.
    """
    betweenness_lookup = node_list.set_index('NodeIndex')['betweenness']
    top_betweenness_buses = set(betweenness_lookup.nlargest(top_n_betweenness).index)

    rows = []
    for line_idx, peak_load in burden_lines.items():
        from_bus = net.line.loc[line_idx, 'from_bus']
        to_bus = net.line.loc[line_idx, 'to_bus']
        line_name = net.line.loc[line_idx, 'name'] if 'name' in net.line.columns else f'line_{line_idx}'

        rows.append({
            'Line': line_name,
            'Peak Loading (%)': round(peak_load, 2),
            'From Bus': from_bus,
            'From Bus Betweenness': round(betweenness_lookup.get(from_bus, float('nan')), 2),
            'To Bus': to_bus,
            'To Bus Betweenness': round(betweenness_lookup.get(to_bus, float('nan')), 2),
            'Connects to Top-Betweenness Bus?': 'Yes' if (from_bus in top_betweenness_buses or to_bus in top_betweenness_buses) else 'No'
        })

    return pd.DataFrame(rows)
