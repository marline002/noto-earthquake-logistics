"""Noto ELN implementation aligned with revised_algorithms.tex (Algorithms 1--4).

Dependencies: numpy, pandas, networkx, scipy. Python >= 3.10.
Run: python v13_core.py --data-dir noto_data --N 1000
Audited experiments: python run_v13_structure.py --stage main

The three original case CSVs are required; no missing case observations are
fabricated. Import has no data-loading or output-directory side effects.
Original public function names and the three-value route_and_plan return are
retained. Legacy mode names soft/soft_mean/frozen_split are supported.

Conventions fixed by the revised algorithm:
* Input demand/capacities are amounts per 3 h; scale them by Delta_t_R/3.
* H0=(1-damage)**alpha, then C=C0*H at every subsequent time.
* Requests drive stress, but only capacity-feasible plans drive deliveries.
* The four factorial cells use the same progressive shared-capacity allocator.
* M4 freezes complete initial OD-path requests and request-relative weights.
* One degradation update and one aggregate reliability check per substep.
* Soft delivery uses pre-update capacity; binary deletion precedes delivery.
* Y is indexed at interval starts; psi is indexed at interval ends, including T.
* New demand enters FIFO before service; remaining batches age after service.
* Process uniforms are indexed by replicate/component/substep, not branch order.
* M3 needs explicit external intensities or a separate calibration ensemble.
* Component PGA inputs are fixed (scenario mode); no random spatial field is
  inferred or simulated from the missing geospatial calibration inputs.

These changes alter the simulator. Previously reported paper numbers are not
validated or preserved by this revision. The supplied CSVs define a scenario, not independently observed capacity-loss data.
Use run_v13_structure.py for the audited manuscript results.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import deque
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import gammaincinv
from scipy.stats import binomtest, gamma as _gamma_dist, norm, beta as beta_dist

BASE = Path(__file__).resolve().parent
DATA = BASE / "noto_data"
FIGDIR = BASE / "noto_results"
DEFAULT_CV, DEFAULT_DT, DEFAULT_DT_D = 0.25, 3.0, 1.0
Y_CRIT, PSI_CRIT, T, EPS, K, N_MC = 0.50, 0.20, 72.0, 1e-6, 6, 1000
REFERENCE_HOURS = 3.0
CAPACITY_TOL = 1e-10  # allocation arithmetic only, not a health-failure threshold
omega_C, omega_R, omega_D, eta = 0.35, 0.35, 0.30, 0.50
LINK_CAP = {"expressway": 333.3, "class1": 166.7, "mountain": 66.7}
THETA_BY_ROLE = {"supply": (0.80, 0.45), "trans": (0.70, 0.50), "demand": (0.55, 0.55)}
THETA_BY_CLASS = {"expressway": (0.90, 0.45), "class1": (0.70, 0.50), "mountain": (0.45, 0.55)}
G = nx.DiGraph()
NODE_ROLE, NODE_CN, NODE_POS, NODE_CAP, DEMAND_D = {}, {}, {}, {}, {}
SUPPLY, DEMAND, COMPONENTS, NODE_COMPONENTS = [], [], [], []
M_pga, theta, beta = {}, {}, {}
gamma, m_exp, delta, n_exp, rho_safe = {}, {}, {}, {}, {}
c0, c1, C0_DICT, ALPHA_BASE, ALPHA_DEFAULT = {}, {}, {}, {}, {}
PATHS, PATH_COMPONENTS, PATH_TRAVEL, PATH_TCOST = {}, {}, {}, {}
b_dict, lambda_exo = {}, {}


def _nonnegative(x, name):
    x = float(x)
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return x


def _time_grid(Delta_t_R, delta_t_D, horizon):
    for value in (Delta_t_R, delta_t_D, horizon):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Time intervals and horizon must be finite and positive")
    nr, steps = horizon / Delta_t_R, Delta_t_R / delta_t_D
    if not math.isclose(nr, round(nr), abs_tol=1e-10) or not math.isclose(steps, round(steps), abs_tol=1e-10):
        raise ValueError("T/Delta_t_R and Delta_t_R/delta_t_D must both be positive integers")
    return int(round(nr)), int(round(steps))


def load_data(data_dir=None):
    """Load actual CSVs. The caller can override DATA via --data-dir."""
    root = Path(data_dir) if data_dir is not None else DATA
    files = [root / name for name in ("nodes_noto.csv", "links_noto.csv", "population_demand_noto.csv")]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required case data are missing: " + ", ".join(missing))
    configure_network(*(pd.read_csv(p) for p in files))


def configure_network(nodes, links, population):
    """Configure from dataframes; also supports explicitly synthetic test data.

    Optional nominal_capacity_t_per_3h columns override case capacity defaults.
    They are supplied inputs, not inferred observations.
    """
    global G, NODE_ROLE, NODE_CN, NODE_POS, NODE_CAP, DEMAND_D, SUPPLY, DEMAND
    global COMPONENTS, NODE_COMPONENTS, M_pga, theta, beta, gamma, m_exp, delta
    global n_exp, rho_safe, c0, c1, C0_DICT, ALPHA_BASE, ALPHA_DEFAULT, b_dict, lambda_exo
    required = [(nodes, {"node", "role", "pga_g"}),
                (links, {"u", "v", "road_class", "travel_time_h", "length_km", "pga_g"}),
                (population, {"node", "demand_t_per_3h"})]
    for frame, columns in required:
        if not columns <= set(frame.columns):
            raise ValueError(f"Missing columns: {sorted(columns-set(frame.columns))}")
    if nodes.node.duplicated().any() or population.node.duplicated().any() or links.duplicated(["u", "v"]).any():
        raise ValueError("Duplicate node, demand or directed-edge identifiers")
    NODE_ROLE = {str(row.node): str(row.role) for row in nodes.itertuples()}
    if any("-" in node for node in NODE_ROLE):
        raise ValueError("Node IDs cannot contain '-' because link IDs use u-v")
    if any(role not in THETA_BY_ROLE for role in NODE_ROLE.values()):
        raise ValueError("Node roles must be supply, trans or demand")
    DEMAND_D = {str(row.node): _nonnegative(row.demand_t_per_3h, "demand") for row in population.itertuples()}
    SUPPLY = sorted(n for n, role in NODE_ROLE.items() if role == "supply")
    DEMAND = sorted(DEMAND_D)
    if not SUPPLY or any(d not in NODE_ROLE for d in DEMAND):
        raise ValueError("Supply and demand nodes must exist in the node inventory")
    G = nx.DiGraph()
    G.add_nodes_from(sorted(NODE_ROLE))
    M_pga, theta, beta, NODE_CAP, NODE_CN, NODE_POS = {}, {}, {}, {}, {}, {}
    default_caps = {"KZ": 400., "HK": 200., "NN": 300., "SK": 150., "AN": 200.}
    for row in nodes.to_dict("records"):
        n = str(row["node"])
        M_pga[n] = _nonnegative(row["pga_g"], "PGA")
        theta[n], beta[n] = THETA_BY_ROLE[NODE_ROLE[n]]
        NODE_CN[n] = row.get("name_cn", n)
        NODE_POS[n] = (row.get("lon", float("nan")), row.get("lat", float("nan")))
        default = default_caps.get(n, 2 * DEMAND_D.get(n, 200.))
        val = row.get("nominal_capacity_t_per_3h", default)
        NODE_CAP[n] = _nonnegative(default if pd.isna(val) else val, "node capacity")
    for row in sorted(links.to_dict("records"), key=lambda r: (str(r["u"]), str(r["v"]))):
        u, v, cls = str(row["u"]), str(row["v"]), str(row["road_class"])
        if u not in NODE_ROLE or v not in NODE_ROLE or cls not in THETA_BY_CLASS:
            raise ValueError("Unknown link endpoint or road class")
        cap = row.get("nominal_capacity_t_per_3h", LINK_CAP[cls])
        cap = LINK_CAP[cls] if pd.isna(cap) else cap
        G.add_edge(u, v, time=_nonnegative(row["travel_time_h"], "travel time"),
                   cost=0.03*_nonnegative(row["length_km"], "length"), road_class=cls,
                   capacity=_nonnegative(cap, "link capacity"))
        comp = f"{u}-{v}"
        M_pga[comp] = _nonnegative(row["pga_g"], "PGA")
        theta[comp], beta[comp] = THETA_BY_CLASS[cls]
    NODE_COMPONENTS = sorted(NODE_ROLE)
    COMPONENTS = NODE_COMPONENTS + [f"{u}-{v}" for u, v in sorted(G.edges)]
    gamma, m_exp, delta, n_exp, rho_safe = {}, {}, {}, {}, {}
    for comp in COMPONENTS:
        is_node = comp in NODE_ROLE
        gamma[comp], m_exp[comp], delta[comp], n_exp[comp], rho_safe[comp] = (
            0.001 if is_node else 0.0013, 2. if is_node else 2.5,
            0.0026 if is_node else 0., 1.5 if is_node else 0., 0.70)
    C0_DICT = {comp: nominal_capacity(comp) for comp in COMPONENTS}
    c0, c1 = {d: 1. for d in DEMAND}, {d: 0.02 for d in DEMAND}
    ALPHA_BASE = {comp: 1. for comp in COMPONENTS}
    ALPHA_DEFAULT = build_alpha_dict()
    b_dict = build_b_dict(DEFAULT_CV)
    lambda_exo = {comp: _intensity(comp, 1.) for comp in COMPONENTS}
    set_K(K)


def _ensure_loaded():
    if not COMPONENTS:
        load_data()


def path_components(path):
    return list(path) + [f"{u}-{v}" for u, v in zip(path, path[1:])]


def nominal_capacity(comp):
    if comp in NODE_ROLE:
        return NODE_CAP[comp]
    u, v = comp.split("-", 1)
    return G.edges[u, v]["capacity"]


def get_k_shortest_paths(graph, source, target, k):
    from itertools import islice
    try:
        return list(islice(nx.shortest_simple_paths(graph, source, target, weight="time"), k))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def set_K(k_new):
    global K, PATHS, PATH_COMPONENTS, PATH_TRAVEL, PATH_TCOST
    if int(k_new) != k_new or k_new <= 0:
        raise ValueError("K must be a positive integer")
    K = int(k_new)
    PATHS = {(i, j): get_k_shortest_paths(G, i, j, K) for i in SUPPLY for j in DEMAND}
    PATH_COMPONENTS, PATH_TRAVEL, PATH_TCOST = {}, {}, {}
    for paths in PATHS.values():
        for p in paths:
            key = tuple(p)
            PATH_COMPONENTS[key] = path_components(p)
            PATH_TRAVEL[key] = sum(G.edges[u, v]["time"] for u, v in zip(p, p[1:]))
            PATH_TCOST[key] = sum(G.edges[u, v]["cost"] for u, v in zip(p, p[1:]))


def export_paths():
    return [{"origin": i, "destination": j, "rank": rank+1, "nodes": path,
             "loopless": len(path) == len(set(path)), "travel_time_h": PATH_TRAVEL[tuple(path)]}
            for (i, j), paths in PATHS.items() for rank, path in enumerate(paths)]


def deprivation_phi(d, tau):
    return c0[d] * math.expm1(c1[d] * tau)


def build_alpha_dict(alpha_mult=1.0, alpha_base=None):
    base = ALPHA_BASE if alpha_base is None else alpha_base
    result = {c: float(base[c]) * float(alpha_mult.get(c, 1.) if isinstance(alpha_mult, dict) else alpha_mult)
              for c in COMPONENTS}
    if any(not math.isfinite(v) or v <= 0 for v in result.values()):
        raise ValueError("alpha must be finite and positive")
    return result


def _intensity(c, rho, gamma_mult=1., m_shift=0., delta_mult=1.):
    rate = gamma[c]*gamma_mult*rho**(m_exp[c]+m_shift)
    if c in NODE_ROLE:
        rate += delta[c]*delta_mult*max(rho-rho_safe[c], 0.)**n_exp[c]
    if not math.isfinite(rate) or rate < 0:
        raise ValueError("Nonfinite or negative degradation intensity")
    return rate


def build_b_dict(cv, reference_hours=1., reference_rho=1.):
    """CV at rho=1 over 1 h by default; scale is independent of integration grid."""
    cv = _nonnegative(cv, "CV")
    if reference_hours <= 0 or reference_rho <= 0:
        raise ValueError("CV reference duration and stress must be positive")
    return {c: _intensity(c, reference_rho)*reference_hours*cv**2 for c in COMPONENTS}


def gamma_ppf(u, a, scale):
    if a <= 0 or scale <= 0:
        raise ValueError("Gamma shape and scale must be positive")
    if not 0 < u < 1:
        raise ValueError("Common uniforms must lie strictly inside (0,1)")
    value = float(_gamma_dist.ppf(u, a=a, scale=scale))
    if not math.isfinite(value):
        raise FloatingPointError("Nonfinite Gamma inverse-CDF result")
    return value


def _uniforms(rng, shape):
    return np.clip(rng.random(shape), np.nextafter(0., 1.), np.nextafter(1., 0.))


def _mode(m):
    aliases = {"gamma": "soft", "mean": "soft_mean", "frozen": "static",
               "link_only": "soft", "node_only": "soft"}
    result = aliases.get(m, m)
    if result not in {"soft", "soft_mean", "static", "binary", "exogenous", "exogenous_matched"}:
        raise ValueError(f"Unknown degradation mode: {m}")
    return result


def degradation_step(m, attempted, H, C, C0_dict, rng, delta_t_D,
                     gamma_mult=1., m_shift=0., delta_mult=1., b_local=None,
                     degrade_nodes=True, degrade_links=True, lambda_bar=None,
                     alpha=None, t_r_q=None, cv=DEFAULT_CV,
                     U_proc=None, substep_idx=None, comp_index=None, epsilon=EPS):
    """Algorithm 3 state update, exactly once per substep. Inputs are not mutated.

    alpha is accepted for compatibility; it acts only in initial_state, not
    through a second exponentiation of the capacity after initialization.
    """
    if m == "link_only":
        degrade_nodes = False
    if m == "node_only":
        degrade_links = False
    mode = _mode(m)
    Hnew, Cnew = dict(H), dict(C)
    if mode in {"static", "binary"}:
        return Hnew, Cnew
    if mode.startswith("exogenous") and lambda_bar is None:
        raise ValueError("An exogenous cell requires explicit external intensities")
    b_local = build_b_dict(cv) if b_local is None else b_local
    stochastic = []
    for c in COMPONENTS:
        if (c in NODE_ROLE and not degrade_nodes) or (c not in NODE_ROLE and not degrade_links) or H[c] == 0:
            continue
        if mode.startswith("exogenous"):
            lam = lambda_bar(c, t_r_q) if callable(lambda_bar) else lambda_bar[c]
            lam = _nonnegative(lam, "external intensity")  # already calibrated; do not multiply twice
        else:
            left = attempted[c]/(C0_dict[c]*H[c]+epsilon)
            pred = max(0., H[c]-_intensity(c, left, gamma_mult, m_shift, delta_mult)*delta_t_D)
            mid = (H[c]+pred)/2
            lam = _intensity(c, attempted[c]/(C0_dict[c]*mid+epsilon), gamma_mult, m_shift, delta_mult)
        if lam == 0:
            inc = 0.
        elif mode == "soft_mean" or b_local[c] == 0 or cv == 0:
            inc = lam*delta_t_D
        else:
            u = U_proc[substep_idx, comp_index[c]] if U_proc is not None else float(_uniforms(rng, (1,))[0])
            stochastic.append((c, u, lam*delta_t_D/b_local[c], b_local[c]))
            continue
        Hnew[c] = max(0., H[c]-inc)
        Cnew[c] = C0_dict[c]*Hnew[c]
    if stochastic:
        cs, us, shapes, scales = zip(*stochastic)
        increments = gammaincinv(np.asarray(shapes), np.asarray(us))*np.asarray(scales)
        if not np.isfinite(increments).all():
            raise FloatingPointError("Nonfinite Gamma inverse-CDF result")
        for c, inc in zip(cs, increments):
            Hnew[c] = max(0., H[c]-float(inc))
            Cnew[c] = C0_dict[c]*Hnew[c]
    return Hnew, Cnew


def feasible_increment(request, paths, weights, capacities):
    """Algorithm 4. paths maps a stable path ID to its component sequence."""
    R = _nonnegative(request, "request")
    c = {k: _nonnegative(v, "residual capacity") for k, v in capacities.items()}
    flows = {p: 0. for p in paths}
    w = {p: _nonnegative(weights[p], "path weight") for p in paths}
    active = [p for p in sorted(paths) if all(c[k] > CAPACITY_TOL for k in paths[p])]
    passes = 0
    while R > CAPACITY_TOL and active:
        total = math.fsum(w[p] for p in active)
        shares = {p: w[p]/total if total > 0 else 1./len(active) for p in active}
        increments = {p: R*shares[p] for p in active}
        loads = {k: math.fsum(increments[p] for p in active if k in paths[p]) for k in c}
        beta_step = min([1.]+[c[k]/loads[k] for k in c if loads[k] > 0])
        for p in active:
            flows[p] += beta_step*increments[p]
        for k in c:
            c[k] = max(0., c[k]-beta_step*loads[k])
        R *= 1-beta_step
        passes += 1
        if beta_step < 1:
            remaining = [p for p in active if all(c[k] > CAPACITY_TOL for k in paths[p])]
            if len(remaining) == len(active):
                raise ArithmeticError("Allocator failed to remove its saturated bottleneck")
            active = remaining
    if passes > len(paths):
        raise AssertionError("Allocator exceeded active-path pass bound")
    return flows, c


def route_and_plan(pi, m, C, batches, Delta_t_R, demand_mult, Z,
                   r=0, prev_attempted=None, supply_budgets=None, baseline_loads=None,
                   path_pools=None):
    """Algorithm 2; return component requests, delivery plan, complete path requests.

    Plan and requests are keyed by destination, then component-sequence tuples
    (the first node identifies the origin). Pass INITIAL requests at r>0 for M4.
    With no separate stock input, supply budget is the nominal supply-node
    interval capacity, explicitly a scenario assumption and not a stock estimate.
    """
    _mode(m)
    pi = {"frozen_request": "frozen_split"}.get(pi, pi)
    if pi not in {"split", "fastest", "frozen_split"}:
        raise ValueError(f"Unknown routing policy: {pi}")
    if pi == "frozen_split" and r > 0 and prev_attempted is None:
        raise ValueError("M4 needs the stored initial OD-path requests")
    scale = Delta_t_R/REFERENCE_HOURS
    eps = EPS*scale
    attempted = {c: _nonnegative((baseline_loads or {}).get(c, 0.), "baseline request") for c in COMPONENTS}
    residual = dict(C)
    plan, Q = {d: [] for d in DEMAND}, {d: {} for d in DEMAND}
    demand = {d: DEMAND_D[d]*scale*demand_mult for d in DEMAND}
    remaining_demand = {d: demand[d]+math.fsum(g for g, _ in batches[d]) for d in DEMAND}
    supply = {i: _nonnegative((supply_budgets or {}).get(i, C0_DICT[i]*scale), "supply budget") for i in SUPPLY}
    def priority(od):
        i, j = od
        age = max((a for _, a in batches[j]), default=0.)
        welfare = remaining_demand[j]*(deprivation_phi(j, age+Delta_t_R)-deprivation_phi(j, age))
        return -welfare, -demand[j], i, j
    order = sorted(((i, d) for i in SUPPLY for d in DEMAND), key=priority)
    pools = PATHS if path_pools is None else path_pools
    for i, j in order:
        candidates = {}
        for p in pools.get((i, j), []):
            nodekey = tuple(p)
            comps = tuple(path_components(p))
            if all(C[c] > 0 for c in comps):
                travel = sum(G.edges[u, v]["time"] for u, v in zip(p, p[1:]))
                cost = sum(G.edges[u, v]["cost"] for u, v in zip(p, p[1:]))
                candidates[comps] = (travel, cost)
        if pi == "frozen_split" and r > 0:
            qod = {tuple(p): v for p, v in prev_attempted.get(j, {}).items() if p[0] == i}
            Q[j].update(qod)
            active = {p: p for p in candidates if qod.get(p, 0.) > 0}
            weights = {p: qod[p] for p in active}
        else:
            a = min(remaining_demand[j], supply[i])
            if not candidates or a == 0:
                active, weights, qod = {}, {}, {}
            elif pi == "fastest":
                fastest = min(candidates, key=lambda p: (candidates[p][0], p))
                active, weights, qod = {fastest: fastest}, {fastest: 1.}, {fastest: a}
            else:
                age = max((a for _, a in batches[j]), default=0.)
                weights = {}
                for p, (travel, cost) in candidates.items():
                    rho = max(attempted[c]/(C[c]+eps) for c in p)
                    urgency = cost+eta*(deprivation_phi(j, age+travel)-deprivation_phi(j, age))
                    # Initial request on this OD's paths is zero: Cp/(Cp+0)=1.
                    weights[p] = max(0., 1-rho)**omega_R*math.exp(-omega_D*urgency)
                total = math.fsum(weights.values())
                qod = {p: a*(weights[p]/total if total > 0 else 1/len(weights)) for p in weights}
                active = {p: p for p in candidates}
            Q[j].update(qod)
        for p, value in qod.items():
            for c in p:
                attempted[c] += value
        request = min(math.fsum(qod.values()), remaining_demand[j], supply[i])
        flows, residual = feasible_increment(request, active, weights, residual)
        for p, flow in flows.items():
            if flow > 0:
                plan[j].append((list(p), flow))
        planned = math.fsum(flows.values())
        remaining_demand[j] = max(0., remaining_demand[j]-planned)
        supply[i] = max(0., supply[i]-planned)
    return attempted, plan, Q


def sample_scenario(rng, beta_mult=1.):
    _ensure_loaded()
    beta_mult = _nonnegative(beta_mult, "fragility dispersion multiplier")
    return {c: rng.lognormal(math.log(theta[c]), beta[c]*beta_mult) for c in COMPONENTS}


def initial_state(theta_real, alpha=None, Delta_t_R=DEFAULT_DT, initial_closed=()):
    """Initialize H=(1-damage)**alpha and C=C0(interval)*H, including zero PGA."""
    alpha = ALPHA_DEFAULT if alpha is None else alpha
    H, C = {}, {}
    for c in COMPONENTS:
        al = float(alpha[c])
        if not math.isfinite(al) or al <= 0:
            raise ValueError("alpha must be finite and positive")
        resistance = _nonnegative(theta_real[c], "resistance")
        h0 = min(1., resistance/M_pga[c]) if M_pga[c] > 0 else 1.
        H[c] = 0. if c in initial_closed else h0**al
        C[c] = C0_DICT[c]*(Delta_t_R/REFERENCE_HOURS)*H[c]
    return H, C


def _fifo_update(queue, arrivals, delivery, dt):
    if arrivals > 0:
        queue.append((arrivals, 0.))
    total = math.fsum(g for g, _ in queue)
    if delivery > total+1e-8*max(1., total):
        raise AssertionError("Offered delivery exceeds available destination demand")
    remaining = delivery
    while queue and remaining > 0:
        amount, age = queue.popleft()
        served = min(amount, remaining)
        remaining -= served
        if amount > served:
            queue.appendleft((amount-served, age))
    return deque((g, age+dt) for g, age in queue if g > 0)


def simulate_cell(pi, m, theta_real, rng_deg, Delta_t_R=3., delta_t_D=1.,
                  gamma_mult=1., m_shift=0., delta_mult=1., demand_mult=1.,
                  routing_mode=None, degrade_nodes=True, degrade_links=True,
                  lambda_bar=None, alpha=None, cv=DEFAULT_CV, U_proc=None,
                  comp_index=None, Y_crit=None, Psi_crit=None, horizon=None,
                  supply_budgets=None, baseline_loads=None, return_diagnostics=False,
                  load_driver="request", delivery_rule="proportional", arrival_delay=False,
                  shared_roads=False, initial_closed=(), initial_stock_hours=0., retry_hours=None):
    """Algorithms 1/3: one configuration and one seismic realization."""
    _ensure_loaded()
    horizon = T if horizon is None else horizon
    nr, steps = _time_grid(Delta_t_R, delta_t_D, horizon)
    demand_mult = _nonnegative(demand_mult, "demand multiplier")
    gamma_mult, delta_mult = _nonnegative(gamma_mult, "rate multiplier"), _nonnegative(delta_mult, "threshold multiplier")
    if any(m_exp[c]+m_shift <= 0 for c in COMPONENTS):
        raise ValueError("Stress exponents must be positive")
    pi = routing_mode if routing_mode is not None else pi
    if m == "link_only":
        degrade_nodes = False
    elif m == "node_only":
        degrade_links = False
    mode = _mode(m)
    if mode.startswith("exogenous") and lambda_bar is None:
        raise ValueError("M3/exogenous requires prescribed intensities or explicit independent calibration")
    nsub, scale = nr*steps, Delta_t_R/REFERENCE_HOURS
    b_local = build_b_dict(cv)
    times = np.arange(nsub)*delta_t_D
    psi_times = (np.arange(nsub)+1)*delta_t_D
    if U_proc is None:
        U_proc = _uniforms(rng_deg, (nsub, len(COMPONENTS)))
    comp_index = {c: i for i, c in enumerate(COMPONENTS)} if comp_index is None else comp_index
    if set(comp_index) != set(COMPONENTS) or set(comp_index.values()) != set(range(len(COMPONENTS))):
        raise ValueError("comp_index must be a bijection over the component inventory")
    U_proc = np.asarray(U_proc)
    if U_proc.shape != (nsub, len(COMPONENTS)) or not np.all((U_proc > 0) & (U_proc < 1)):
        raise ValueError("Process uniforms have incorrect shape or values outside (0,1)")
    if load_driver not in {"request", "throughput"} or delivery_rule not in {"proportional", "cap"}:
        raise ValueError("Invalid load driver or delivery rule")
    if set(initial_closed)-set(COMPONENTS):
        raise ValueError("Unknown initially closed component")
    road_pairs = [(f"{u}-{v}", f"{v}-{u}") for u,v in G.edges if u < v and G.has_edge(v,u)]
    theta_real = dict(theta_real)
    if shared_roads:
        U_proc = U_proc.copy()
        for a,b in road_pairs:
            theta_real[b] = theta_real[a]
            U_proc[:, comp_index[b]] = U_proc[:, comp_index[a]]
    H, C = initial_state(theta_real, alpha, Delta_t_R, initial_closed)
    initial_H, initial_C = dict(H), dict(C)
    # Pools are generated on G0, before subsequent degradation. If every
    # initial capacity is positive, the precomputed intact-graph pool is identical.
    if all(value > 0 for value in initial_C.values()):
        path_pools = PATHS
    else:
        initial_graph = G.copy()
        initial_graph.remove_nodes_from([c for c in NODE_COMPONENTS if initial_C[c] == 0])
        initial_graph.remove_edges_from([(u, v) for u, v in list(initial_graph.edges)
                                         if initial_C[f"{u}-{v}"] == 0])
        path_pools = {(i, j): get_k_shortest_paths(initial_graph, i, j, K) for i in SUPPLY for j in DEMAND}
    Cnominal = {c: C0_DICT[c]*scale for c in COMPONENTS}
    Z = {c: float(C[c] > 0) for c in COMPONENTS}
    batches = {d: deque() for d in DEMAND}
    arrivals = {d: DEMAND_D[d]/REFERENCE_HOURS*demand_mult*delta_t_D for d in DEMAND}
    total_demand = math.fsum(arrivals.values())
    psi_ref = math.fsum(DEMAND_D[d]/REFERENCE_HOURS*demand_mult*horizon*deprivation_phi(d, horizon) for d in DEMAND)
    Y, psi = np.ones(nsub), np.zeros(nsub)
    ycrit = Y_CRIT if Y_crit is None else Y_crit
    pcrit = PSI_CRIT if Psi_crit is None else Psi_crit
    tc = tdc = horizon+delta_t_D
    frozen_requests, diagnostics = None, []
    transit = {d: {} for d in DEMAND}
    history_H, history_A = [], []
    initial_stock = {d: arrivals[d]/delta_t_D*initial_stock_hours for d in DEMAND}
    for d in DEMAND:
        transit[d][0] = initial_stock[d]
    dispatched_total=0.
    served_total=0.
    def dispatch(plan, current, epoch):
        if delivery_rule == "proportional":
            return {d: [(p, f*min([1.]+[current[c]/epoch[c] for c in p])) for p,f in plan[d]] for d in DEMAND}
        residual = dict(current)
        output = {}
        for d in DEMAND:
            flows = plan[d]
            # Global capacity feasibility and per-path upper bounds. Greedy in
            # stable stored path order; it never redistributes above a plan.
            output[d] = []
            for p,f in flows:
                value = min([f]+[residual[c] for c in p])
                output[d].append((p, value))
                for c in p:
                    residual[c] = max(0., residual[c]-value)
        return output
    for r in range(nr):
        budgets = supply_budgets(r) if callable(supply_budgets) else supply_budgets
        background = baseline_loads(r) if callable(baseline_loads) else baseline_loads
        routing_batches = batches
        if arrival_delay:
            routing_batches = copy.deepcopy(batches)
            # Committed shipments offset the oldest backlog in the planning
            # ledger only. They serve real FIFO demand only when they arrive.
            for d in DEMAND:
                credit = math.fsum(transit[d].values())
                while routing_batches[d] and credit > 0:
                    g,age = routing_batches[d].popleft()
                    used = min(g, credit)
                    credit -= used
                    if g > used:
                        routing_batches[d].appendleft((g-used, age))
        attempted, plan, Q = route_and_plan(pi, mode, C, routing_batches, Delta_t_R, demand_mult, Z,
                                          r, frozen_requests, budgets, background, path_pools)
        if pi in {"frozen_request", "frozen_split"} and r == 0:
            frozen_requests = copy.deepcopy(Q)
        epoch_C = dict(C)
        if return_diagnostics:
            diagnostics.append({"epoch": r, "requests": copy.deepcopy(Q), "attempted": dict(attempted),
                                "plan": copy.deepcopy(plan), "capacities": dict(C)})
        for q in range(steps):
            idx = r*steps+q
            old_C = dict(C)
            actual_plan = dispatch(plan, old_C, epoch_C)
            driver = dict(attempted)
            if retry_hours is not None:
                if retry_hours <= 0: raise ValueError('positive retry clock required')
                driver = {c: 0. for c in COMPONENTS}
                for j, paths in Q.items():
                    B = sum(g for g,age in routing_batches[j])
                    D = DEMAND_D[j]*scale*demand_mult
                    factor = (D+B*Delta_t_R/retry_hours)/(D+B) if D+B>0 else 0.
                    for path,amount in paths.items():
                        for c in path: driver[c]+=amount*factor
            if load_driver == "throughput":
                driver = {c: 0. for c in COMPONENTS}
                for entries in actual_plan.values():
                    for p,f in entries:
                        for c in p:
                            driver[c] += f
            if return_diagnostics:
                history_H.append([H[c] for c in COMPONENTS])
                history_A.append([driver[c] for c in COMPONENTS])
            if mode == "binary":
                for c in COMPONENTS:
                    Z[c] *= float(attempted[c] <= initial_C[c]*Z[c])
                    C[c] = initial_C[c]*Z[c]
            else:
                H, C = degradation_step(mode, driver, H, C, Cnominal, rng_deg, delta_t_D,
                    gamma_mult, m_shift, delta_mult, b_local, degrade_nodes, degrade_links,
                    lambda_bar, alpha, times[idx], cv, U_proc, idx, comp_index, EPS*scale)
            if shared_roads:
                for a,b in road_pairs:
                    H[a] = H[b] = min(H[a], H[b])
                    C[a], C[b] = Cnominal[a]*H[a], Cnominal[b]*H[b]
            if mode == "binary":
                actual_plan = {d: [(p, f*float(all(Z[c] > 0 for c in p))) for p,f in plan[d]] for d in DEMAND}
            delivered = {}
            for d in DEMAND:
                if arrival_delay:
                    for comps,flow in actual_plan[d]:
                        nodes = [c for c in comps if c in NODE_ROLE]
                        travel = sum(G.edges[u,v]["time"] for u,v in zip(nodes,nodes[1:]))
                        due = idx+max(0, int(math.ceil(travel/delta_t_D))-1)
                        transit[d][due] = transit[d].get(due, 0.)+flow*delta_t_D/Delta_t_R
                        dispatched_total += flow*delta_t_D/Delta_t_R
                    value = transit[d].pop(idx, 0.)
                    outstanding = math.fsum(g for g,_ in batches[d])+arrivals[d]
                    # Arrived stock exceeding current demand remains at destination.
                    served = min(value, outstanding)
                    if value > served:
                        transit[d][idx+1] = transit[d].get(idx+1,0.)+value-served
                    value = served
                else:
                    value = math.fsum(f for p,f in actual_plan[d])*delta_t_D/Delta_t_R
                delivered[d] = value
                served_total += value
                batches[d] = _fifo_update(batches[d], arrivals[d], value, delta_t_D)
            # Aggregate AFTER every destination is served; never inside its loop.
            Y[idx] = min(1., math.fsum(delivered.values())/total_demand) if total_demand > 0 else 1.
            cost = math.fsum(g*deprivation_phi(d, age) for d in DEMAND for g, age in batches[d])
            psi[idx] = cost/psi_ref if psi_ref > 0 else 0.
            if tc > horizon and Y[idx] < ycrit:
                tc = float(times[idx])
            if tdc > horizon and psi[idx] > pcrit:
                tdc = float(psi_times[idx])
    return {"times": times, "psi_times": psi_times, "Y": Y, "psi": psi,
            "Y_min": float(Y.min()), "psi_max": float(psi.max()), "tc": tc, "tdc": tdc,
            "H_initial": initial_H, "H_final": dict(H), "C_final": dict(C),
            "batches_final": {d: list(v) for d, v in batches.items()},
            "diagnostics": diagnostics, "path_pools": copy.deepcopy(path_pools),
            "health_history": np.asarray(history_H), "driver_history": np.asarray(history_A),
            "in_transit_final": {d: math.fsum(v.values()) for d,v in transit.items()},
            "initial_stock": initial_stock, "dispatched_total": dispatched_total, "served_total": served_total,
            "degradation_updates": nsub if mode != "binary" else 0}


# Explicit configurations, with legacy names retained for downstream scripts.
CELL_SPECS = {
    "M0_gamma_prop": dict(pi="split", m="soft"),
    "M1_mean": dict(pi="split", m="soft_mean"),
    "M2_static": dict(pi="split", m="static"),
    "M3_exogenous_match": dict(pi="split", m="exogenous_matched"),
    "M4_frozen_split": dict(pi="frozen_split", m="soft"),
    "M5_link_only": dict(pi="split", m="link_only"),
    "M6_node_only": dict(pi="split", m="node_only"),
    "soft_split": dict(pi="split", m="soft"),
    "soft_fastest": dict(pi="fastest", m="soft"),
    "binary_split": dict(pi="split", m="binary"),
    "binary_fastest": dict(pi="fastest", m="binary"),
}
CELLS = ["soft_split", "soft_fastest", "binary_split", "binary_fastest"]
MECH_CELLS = [c for c in CELL_SPECS if c.startswith("M")]


def _resolve_cell(cell):
    if isinstance(cell, str):
        if cell not in CELL_SPECS:
            raise ValueError(f"Unknown experiment: {cell}")
        pi, m = CELL_SPECS[cell]["pi"], CELL_SPECS[cell]["m"]
    else:
        pi, m = cell
    _mode(m)
    return dict(pi=pi, m=m, deg_nodes=m != "link_only", deg_links=m != "node_only",
                lam="matched" if m in {"exogenous", "exogenous_matched"} else "none")


def calibrate_exogenous_matched(N_cal=200, seed_cal=999, Delta_t_R=None, delta_t_D=None,
                                 gamma_mult=1., m_shift=0., delta_mult=1., alpha=None,
                                 cv=DEFAULT_CV, verbose=True, demand_mult=1., beta_mult=1.,
                                 horizon=None, return_report=False):
    """Fit constant M3 intensities to componentwise terminal mean health loss.

    Accounts for truncation at H=0 (loss/T alone does not). Calibration scenarios
    and endpoint uniforms are separate from evaluation. This matches a finite
    calibration sample's expected-loss surrogate, not time profiles or dependence.
    A fresh M3 evaluation and its observed residual are still necessary.
    """
    _ensure_loaded()
    if N_cal <= 0:
        raise ValueError("N_cal must be positive")
    horizon = T if horizon is None else horizon
    dt, sub = DEFAULT_DT if Delta_t_R is None else Delta_t_R, DEFAULT_DT_D if delta_t_D is None else delta_t_D
    initial, final = [], []
    endpoint_u = _uniforms(np.random.default_rng(np.random.SeedSequence([seed_cal, 991])), (N_cal, len(COMPONENTS)))
    for s in range(N_cal):
        resistance = sample_scenario(np.random.default_rng(np.random.SeedSequence([seed_cal, 101, s])), beta_mult)
        result = simulate_cell("split", "soft", resistance,
            np.random.default_rng(np.random.SeedSequence([seed_cal, 202, s])), dt, sub,
            gamma_mult, m_shift, delta_mult, demand_mult, alpha=alpha, cv=cv, horizon=horizon)
        initial.append([result["H_initial"][c] for c in COMPONENTS])
        final.append([result["H_final"][c] for c in COMPONENTS])
    h0, hf = np.asarray(initial), np.asarray(final)
    scales = build_b_dict(cv)
    rates, report = {}, {}
    for j, c in enumerate(COMPONENTS):
        target = max(0., float(np.mean(h0[:, j]-hf[:, j])))
        def predicted(rate):
            if rate == 0:
                return 0.
            increments = (np.full(N_cal, rate*horizon) if scales[c] == 0 else
                          _gamma_dist.ppf(endpoint_u[:, j], a=rate*horizon/scales[c], scale=scales[c]))
            return float(np.mean(np.minimum(h0[:, j], increments)))
        if target == 0:
            rate = 0.
        else:
            upper = max(1./horizon, target/horizon*2)
            while predicted(upper) < target-1e-12:
                upper *= 2
                if upper > 1e12:
                    raise ArithmeticError("M3 matching target cannot be bracketed")
            rate = upper if predicted(upper) <= target else brentq(lambda x: predicted(x)-target, 0., upper, xtol=1e-14)
        rates[c] = rate
        report[c] = {"lambda": rate, "target_health_loss": target,
                     "matched_health_loss": predicted(rate), "residual": predicted(rate)-target}
    if verbose:
        print(f"M3 independent calibration: N={N_cal}, max matching residual="
              f"{max(abs(v['residual']) for v in report.values()):.3g}")
    return (rates, report) if return_report else rates


def _wilson(indicators):
    n = len(indicators)
    p, z = float(np.mean(indicators)), float(norm.ppf(.975))
    denom = 1+z*z/n
    center = (p+z*z/(2*n))/denom
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return center-half, center+half


def run_factorial(N=N_MC, Delta_t_R=DEFAULT_DT, delta_t_D=1., beta_mult=1., demand_mult=1.,
                  cells=None, verbose=True, auto_calibrate_exo=True, m_shift=0., delta_mult=1.,
                  alpha=None, Y_crit=None, Psi_crit=None, cv=DEFAULT_CV, gamma_mult=1.,
                  lambda_bar=None, N_cal=200, seed=0, horizon=None):
    _ensure_loaded()
    if int(N) != N or N <= 0:
        raise ValueError("N must be a positive integer")
    N = int(N)
    horizon = T if horizon is None else horizon
    nr, steps = _time_grid(Delta_t_R, delta_t_D, horizon)
    cells = list(CELLS if cells is None else cells)
    if not cells or len(set(cells)) != len(cells):
        raise ValueError("Experiment labels must be nonempty and distinct")
    resolved = {c: _resolve_cell(c) for c in cells}
    needs_exo = any(v["lam"] == "matched" for v in resolved.values())
    if needs_exo and lambda_bar is None:
        if not auto_calibrate_exo:
            raise ValueError("M3 requested without prescribed intensity or calibration")
        lambda_bar = calibrate_exogenous_matched(N_cal=N_cal, seed_cal=seed+999,
            Delta_t_R=Delta_t_R, delta_t_D=delta_t_D, gamma_mult=gamma_mult,
            m_shift=m_shift, delta_mult=delta_mult, alpha=alpha, cv=cv, verbose=verbose,
            demand_mult=demand_mult, beta_mult=beta_mult, horizon=horizon)
    fields = ("Y_min", "psi_max", "tc", "tdc")
    rec = {cell: {f: np.empty(N) for f in fields} for cell in cells}
    ycrit, pcrit = Y_CRIT if Y_crit is None else Y_crit, PSI_CRIT if Psi_crit is None else Psi_crit
    for s in range(N):
        resistance = sample_scenario(np.random.default_rng(np.random.SeedSequence([seed, 501, s])), beta_mult)
        uniforms = _uniforms(np.random.default_rng(np.random.SeedSequence([seed, 502, s])), (nr*steps, len(COMPONENTS)))
        cache = {}
        for cell, spec in resolved.items():
            key = spec["pi"], spec["m"]
            if key not in cache:  # M0 and soft_split reuse the identical realization
                cache[key] = simulate_cell(spec["pi"], spec["m"], resistance, np.random.default_rng(0),
                    Delta_t_R, delta_t_D, gamma_mult, m_shift, delta_mult, demand_mult,
                    alpha=alpha, cv=cv, lambda_bar=lambda_bar if spec["lam"] == "matched" else None,
                    U_proc=uniforms, Y_crit=ycrit, Psi_crit=pcrit, horizon=horizon)
            for f in fields:
                rec[cell][f][s] = cache[key][f]
        if verbose and (s+1) % max(1, N//10) == 0:
            print(f"replicate {s+1}/{N}")
    summary = {}
    for cell, r in rec.items():
        r["I_sys"] = r["Y_min"] >= ycrit
        r["I_dc"] = r["psi_max"] <= pcrit
        r["replicate_id"] = np.arange(N)
        summary[cell] = {"R_sys": float(r["I_sys"].mean()), "R_dc": float(r["I_dc"].mean()),
                         "wilson_R": _wilson(r["I_sys"]), "wilson_Rdc": _wilson(r["I_dc"]),
                         "sentinel": horizon+delta_t_D, "Y_crit": ycrit, "Psi_crit": pcrit}
        for metric, field in (("R", "I_sys"), ("Rdc", "I_dc")):
            p = float(r[field].mean())
            summary[cell][f"hw_{metric}"] = (_wilson(r[field])[1]-_wilson(r[field])[0])/2
        for name, field in (("tc", "tc"), ("tdc", "tdc")):
            cross = r[field] <= horizon
            summary[cell].update({f"mean_{name}_uncond": float(r[field].mean()),
                f"mean_{name}_cond": float(r[field][cross].mean()) if cross.any() else float("nan"),
                f"frac_{name}": float(cross.mean()), f"N_cross_{name}": int(cross.sum())})
        both = (r["tc"] <= horizon) & (r["tdc"] <= horizon)
        delivery_only = (r["tc"] <= horizon) & (r["tdc"] > horizon)
        deprivation_only = (r["tc"] > horizon) & (r["tdc"] <= horizon)
        summary[cell].update(frac_both=float(both.mean()), frac_delivery_only=float(delivery_only.mean()),
            frac_deprivation_only=float(deprivation_only.mean()), frac_neither=float((~(both|delivery_only|deprivation_only)).mean()),
            mean_paired_delay=float(np.mean(r["tdc"][both]-r["tc"][both])) if both.any() else float("nan"))
    return summary, rec


def _indicators(record, endpoint, Y_crit=None, Psi_crit=None):
    key = "I_sys" if endpoint == "sys" else "I_dc"
    threshold = Y_crit if endpoint == "sys" else Psi_crit
    if threshold is None and key in record:
        return np.asarray(record[key], dtype=bool)
    if endpoint == "sys":
        return np.asarray(record["Y_min"]) >= (Y_CRIT if threshold is None else threshold)
    return np.asarray(record["psi_max"]) <= (PSI_CRIT if threshold is None else threshold)


def _paired_interval(values):
    """Paired normal interval for descriptive reporting, not sparse exact inference."""
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1)/math.sqrt(len(values))) if len(values) > 1 else float("nan")
    return mean, se, mean-1.96*se, mean+1.96*se


def _check_pairing(rec, labels):
    reference = rec[labels[0]].get("replicate_id")
    for label in labels:
        if reference is not None and not np.array_equal(reference, rec[label].get("replicate_id")):
            raise ValueError("Paired records have different replicate keys")


def paired_comparison(rec, cell_a, cell_b, N=None, Y_crit=None, Psi_crit=None):
    """A-minus-B convention: n10=A success/B failure; exact McNemar p."""
    _check_pairing(rec, [cell_a, cell_b])
    result = {}
    for endpoint, suffix in (("sys", "sys"), ("dc", "dc")):
        a, b = (_indicators(rec[c], endpoint, Y_crit, Psi_crit) for c in (cell_a, cell_b))
        if len(a) != len(b) or (N is not None and N != len(a)):
            raise ValueError("Pair lengths and N disagree")
        mean, se, _, _ = _paired_interval(a.astype(float)-b.astype(float))
        n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
        def cp(x):
            return (0. if x == 0 else float(beta_dist.ppf(.0125, x, len(a)-x+1)),
                    1. if x == len(a) else float(beta_dist.ppf(.9875, x+1, len(a)-x)))
        l10,h10 = cp(n10)
        l01,h01 = cp(n01)
        lo,hi = l10-h01,h10-l01
        result.update({f"delta_R_{suffix}": mean, f"se_R_{suffix}": se,
            f"ci_lo_R_{suffix}": lo, f"ci_hi_R_{suffix}": hi,
            f"n10_R_{suffix}": n10, f"n01_R_{suffix}": n01,
            f"mcnemar_p_R_{suffix}": float(binomtest(n10, n10+n01, .5).pvalue) if n10+n01 else 1.,
            f"sparse_pairs_R_{suffix}": n10+n01 < 25})
    return result


def paired_did(rec):
    _check_pairing(rec, CELLS)
    result = {}
    for endpoint in ("sys", "dc"):
        a, b, c, d = (_indicators(rec[label], endpoint).astype(float) for label in CELLS)
        mean, se, lo, hi = _paired_interval((a-c)-(b-d))
        result[endpoint] = {"DiD": mean, "paired_SE": se, "ci_lo": lo, "ci_hi": hi}
    return result


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=FIGDIR)
    parser.add_argument("--N", type=int, default=N_MC)
    parser.add_argument("--N-cal", type=int, default=200, help="Independent M3 calibration replicates")
    parser.add_argument("--routing-hours", type=float, default=DEFAULT_DT)
    parser.add_argument("--substep-hours", type=float, default=DEFAULT_DT_D)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cells", nargs="+", choices=list(CELL_SPECS), default=CELLS)
    parser.add_argument("--lambda-file", type=Path, help="JSON mapping of prescribed M3 intensities (per h)")
    args = parser.parse_args()
    load_data(args.data_dir)
    lam = json.loads(args.lambda_file.read_text()) if args.lambda_file else None
    calibration_report = None
    if any(_resolve_cell(c)["lam"] == "matched" for c in args.cells) and lam is None:
        lam, calibration_report = calibrate_exogenous_matched(
            N_cal=args.N_cal, seed_cal=args.seed+999, Delta_t_R=args.routing_hours,
            delta_t_D=args.substep_hours, return_report=True)
    summary, records = run_factorial(N=args.N, Delta_t_R=args.routing_hours, delta_t_D=args.substep_hours,
        cells=args.cells, seed=args.seed, lambda_bar=lam)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_dict(summary, orient="index").to_csv(args.output_dir/"summary.csv", index_label="cell")
    rows = [{"cell": cell, **{k: v[s].item() for k, v in record.items()}} for cell, record in records.items() for s in range(args.N)]
    pd.DataFrame(rows).to_csv(args.output_dir/"replicates.csv", index=False)
    (args.output_dir/"paths.json").write_text(json.dumps(export_paths(), indent=2))
    contrasts = {f"{a}-minus-{b}": paired_comparison(records, a, b) for a in args.cells for b in args.cells if a != b}
    if set(CELLS) <= set(args.cells):
        contrasts["factorial_DiD"] = paired_did(records)
    (args.output_dir/"paired_statistics.json").write_text(json.dumps(contrasts, indent=2))
    metadata = {"algorithm_reference": "revised_algorithms.tex, Algorithms 1--4",
        "seed": args.seed, "N": args.N, "horizon_h": T,
        "routing_hours": args.routing_hours, "substep_hours": args.substep_hours,
        "cells": args.cells, "PGA_mode": "fixed supplied component PGA",
        "CV": DEFAULT_CV, "CV_reference_stress": 1., "CV_reference_hours": 1.,
        "alpha_convention": "H0=(1-damage)**alpha; C(t)=C0*H(t)",
        "supply_budget_default": "nominal supply-node interval capacity; scenario assumption",
        "process_coupling": "indexed uniforms within a fixed grid; no cross-grid coupling claim",
        "M3_intensities_per_h": lam, "M3_calibration": calibration_report,
        "M3_calibration_N": args.N_cal if calibration_report is not None else None,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.data_dir/name for name in ("nodes_noto.csv", "links_noto.csv", "population_demand_noto.csv"))}}
    (args.output_dir/"run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(pd.DataFrame.from_dict(summary, orient="index")[["R_sys", "R_dc", "mean_tc_cond", "mean_tdc_cond"]])

if __name__ == "__main__":
    _main()
