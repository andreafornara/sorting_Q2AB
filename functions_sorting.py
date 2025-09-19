# %%
import xtrack as xt
import numpy as np
import pandas as pd
import xpart as xp
import xobjects as xo
import yaml
import json
from xtrack.twiss import twiss_line
from dataclasses import dataclass
from typing import Dict, Tuple, List
import pandas as pd
from itertools import product
import xmask as xm
import ruamel.yaml

# ---- 1) Utilities to compute pair averages and relative errors ----
@dataclass(frozen=True)
class MagnetInfo:
    name: str         # "MQXFB0X"
    role: str         # "Q2A" or "Q2B"  (A→a2, B→b2)

# Return signed relative errors (dev/avg) for A and B of a Q2A/Q2B pair.
def pair_rel_errors(a: MagnetInfo, b: MagnetInfo, tf: Dict[str, float]):
    TF_A = tf[a.name]
    TF_B = tf[b.name]
    avg = 0.5*(TF_A + TF_B)
    rel_A = (TF_A - avg) / avg
    rel_B = (TF_B - avg) / avg
    return {"A_rel": rel_A, "B_rel": rel_B, "avg": avg}

# Simple function to build a map of relative errors from a TF dict and an option assignment
def build_relerr_map(tf: Dict[str,float], assignment: Dict[str, Tuple[str, str]]):
    relmap = {}
    for side, (nameA, nameB) in assignment.items():
        rels = pair_rel_errors(MagnetInfo(nameA,"Q2A"), MagnetInfo(nameB,"Q2B"), tf)
        relmap[side] = {"a2": rels["A_rel"], "b2": rels["B_rel"], "avg_TF": rels["avg"]}
    return relmap

# Collect all the element names in the four sides
def get_side_lists(collider, beam="lhcb1"):
    l5_magnets, r5_magnets, l1_magnets, r1_magnets = [], [], [], []
    line = collider[beam]
    for element in line.element_names:
        if element.startswith('mqxf') and not (element.endswith('fl') or element.endswith('fr')):
            try:
                if line[element].__class__.__name__ == 'Multipole':
                    if 'a2' in element or 'b2' in element:
                        if   'l5' in element: l5_magnets.append(element)
                        elif 'r5' in element: r5_magnets.append(element)
                        elif 'l1' in element: l1_magnets.append(element)
                        elif 'r1' in element: r1_magnets.append(element)
            except Exception:
                pass
    return l5_magnets, r5_magnets, l1_magnets, r1_magnets

# ---- 2) Apply errors to all magnets in a group (A,B) ----
# Scale the quad terms (knl[1]) of a Multipole element by (1+rel)
def scale_quad_terms(elem, rel):
    knl = np.array(elem.knl, dtype=float)
    if len(knl) > 1: knl[1] = knl[1] * (1.0 + rel)
    elem.knl = knl
    return knl

# From a list of element names, apply the corresponding relative errors
def apply_side_errors(line, element_names: List[str], rel_for_a2: float, rel_for_b2: float):
    """
    element_names: list like l5_magnets / r5_magnets / l1_magnets / r1_magnets
    rel_for_a2:    signed relative error to apply to 'a2' slices (Q2A)
    rel_for_b2:    signed relative error to apply to 'b2' slices (Q2B)
    """
    for en in element_names:
        if 'a2' in en:
            rel = rel_for_a2
        elif 'b2' in en:
            rel = rel_for_b2
        else:
            # safety: skip non-a2/b2 entries if any
            continue
        elem = line[en]
        line[en].knl = scale_quad_terms(elem, rel)

ryaml = ruamel.yaml.YAML()

# ---- 3) Rematching and Twiss ----
# Load the knob/tuning section from config file
def load_knob_tuning_section(config_path="config.yaml"):
    """Load the knob/tuning part used by machine_tuning."""
    with open(config_path, "r") as f:
        cfg_all = ryaml.load(f)
    # same structure your big script expects:
    return cfg_all["config_collider"]["config_knobs_and_tuning"]

# Function to rematch tunes and chroma from xmask
def match_tune_and_chroma(collider, conf_knobs_and_tuning, match_linear_coupling_to_zero=True):
    # Tunings (same as your builder)
    for line_name in ["lhcb1", "lhcb2"]:
        knob_names = conf_knobs_and_tuning["knob_names"][line_name]
        targets = {
            "qx":  conf_knobs_and_tuning["qx"][line_name],
            "qy":  conf_knobs_and_tuning["qy"][line_name],
            "dqx": conf_knobs_and_tuning["dqx"][line_name],
            "dqy": conf_knobs_and_tuning["dqy"][line_name],
        }

        # Ensure co_ref exists (some JSON dumps include it; if missing, clone)
        co_ref_name = f"{line_name}_co_ref"
        if co_ref_name not in collider.line_names:
            collider[co_ref_name] = collider[line_name].copy()

        xm.machine_tuning(
            line=collider[line_name],
            enable_closed_orbit_correction=True,
            enable_linear_coupling_correction=match_linear_coupling_to_zero,
            enable_tune_correction=True,
            enable_chromaticity_correction=True,
            knob_names=knob_names,
            targets=targets,
            line_co_ref=collider[co_ref_name],
            co_corr_config=conf_knobs_and_tuning["closed_orbit_correction"][line_name],
        )
    return collider

# ---- 4) Monte Carlo functions ----
#  Return a new TF dict with +U units on extrapolated magnets (U ~ U[0,5]) for 08-09-10
def perturb_tf_units(tf_base, rng, min_units=0.0, max_units=5.0, magnets_extrapolated=["MQXFB08","MQXFB09","MQXFB10"]):
    tf = dict(tf_base)
    for m in magnets_extrapolated:
        u = rng.uniform(min_units, max_units)       
        rel = u * 1e-4                              
        tf[m] = tf[m] * (1.0 + rel)            # The extrapolated TF is always less than the real one     
    return tf

# Retrieve beta-beating statistics
def beta_beat_stats(twiss_ref, twiss_new):
    bbx = (twiss_new['betx'] - twiss_ref['betx']) / twiss_ref['betx']
    bby = (twiss_new['bety'] - twiss_ref['bety']) / twiss_ref['bety']
    # return percentages
    return (100*np.mean(np.abs(bbx)), 100*np.std(np.abs(bbx)), 100*np.max(np.abs(bbx)),
            100*np.mean(np.abs(bby)), 100*np.std(np.abs(bby)), 100*np.max(np.abs(bby)))