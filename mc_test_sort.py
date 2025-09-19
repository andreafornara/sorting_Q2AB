# %%
# source /afs/cern.ch/work/a/afornara/public/new_HL_LHC/example_DA_study/miniforge/bin/activate
import numpy as np
import pandas as pd
import xtrack as xt
import time
from functions_sorting import *

# We analyze both beams
beams = ['lhcb1', 'lhcb2']

for beam in beams:
    # Baseline TFs (T/kA); 08-09-10 are extrapolated and will be biased upward
    TF_BASE = {
        "MQXFB03": 58.571,  # Q2A (fixed in R1)
        "MQXFB04": 58.655,  # Q2B (fixed in L5)
        "MQXFB05": 58.700,  # Q2A (fixed in L5)
        "MQXFB06": 58.523,  # Q2B (fixed in R1)
        "MQXFB07": 58.563,  # Q2A  (measured cold)
        "MQXFB08": 58.568,  # Q2B  (extrapolated)
        "MQXFB09": 58.643,  # Q2A  (extrapolated)
        "MQXFB10": 58.487,  # Q2B  (extrapolated)
    }

    # Options (Q2A, Q2B) assignments
    # These are all the possible combinations with fixed L5 and R1
    assignment_opt1 = {"l5": ("MQXFB05","MQXFB04"), "r1": ("MQXFB03","MQXFB06"),
                    "r5": ("MQXFB07","MQXFB08"), "l1": ("MQXFB09","MQXFB10")}
    assignment_opt2 = {"l5": ("MQXFB05","MQXFB04"), "r1": ("MQXFB03","MQXFB06"),
                    "r5": ("MQXFB07","MQXFB10"), "l1": ("MQXFB09","MQXFB08")}
    assignment_opt3 = {"l5": ("MQXFB05","MQXFB04"), "r1": ("MQXFB03","MQXFB06"),
                    "r5": ("MQXFB09","MQXFB08"), "l1": ("MQXFB07","MQXFB10")}
    assignment_opt4 = {"l5": ("MQXFB05","MQXFB04"), "r1": ("MQXFB03","MQXFB06"),
                    "r5": ("MQXFB09","MQXFB10"), "l1": ("MQXFB07","MQXFB08")}
    ASSIGNMENTS = {1: assignment_opt1, 2: assignment_opt2, 3: assignment_opt3, 4: assignment_opt4}

    # Monte Carlo settings (1 trial ~ 30s on 1 core)
    N_trials   = 100         # 
    seed       = 42          # fixed seed for reproducibility
    min_units  = 0.0         # minimum error, in units
    max_units  = 5.0         # maximum error, in units
    # Magnets to which we apply a +U units bias (U ~ U[0,5])
    magnets_extrapolated = ["MQXFB08","MQXFB09","MQXFB10"]
    rng = np.random.default_rng(seed)
    
    # Load config for rematch (crossing angles, tunes, chroma, ...)
    conf_knobs_and_tuning = load_knob_tuning_section("config.yaml")
    # rows for the final summary dataframe
    rows = []
    time_start = time.time()
    for trial in range(0, N_trials):
        print("------------------------------------------------------------")
        print(f"\n--- Trial {trial} of {N_trials} ---")
        # Perturb the TF with the chosen errors
        tf_mc = perturb_tf_units(TF_BASE, rng, min_units=min_units, max_units=max_units, magnets_extrapolated=magnets_extrapolated)
        print(f"Time elapsed: {time.time()-time_start:.1f} s")
        # Loop over the 4 options for each trial
        for opt in [1,2,3,4]:
            print(f"\n--- Option {opt} for trial {trial} ---")
            # New collider for each option (keeps runs independent)
            collider = xt.Multiline.from_json('HL_NOBB_september_CC_flatmachine.json')
            collider.build_trackers()
            tw_ref = collider[beam].twiss()
            # Check that qx and qy are as expected before applying errors
            print(f"Tunes before applying errors: qx={tw_ref.qx:.6f}, qy={tw_ref.qy:.6f}")

            # Collect all the element names in the four sides
            l5_magnets, r5_magnets, l1_magnets, r1_magnets = get_side_lists(collider, beam=beam)

            # Build relative-error map from the TFs for this option
            relmap = build_relerr_map(tf_mc, ASSIGNMENTS[opt])

            # Apply errors to the four sides
            apply_side_errors(collider[beam], l5_magnets, rel_for_a2=relmap["l5"]["a2"], rel_for_b2=relmap["l5"]["b2"])
            apply_side_errors(collider[beam], r1_magnets, rel_for_a2=relmap["r1"]["a2"], rel_for_b2=relmap["r1"]["b2"])
            apply_side_errors(collider[beam], r5_magnets, rel_for_a2=relmap["r5"]["a2"], rel_for_b2=relmap["r5"]["b2"])
            apply_side_errors(collider[beam], l1_magnets, rel_for_a2=relmap["l1"]["a2"], rel_for_b2=relmap["l1"]["b2"])

            # Rematch and Twiss
            collider = match_tune_and_chroma(collider, conf_knobs_and_tuning, match_linear_coupling_to_zero=False)
            tw_new = collider[beam].twiss()
            # Check that qx and qy are as expected after applying errors and rematching
            print(f"Tunes after applying all errors and rematching: qx={tw_new.qx:.6f}, qy={tw_new.qy:.6f}")
            # Retrieve beta-beating statistics
            avgx, stdx, maxx, avgy, stdy, maxy = beta_beat_stats(tw_ref, tw_new)

            # Store one line per (trial, option) for the final summary dataframe
            rows.append({
                "trial": trial,
                "option": opt,
                # errors applied
                "u08_units": round( (tf_mc["MQXFB08"]/TF_BASE["MQXFB08"] - 1.0) * 1e4, 3 ),
                "u09_units": round( (tf_mc["MQXFB09"]/TF_BASE["MQXFB09"] - 1.0) * 1e4, 3 ),
                "u10_units": round( (tf_mc["MQXFB10"]/TF_BASE["MQXFB10"] - 1.0) * 1e4, 3 ),
                # beta-beating statistics in %
                "avg_bbx_pct": avgx, "std_bbx_pct": stdx, "max_bbx_pct": maxx,
                "avg_bby_pct": avgy, "std_bby_pct": stdy, "max_bby_pct": maxy,
            })
    # Save compact summary as dataframe
    df_out = pd.DataFrame(rows)
    if beam == 'lhcb1':
        df_out.to_csv("beta_beat_summary_mc_b1_long.csv", index=False)
        print("Saved: beta_beat_summary_mc_b1_long.csv")
    else:
        df_out.to_csv("beta_beat_summary_mc_b2_long.csv", index=False)
        print("Saved: beta_beat_summary_mc_b2_long.csv")
