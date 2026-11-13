# -*- coding: utf-8 -*-
"""Draw the four data figures in the only order that yields a consistent set.

analyze_v2.py also writes v2_ablation.png (single seed, registry labels); the
two-seed version overwrites it, so it must run last among the two.
"""
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
for script in ("analyze_v2.py", "fig_ablation_2seed.py", "fig_data_eff.py"):
    print("running", script)
    subprocess.run([sys.executable, os.path.join(HERE, script)], check=True,
                   cwd=HERE, capture_output=True)
print("figures_v2 rebuilt in order")
