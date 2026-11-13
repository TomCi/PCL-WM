# -*- coding: utf-8 -*-
"""Copy the seven manuscript figures into the submission set and rebuild the zip."""
import os
import shutil
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(HERE, "figs")
DST = os.path.join(ROOT, "figures_submission")
ZIP = os.path.join(ROOT, "PCL-WM_figures_elsevier.zip")

NAMES = {
    "fig1_architecture": "fig1",
    "fig2_error_growth": "fig2",
    "fig3_beta": "fig3",
    "fig4_ablation": "fig4",
    "fig5_data_efficiency": "fig5",
    "fig6_closedloop": "fig6",
    "fig7_public": "fig7",
}

os.makedirs(DST, exist_ok=True)
for stem, short in NAMES.items():
    for ext in ("pdf", "tif"):
        shutil.copyfile(os.path.join(SRC, stem + "." + ext),
                        os.path.join(DST, short + "." + ext))

GA = "fig_graphical_abstract"
for ext in ("pdf", "tif"):
    shutil.copyfile(os.path.join(SRC, GA + "." + ext),
                    os.path.join(DST, "graphical_abstract." + ext))

with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for short in sorted(NAMES.values()):
        for ext in ("pdf", "tif"):
            z.write(os.path.join(DST, short + "." + ext), short + "." + ext)
    for ext in ("pdf", "tif"):
        z.write(os.path.join(DST, "graphical_abstract." + ext), "graphical_abstract." + ext)

print("exported", len(NAMES) * 2 + 2, "files")
print("zip bytes", os.path.getsize(ZIP))
