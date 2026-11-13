# -*- coding: utf-8 -*-
"""Report the pixel size and resolution of each exported figure."""
import glob
import os

from PIL import Image

FIG = r"C:\Users\danta\Doubao\chats\2026-11-07\new-chat\paper\figs"
for p in sorted(glob.glob(os.path.join(FIG, "*.tif"))):
    im = Image.open(p)
    dpi = im.info.get("dpi", (0, 0))
    print("%-30s %5dx%-5d px  dpi %.0f  %.2f x %.2f in  %.0f KB" % (
        os.path.basename(p), im.size[0], im.size[1], dpi[0],
        im.size[0] / dpi[0], im.size[1] / dpi[1], os.path.getsize(p) / 1024))
print()
for p in sorted(glob.glob(os.path.join(FIG, "*.pdf"))):
    data = open(p, "rb").read()
    print("%-30s %6.0f KB  images_in_pdf=%d" % (os.path.basename(p), len(data) / 1024,
                                               data.count(b"/Subtype /Image")))
