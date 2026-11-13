# -*- coding: utf-8 -*-
"""Does per-operating-point normalisation leave a signal the protocol can score?"""
import numpy as np

import public_domain
from train import build_data


def report(subset, condnorm):
    d = public_domain.make(subset=subset, condnorm=condnorm)
    dd = build_data(d, 0, 0, d.steps, 0)
    Z = dd["yte"].cpu().numpy()
    rms = np.sqrt((Z ** 2).mean())
    one = np.sqrt((np.diff(Z, axis=1) ** 2).mean()) / rms
    lag1 = np.mean([np.corrcoef(Z[:, :-1, j].ravel(), Z[:, 1:, j].ravel())[0, 1]
                    for j in range(Z.shape[-1])])
    pers = [np.sqrt(((Z[:, 1:h + 1, :] - Z[:, :1, :]) ** 2).mean()) / rms for h in (25, 50, 100)]
    print("%s condnorm=%-5s one-step %.4f  lag1 %.3f  persistence @25/50/100 %.3f %.3f %.3f"
          % (subset, condnorm, one, lag1, pers[0], pers[1], pers[2]))
    print("   normalised observed std %s" % np.round(Z.reshape(-1, Z.shape[-1]).std(0), 3))


report("FD001", False)
report("FD002", False)
report("FD002", True)

raw = public_domain._records(public_domain.os.path.join(public_domain.DATA, "train_FD002.txt"))
held = raw[-16:]
print("\nT24 at a fixed operating point, 16 held-out engines:")
for c in range(6):
    within, across = [], []
    for S, U in held:
        lab = public_domain.condition_labels(U)
        seg = S[lab == c][:, 1]
        if len(seg) > 1:
            within.append(np.std(seg))
            across.append(np.mean(seg))
    print("  cond %d  within-engine std %.3f  across-engine std %.3f  spread over life %.1f"
          % (c, np.mean(within), np.std(across),
             np.mean([np.max(S[public_domain.condition_labels(U) == c][:, 1]) -
                      np.min(S[public_domain.condition_labels(U) == c][:, 1])
                      for S, U in held if (public_domain.condition_labels(U) == c).sum() > 1])))
