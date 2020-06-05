from typing import Optional

import torch


def levenshtein_distance(r: str, h: str, device: Optional[str] = None):

    # initialisation
    d = torch.zeros((2, len(h) + 1), dtype=torch.long)  # , device=device)
    dold = 0
    dnew = 1

    # computation
    for i in range(1, len(r) + 1):
        d[dnew, 0] = 0
        for j in range(1, len(h) + 1):

            if r[i - 1] == h[j - 1]:
                d[dnew, j] = d[dnew - 1, j - 1]
            else:
                substitution = d[dnew - 1, j - 1] + 1
                insertion = d[dnew, j - 1] + 1
                deletion = d[dnew - 1, j] + 1
                d[dnew, j] = min(substitution, insertion, deletion)

        dnew, dold = dold, dnew

    return d[dnew, -1].item()
