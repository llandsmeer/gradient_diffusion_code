import os

os.environ['JAX_PLATFORM'] = 'cpu'

import re
import numpy as np
import matplotlib.pyplot as plt
import arbor

morph = arbor.load_swc_arbor('./C4A.swc')

t = 671

cmap = plt.get_cmap('RdBu')
for branch_id in range(morph.num_branches):
    segments = morph.branch_segments(branch_id)
    for seg in segments:
        x1 = np.array([seg.dist.x, seg.dist.y, seg.dist.z])
        x2 = np.array([seg.prox.x, seg.prox.y, seg.prox.z])
        r1 = seg.dist.radius
        r2 = seg.prox.radius
        plt.plot([x1[0], x2[0]], [x1[1], x2[1]], color='black', lw=(r1+r2)/2)
plt.axis('equal')
plt.show()
