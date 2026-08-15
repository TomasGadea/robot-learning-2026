import zarr
import numpy as np
from pathlib import Path
zarr_dir = Path('datasets/raw/multi_cube')
stores = list(zarr_dir.rglob('*.zarr'))
total_red = 0
total_green = 0
total_blue = 0
total_eps = 0
for store_path in sorted(stores):
    z = zarr.open_group(str(store_path), mode='r')
    if 'meta' not in z or 'episode_ends' not in z['meta']:
        continue
    ends = z['meta']['episode_ends'][:]
    if len(ends) == 0:
        continue
    goals = z['data']['state_goal'][:]
    starts = np.concatenate([[0], ends[:-1]])
    
    for i, (s, e) in enumerate(zip(starts, ends)):
        goal = goals[s]
        if goal[0] > 0.5:
            total_red += 1
        elif goal[1] > 0.5:
            total_green += 1
        elif goal[2] > 0.5:
            total_blue += 1
    total_eps += len(ends)
print(f'Total episodes: {total_eps}')
print(f'  Red:   {total_red}')
print(f'  Green: {total_green}')
print(f'  Blue:  {total_blue}')
