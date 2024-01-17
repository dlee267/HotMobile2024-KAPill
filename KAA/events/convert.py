from pathlib import Path
args = ["NA", "R1", "M1"]
locs = ["middle", "topright", "topleft", "bottomright", "bottomleft"]

map = {
        "NA": "B1",
        "R1": "B2",
        "M1": "B3"
        }

for loc in locs:
    for arg in args:
        fname = f'Dong1_{arg}_{loc}.npy'
        Path(fname).rename(f"KAA_{arg}_{loc}.npy")
