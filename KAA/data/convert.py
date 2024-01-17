from pathlib import Path

locs = [
        "middle",
        "topright",
        "topleft",
        "bottomleft",
        "bottomright"
        ]

for aug in ["M1", "NA"]:
    for loc in locs:
        fname = f"{aug}_{loc}.npy"
        Path(f"Dong12_{fname}").rename(f"KAA_{fname}")

for aug in ["R1"]:
    for loc in locs:
        fname = f"{aug}_{loc}.npy"
        Path(f"Dong1_{fname}").rename(f"KAA_{fname}")
