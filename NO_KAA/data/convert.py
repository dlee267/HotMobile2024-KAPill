from pathlib import Path

locs = [
        "middle",
        "topright",
        "topleft",
        "bottomleft",
        "bottomright"
        ]
map = {
        "NA": "B1",
        "R1": "B2",
        "M1": "B3",
        }
for aug in ["M1", "NA", "R1"]:
    for loc in locs:
        fname = f"{aug}_{loc}.npy"
        fname3 = f"{map[aug]}_{loc}.npy"
        Path(f"Yihong1_{fname}").rename(f"no_KAA_{fname3}")

