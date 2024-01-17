import numpy as np
from pathlib import Path
from scipy import signal
import matplotlib.pyplot as plt

fill_lvls = [f'MP{x}' for x in range(1,8)]
# fill_lvls.remove("MP3")
# fill_lvls = list()
# fill_lvls.append("KAA")
# fill_lvls.append("no_KAA")

augments = ["NA", "R1", "M1"]
# augments = ["NA", "M1"]
locations = ["middle", "topright","topleft","bottomright","bottomleft"]
# locations = ["middle"]
raw_data_dir = Path("./MP_data/")
# output_data_dir = Path("./MP_raw/")
# if not output_data_dir.exists():
#     output_data_dir.mkdir()
for fill_lvl in fill_lvls:
    for augment in augments:
        for location in locations:
            fname = f'{fill_lvl}_{augment}_{location}.npy'
            data = np.load(raw_data_dir / fname)
            peaks, _ = signal.find_peaks(data, 630, distance=2000)
            print(fname, len(peaks))
            o = list()
            if peaks[-1] + 2000 > len(data):
                data = list(data)
                data.extend([np.average(data)] * (peaks[-1] + 2000 - len(data)))
            for peak in peaks[::]:
                # print(peak)
                o.append(data[peak-200:peak+2000])
            # plt.plot(o[-1])
            # plt.show()
            o = np.array(o)
            # fname = f'{fill_lvl}_{augment}_{location}.npy'
            # fname = f'{fill_lvl}_{augment}_{location}.npy'
            output_data_dir = Path("./MP_raw/") / fill_lvl
            if not output_data_dir.exists():
                output_data_dir.mkdir()
            np.save(output_data_dir / fname, o)
