import numpy as np
from glob import glob

if __name__ == "__main__":
    dir_read = "data/7"
    files = glob(f"{dir_read}/*.dat")
    helices = {}
    sheets = {}
    for file in files:
        with open(file, "r") as f:
            l = 0
            prev_ss = "-"
            helix = []
            sheet = []
            for line in f.readlines():
                line_split = line.split()
                if len(line_split) < 9:
                    continue
                ss = line_split[4]
                if ss != prev_ss:
                    if prev_ss == "H":
                        if l in helices:
                            a = np.stack(helix, axis=0)
                            a = a - a[0]
                            helices[l].append(a)
                        else:
                            a = np.stack(helix, axis=0)
                            a = a - a[0]
                            helices[l] = [a]
                        helix = []
                    elif prev_ss == "S":
                        if l in sheets:
                            a = np.stack(sheet, axis=0)
                            a = a - a[0]
                            sheets[l].append(a)
                        else:
                            a = np.stack(sheet, axis=0)
                            a = a - a[0]
                            sheets[l] = [a]
                        sheet = []
                    l = 0
                if ss == "H":
                    a = np.array([float(i) for i in line_split[5:8]])
                    helix.append(a)
                elif ss == "S":
                    sheet.append(a)
                l += 1
                prev_ss = ss
    for key, value in helices.items():
        np.save(f"validation/helix_{key}.npy", value)
    for key, value in sheets.items():
        np.save(f"validation/sheet_{key}.npy", value)