"""Check whether rows are contiguous per subject and how labels change."""
import csv
from collections import Counter

path = r"C:/Users/aarus/Parkinson's-FoG-v2/sensor_data.csv"

with open(path, newline="") as f:
    r = csv.reader(f)
    header = next(r)
    idx = {name: i for i, name in enumerate(header)}
    isub, isess, itask = idx["subjectID"], idx["sessionID"], idx["taskID"]
    ia, ifog, isev = idx["activity"], idx["fog"], idx["fog_severity"]

    prev = None
    switches = Counter()
    runs = []  # (subj, sess, task, n, fog_frac)
    cur_key = None
    cur_n = 0
    cur_fog = 0
    fog_runs = []  # consecutive fog==1 lengths
    fog_run = 0
    prev_fog_key = None

    for row in r:
        key = (int(float(row[isub])), int(float(row[isess])), int(float(row[itask])))
        fog = int(float(row[ifog]))
        if key != cur_key:
            if cur_key is not None:
                runs.append((cur_key, cur_n, cur_fog))
            if prev is not None and key[0] != prev[0]:
                switches["subject"] += 1
            if prev is not None and key[1] != prev[1]:
                switches["session"] += 1
            if prev is not None and key[2] != prev[2]:
                switches["task"] += 1
            cur_key = key
            cur_n = 0
            cur_fog = 0
        cur_n += 1
        cur_fog += fog
        fog_key = key
        if fog == 1 and fog_key == prev_fog_key:
            fog_run += 1
        else:
            if fog_run:
                fog_runs.append(fog_run)
            fog_run = 1 if fog == 1 else 0
        prev_fog_key = fog_key
        prev = key
    if cur_key is not None:
        runs.append((cur_key, cur_n, cur_fog))
    if fog_run:
        fog_runs.append(fog_run)

print("block switches", dict(switches))
print("n contiguous (subj,sess,task) blocks", len(runs))
print("first 15 blocks (subj,sess,task, rows, fog):")
for b in runs[:15]:
    print(" ", b)
print("...")
print("last 5:")
for b in runs[-5:]:
    print(" ", b)

# subject order of first appearance
seen = []
for (k, n, g) in runs:
    if k[0] not in seen:
        seen.append(k[0])
print("subject first-appearance order", seen)

# blocks per subject
from collections import defaultdict
per = defaultdict(int)
for (k, n, g) in runs:
    per[k[0]] += 1
print("blocks per subject", dict(sorted(per.items())))

fog_runs.sort()
print("n fog episodes (contiguous fog==1 within a block)", len(fog_runs))
if fog_runs:
    print("fog run samples: min", fog_runs[0], "median", fog_runs[len(fog_runs)//2], "max", fog_runs[-1])
    print("fog run seconds @60Hz: min", fog_runs[0]/60, "median", fog_runs[len(fog_runs)//2]/60, "max", fog_runs[-1]/60)
    # histogram coarse
    bins = [0, 30, 60, 120, 300, 600, 10**9]
    labels = ["<0.5s", "0.5-1s", "1-2s", "2-5s", "5-10s", ">10s"]
    hist = Counter()
    for L in fog_runs:
        for i in range(len(labels)):
            if bins[i] < L <= bins[i+1] or (i == 0 and L <= bins[1]):
                # fix
                pass
    hist = Counter()
    for L in fog_runs:
        if L < 30:
            hist["<0.5s"] += 1
        elif L < 60:
            hist["0.5-1s"] += 1
        elif L < 120:
            hist["1-2s"] += 1
        elif L < 300:
            hist["2-5s"] += 1
        elif L < 600:
            hist["5-10s"] += 1
        else:
            hist[">10s"] += 1
    print("episode duration hist", dict(hist))
