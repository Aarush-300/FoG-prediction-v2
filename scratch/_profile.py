"""Profile sensor_data.csv without pandas (broken zstandard namespace)."""
import csv
from collections import Counter

path = r"C:/Users/aarus/Parkinson's-FoG-v2/sensor_data.csv"

activity = Counter()
fog = Counter()
sev = Counter()
subj = Counter()
sess = Counter()
task = Counter()
fog_sev = Counter()
act_fog = Counter()
act_sev = Counter()
subj_fog = Counter()
subj_sev = Counter()
subj_act = Counter()
subj_sess = Counter()
sess_sev = Counter()
sess_fog = Counter()
sess_act = Counter()
sess_subj = {}
ts_diff = Counter()
prev_ts = None
n = 0
ts_min = None
ts_max = None

with open(path, newline="") as f:
    r = csv.reader(f)
    header = next(r)
    idx = {name: i for i, name in enumerate(header)}
    print("ncols", len(header))
    print("header", header)
    ia, ifog, isev = idx["activity"], idx["fog"], idx["fog_severity"]
    isub, isess, itask = idx["subjectID"], idx["sessionID"], idx["taskID"]
    its = idx["timestamp"]
    for row in r:
        n += 1
        a = int(float(row[ia]))
        g = int(float(row[ifog]))
        s = int(float(row[isev]))
        u = int(float(row[isub]))
        e = int(float(row[isess]))
        t = int(float(row[itask]))
        ts = float(row[its])
        activity[a] += 1
        fog[g] += 1
        sev[s] += 1
        subj[u] += 1
        sess[e] += 1
        task[t] += 1
        fog_sev[(g, s)] += 1
        act_fog[(a, g)] += 1
        act_sev[(a, s)] += 1
        subj_fog[(u, g)] += 1
        subj_sev[(u, s)] += 1
        subj_act[(u, a)] += 1
        subj_sess[(u, e)] += 1
        sess_sev[(e, s)] += 1
        sess_fog[(e, g)] += 1
        sess_act[(e, a)] += 1
        sess_subj.setdefault(e, set()).add(u)
        if ts_min is None or ts < ts_min:
            ts_min = ts
        if ts_max is None or ts > ts_max:
            ts_max = ts
        if prev_ts is not None:
            d = round(ts - prev_ts, 6)
            if d > 0:
                ts_diff[d] += 1
        prev_ts = ts

print("nrows", n)
print("ts", ts_min, ts_max)
print("ts_diff top", ts_diff.most_common(6))


def show(title, counter, keys=None):
    print("\n==", title)
    items = counter.most_common() if keys is None else [(k, counter[k]) for k in keys]
    for k, v in sorted(items, key=lambda kv: kv[0] if not isinstance(kv[0], tuple) else kv[0]):
        print(f"  {k}: {v}")


show("activity", activity)
show("fog", fog)
show("fog_severity", sev)
show("subjectID", subj)
show("sessionID", sess)
show("taskID", task)
show("fog x severity", fog_sev)
show("activity x fog", act_fog)
show("activity x severity", act_sev)

print("\n== per subjectID")
for u in sorted(subj):
    n_u = subj[u]
    fog1 = subj_fog[(u, 1)]
    sevs = {s: subj_sev[(u, s)] for s in sorted(sev)}
    acts = {a: subj_act[(u, a)] for a in sorted(activity)}
    sessions = sorted(e for (uu, e) in subj_sess if uu == u)
    print(f"  subj {u}: rows={n_u} fog1={fog1} ({100*fog1/n_u:.2f}%) sev={sevs} act={acts} sessions={sessions}")

print("\n== per sessionID")
for e in sorted(sess):
    n_e = sess[e]
    fog1 = sess_fog[(e, 1)]
    sevs = {s: sess_sev[(e, s)] for s in sorted(sev)}
    acts = {a: sess_act[(e, a)] for a in sorted(activity)}
    print(f"  sess {e}: rows={n_e} subjects={sorted(sess_subj[e])} fog1={fog1} ({100*fog1/n_e:.2f}%) sev={sevs} act={acts}")
