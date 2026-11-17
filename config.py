"""Central config. Run everything with the `fog` conda env:
   C:\\Users\\aarus\\anaconda3\\envs\\fog\\python.exe  (Python 3.11, TF 2.15.1, tfmot 0.8.0)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT.parent / "sensor_data.csv"
OUT = ROOT / "outputs"
CACHE = OUT / "cache"
PLOTS = OUT / "plots"
EXPORT = OUT / "export"
for _p in (OUT, CACHE, PLOTS, EXPORT):
    _p.mkdir(parents=True, exist_ok=True)

# ---- signal / windowing -------------------------------------------------
SENSOR = "ankleL"                       # deployed sensor (matches hardware); "ankleR" for extra check
AXES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
CHANNELS = [f"{SENSOR}_{a}" for a in AXES]
FS_RAW, FS = 60, 50
RESAMPLE_UP, RESAMPLE_DOWN = 5, 6       # 60 -> 50 Hz, polyphase (anti-aliased FIR)
MATCH_HW_LPF = True                     # causal 2nd-order Butterworth @21 Hz to mimic MPU6050 DLPF
HW_LPF_HZ = 21.0
WIN, STRIDE = 100, 25

# ---- labels --------------------------------------------------------------
# fog == 1  <=>  fog_severity > 0 in this file (verified by crosstab). Window label = fog fraction >= FOG_FRAC.
FOG_FRAC = 0.5
# Windows with 0 < frac < FOG_FRAC are boundary windows. They stay NEGATIVE in evaluation. Training can drop them.
TRAIN_DROP_BOUNDARY = False
# activity-code semantics are NOT documented in the CSV. WALK_ACTIVITIES decides which time counts as
# "walking" for the false-alarms/hour denominator. None = every activity except 0 (set after running
# `python data.py --activity-report`, which prints ankle-gyro RMS per activity code).
WALK_ACTIVITIES = None

# ---- split ---------------------------------------------------------------
TEST_SUBJECTS = [18, 20, 21, 22]        # whole held-out subjects, touched once at the end
N_FOLDS = 5                             # subject-grouped, FoG-balanced folds over the 18 dev subjects
SEEDS = [0, 1, 2, 3, 4]
FEATURES = "raw6+mag"                   # "raw6" or "raw6+mag" (adds |acc| and |gyro|, cheap on ESP32)

# ---- MCU budget ----------------------------------------------------------
MAX_PARAMS = 150_000
MAX_INT8_KB = 150
MAX_MACS = 1_500_000
MAX_ARENA_KB = 200

# ---- training ------------------------------------------------------------
TRAIN = dict(
    epochs=40, batch=128, lr=2e-3, wd=1e-4, warmup_epochs=3, patience=8,
    label_smoothing=0.05, focal_gamma=2.0, pos_weight=True, balanced_sampling=True,
    ema=0.99, qat_epochs=6, qat_lr=2e-4,
)
AUG = dict(rot_deg=12.0, scale=0.0, noise=0.03, warp=0.10, shift=5, ch_drop=0.05, p=0.8)
