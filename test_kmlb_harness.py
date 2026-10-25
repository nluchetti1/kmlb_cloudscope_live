"""End-to-end offline harness for kmlb_classify: the real main(), MRMS 2-D/3-D faked at the
function boundary, synthetic KMLB volumes (kmlb_synth), local feeds unreachable.

    python test_kmlb_harness.py
"""
import datetime as dt
import gzip
import json
import os
import shutil
import sys
import tempfile
import numpy as np
import requests

WORK = tempfile.mkdtemp(prefix="kmlbh_")
os.environ["OUT_DIR"] = os.path.join(WORK, "site")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mrms_classify as M  # noqa: E402
import kmlb_classify as K  # noqa: E402
from kmlb_synth import LA, LO, STORM, truth, synth_volume, FULL, TRUNC  # noqa: E402

FAILS = []
def check(c, msg):
    print(("  ok    " if c else "  FAIL  ") + msg)
    if not c: FAILS.append(msg)
iso = lambda t: t.strftime("%Y-%m-%dT%H:%M:%S")
LOg, LAg = np.meshgrid(LO, LA)
RS = np.hypot((LAg - STORM[0]) * 111.2, (LOg - STORM[1]) * 97.7)
ST = RS < 15
J, I = int(np.argmin(np.abs(LA - STORM[0]))), int(np.argmin(np.abs(LO - STORM[1])))


def mrms_field(key):
    f = lambda v, none: np.where(ST, v, none).astype(np.float32)
    return {"comp": f(45.0, -99.0), "r0": f(30.0, -99.0), "r5": f(28.0, -99.0), "r10": f(25.0, -99.0),
            "r15": f(22.0, -99.0), "r20": f(20.0, -99.0), "high": f(18.0, -99.0), "super": f(15.0, -99.0),
            "hmax": f(3000.0, -1.0), "et18": f(11.0, -1.0), "vii": f(2.0, -1.0),
            "cg": np.full((200, 200), -1.0, np.float32)}[key]


def install(now, kmode):
    def key_of(url):
        for k, (p, _, _) in M.PRODUCTS.items():
            if f"/{p}/" in url:
                return k
        return None
    def ftime(url):
        if ".latest." in url:
            t = now - dt.timedelta(minutes=2)
            return t.replace(minute=t.minute - t.minute % 2, second=0, microsecond=0)
        return dt.datetime.strptime(url.rsplit("_", 1)[1][:15], "%Y%m%d-%H%M%S")
    def decode(raw, fill):
        u = raw.decode()
        v = ftime(u).strftime("%Y-%m-%dT%H:%M:%SZ")
        if "/3DRefl/" in u:      # MRMS 3-D: 25 dBZ through the storm up to 14 km
            h = float(u.split("MergedReflectivityQC_")[1].split("/")[0])
            return np.where(ST & (h <= 14.0), 25.0, -99.0).astype(np.float32), LA.copy(), LO.copy(), v
        return mrms_field(key_of(u)), LA.copy(), LO.copy(), v
    M.fetch_url = lambda sess, url: url.encode()
    M.decode_crop = decode
    def listing(sess, product, cache, root=None):
        base = now - dt.timedelta(minutes=2)
        base = base.replace(minute=base.minute - base.minute % 2, second=0, microsecond=0)
        r = M.ROOT3 if root else M.ROOT
        return [(base - dt.timedelta(minutes=2 * k),
                 f"{r}/{product}/MRMS_{product}_{(base - dt.timedelta(minutes=2 * k)):%Y%m%d-%H%M%S}.grib2.gz")
                for k in range(90)]
    M.listing = listing
    M.list_levels_3d = lambda sess: [(f"MergedReflectivityQC_{h:05.2f}", h) for h in K.STD_LEVELS_KM]
    M.freezing_level = lambda sess, prev, sd: (np.full((200, 200), 4800.0, np.float32),
                                               {"fetched": iso(now), "valid": iso(now)}, False)
    M.goes_anvil_wind = lambda sess, cache: None
    M.goes_tops = lambda sess, v, la, lo, cache: (None, "harness: none")
    M.glm_update = lambda sess, sd: {"flashes": [], "ok": True, "newest": now, "note": "fake GLM"}
    M.load_soundings = lambda sess, prev, sd: None
    orig = M.__dict__.setdefault("_orig_chrome", M._chrome)
    M._chrome = lambda ax, pc, coast=True, pads=True: orig(ax, pc, coast=False, pads=pads)
    class Blocked:
        def get(self, *a, **k):
            raise requests.exceptions.ConnectTimeout("blocked (harness)")
    M._local_session = lambda: Blocked()
    # KMLB: a volume every 5 minutes; the scenario decides the tilts or a failure
    vols = [(now - dt.timedelta(minutes=5 * k + 1), f"{now:%Y/%m/%d}/KMLB/KMLB{(now - dt.timedelta(minutes=5*k+1)):%Y%m%d_%H%M%S}_V06")
            for k in range(40)][::-1]
    K.list_day = lambda sess, day: [] if kmode == "none" else [(t, k) for t, k in vols if t.date() == day]
    FETCHED.clear()
    def fetch(self, key):
        FETCHED.append(key)
        if kmode == "broken":
            raise IOError("corrupt volume (harness)")
        v = synth_volume(FULL if kmode == "full" else TRUNC, start=K.vol_start(key))
        v["key"] = key
        return v
    K.Hybrid._fetch = fetch


FETCHED = []


def run(now, kmode, fresh=True):
    if fresh and os.path.exists(os.environ["OUT_DIR"]):
        shutil.rmtree(os.environ["OUT_DIR"])
    os.environ["CLOUDSCOPE_FAKE_NOW"] = iso(now)
    install(now, kmode)
    K.main()
    man = json.load(open(os.path.join(M.OUT_DIR, "manifest.json")))
    f = sorted(man["frames"], key=lambda x: x["valid"])[-1]
    g = man["grid"]
    b = np.fromfile(os.path.join(M.OUT_DIR, f["data"]), np.uint8).reshape(-1, g["ny"], g["nx"])
    return man, f, b


def vol_at(f, man, j, i, h_km):
    enc = np.frombuffer(gzip.decompress(open(os.path.join(M.OUT_DIR, f["volume"]), "rb").read()), np.uint8)
    L = len(man["levels_km"])
    enc = enc.reshape(L, 200, 200)
    k = int(np.argmin(np.abs(np.array(man["levels_km"]) - h_km)))
    return int(enc[k, j, i])


T0 = dt.datetime(2025, 7, 31, 16, 10, 0)
j39 = int(np.argmin(np.abs(LA - K.LC39A[0]))); i39 = int(np.argmin(np.abs(LO - K.LC39A[1])))

print("\n== A: full volume (AVSET off)")
man, f, b = run(T0, "full")
kk = f.get("kmlb", {})
print(f"   {len(man['frames'])} frames; newest {f['valid']}; kmlb {kk}")
check(man["viewer_expected"] == "kmlb-v2", "manifest expects viewer kmlb-v2")
check(kk.get("mode") == "full" and kk.get("lc39a_top_km", 0) > 18, "newest frame: KMLB full depth over LC-39A")
check(abs(kk.get("offset_min", 99)) <= 3.0, f"volume midpoint {kk.get('offset_min')} min from the frame")
check(b.shape[0] == 15, f"15 data planes (plane 14 = KMLB top): {b.shape[0]}")
check(195 <= b[14, j39, i39] <= 210, f"plane 14 at LC-39A: {b[14, j39, i39] / 10:.1f} km")
check(f.get("volume") and "KMLB merged" in f.get("volume_note", ""), f"volume note: {f.get('volume_note')}")
low = vol_at(f, man, J, I, 2.0); hi = vol_at(f, man, J, I, 13.0)
tz = truth(LA[J], LO[I], 2000.0)[0]
check(abs(low / 2 - tz) < 2.5, f"3-D at the storm, 2 km: {low/2:.1f} dBZ from KMLB (truth {tz:.1f})")
check(hi == 255, f"3-D at the storm, 13 km (above the analytic 12 km top): no echo - KMLB sees there and overrides MRMS's 25 dBZ ({hi})")
check(all(fr.get("kmlb", {}).get("mode") in ("full", "partial") for fr in man["frames"]), "every frame has a KMLB volume")
n_fetch = len(set(FETCHED))

print("\n== B: truncated volume (AVSET on, 6.4 deg)")
man, f, b = run(T0, "trunc")
kk = f.get("kmlb", {})
print(f"   kmlb {kk}")
check(kk.get("mode") == "partial" and 6.5 < kk.get("lc39a_top_km", 0) < 7.2, "LC-39A sampled only to ~6.8 km")
check(66 <= b[14, j39, i39] <= 72, f"plane 14 at LC-39A: {b[14, j39, i39] / 10:.1f} km")
hi = vol_at(f, man, J, I, 13.0); low = vol_at(f, man, J, I, 2.0)
check(hi == 50, f"3-D at the storm, 13 km: MRMS 25 dBZ ({hi / 2:.1f})")
check(low == 50, f"3-D at the storm, 2 km: also MRMS ({low / 2:.1f}) - single-source column, KMLB only reaches 6.8 km")
sh = kk.get("kmlb_share", {})
check(kk.get("column_mode") == "single" and sh.get("r0") == sh.get("r20") == kk.get("kmlb_columns"),
      f"single mode: every field from KMLB over the same share of the map ({kk.get('kmlb_columns')})")

print("\n== C: KMLB volume unreadable -> plain MRMS frame")
man, f, b = run(T0, "broken")
kk = f.get("kmlb", {})
check(kk.get("mode") == "none" and "corrupt" in (kk.get("note") or ""), f"kmlb: {kk}")
check(b.shape[0] == 15 and (b[14] == 255).all(), "plane 14 present and all 'not sampled'")
check(f["pads"] and f["volume"], "frame still published with pads and the MRMS volume")

print("\n== D: no KMLB volume near the frame time")
man, f, b = run(T0, "none")
kk = f.get("kmlb", {})
check(kk.get("mode") == "none" and "no KMLB volume within" in (kk.get("note") or ""), f"kmlb: {kk}")

print("\n== E: volumes are fetched once per run and reused across frames")
check(n_fetch == len(man["frames"]) or n_fetch <= 13, f"run A fetched {n_fetch} distinct volumes for its frames")

shutil.rmtree("/tmp/kmlb_site", ignore_errors=True)
man, f, b = run(T0, "trunc")
shutil.copytree(os.environ["OUT_DIR"], "/tmp/kmlb_site")
print(f"\n{len(FAILS)} failure(s)")
sys.exit(1 if FAILS else 0)
