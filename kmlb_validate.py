#!/usr/bin/env python3
"""One-off check of the real KMLB path before the page goes live: download a real volume, read it
with Py-ART, sample it onto the map and report what the hybrid would use. Default: the Crew-11
scrub volume (31 Jul 2025 16:05Z, AVSET off). Try 2026-10-24T15:22 for the speckle case.
    python kmlb_validate.py [YYYY-MM-DDTHH:MM]"""
import datetime as dt
import sys
import time
import numpy as np
import requests
import kmlb_classify as K

at = dt.datetime.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else dt.datetime(2025, 7, 31, 16, 5)
s = requests.Session()
vols = K.list_day(s, at.date())
t, key = min(vols, key=lambda v: abs((v[0] - at).total_seconds()))
print(f"{len(vols)} volumes that day; using {key}")
t0 = time.time(); raw = s.get(f"{K.BUCKET}/{key}", timeout=60).content; t1 = time.time()
vol = K.read_volume(raw, key); t2 = time.time()
la = np.round(29.595 - 0.01 * np.arange(200), 3); lo = np.round(-81.595 + 0.01 * np.arange(200), 3)
fr = K.KmlbFrame(vol, la, lo, None, np.full((200, 200), 4800.0), (vol["mid"]).strftime("%Y-%m-%dT%H:%M:%SZ")); t3 = time.time()
print(f"download {t1-t0:.1f} s, read {t2-t1:.1f} s, sample + 33 levels {t3-t2:.1f} s")
print(f"VCP {vol['vcp']}, {vol['nsweeps']} sweeps -> {len(vol['sweeps'])} elevations {[sw['el'] for sw in vol['sweeps']]}")
print(f"radar at {vol['lat']:.4f}, {vol['lon']:.4f}, {vol['alt']:.0f} m; midpoint {vol['mid']:%H:%M:%SZ}")
print(f"frame info: {fr.info}")
LOg, LAg = np.meshgrid(lo, la)
near = np.hypot((LAg - K.LC39A[0]) * 60, (LOg - K.LC39A[1]) * 60 * np.cos(np.radians(28.6))) <= 10
for name, h in (("0 C (4.8 km)", 4800.0), ("-10 C (6.3 km)", 6338.0), ("-20 C (7.9 km)", 7877.0), ("10 km", 10000.0)):
    v = K.at_heights(fr.Z, fr.H, fr.W, h)
    vn = v[near]
    print(f"  {name:15} within 10 nmi of LC-39A: sampled {np.isfinite(vn).mean():5.0%}, "
          + (f"echo {np.nanmax(np.where(vn > -99, vn, np.nan)):.1f} dBZ max" if (vn > -99).any() else "no echo"))
# QC: what each screen removed, and low-level echo near the pads with the screens off vs on
print(f"QC removed (gates, all sweeps): {fr.qc}")
saved = (K.QC_RHO, K.QC_NORHO_DBZ, K.SPECKLE_GATES)
K.QC_RHO, K.QC_NORHO_DBZ, K.SPECKLE_GATES = -1.0, -999.0, 0
Zr, Hr, Wr, _ = K.sample_sweeps(vol, la, lo)
K.QC_RHO, K.QC_NORHO_DBZ, K.SPECKLE_GATES = saved
for name, (Z_, H_, W_) in (("screens off", (Zr, Hr, Wr)), ("screens on ", (fr.Z, fr.H, fr.W))):
    lo1 = K.at_heights(Z_, H_, W_, 1000.0)
    print(f"  {name}: echo cells within 10 nmi of LC-39A at 1 km: {int(((lo1 > -99) & near).sum())} "
          f"of {int((np.isfinite(lo1) & near).sum())} sampled")
