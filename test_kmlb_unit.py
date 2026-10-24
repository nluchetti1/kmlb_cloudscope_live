"""Unit tests for kmlb_classify's beam sampling, against an analytic 3-D storm."""
import sys
import numpy as np
sys.path.insert(0, ".")
import types
for m in ["pygrib"]:
    sys.modules.setdefault(m, types.ModuleType(m))
import kmlb_classify as K

FAILS = []
def check(c, msg):
    print(("  ok    " if c else "  FAIL  ") + msg)
    if not c: FAILS.append(msg)

from kmlb_synth import LA, LO, STORM, CLUTTER, synth_volume, FULL, TRUNC
j = int(np.argmin(np.abs(LA - STORM[0]))); i = int(np.argmin(np.abs(LO - STORM[1])))
jc = int(np.argmin(np.abs(LA - CLUTTER[0]))); ic = int(np.argmin(np.abs(LO - CLUTTER[1])))

print("== geometry")
s, az = K.column_geometry(np.array([K.LC39A[0]]), np.array([K.LC39A[1]]), 28.1133, -80.6542)
check(abs(s[0, 0] / 1000 - 55.4) < 1.0 and abs(az[0, 0] - 7.0) < 2.0, f"LC-39A: {s[0,0]/1000:.1f} km at {az[0,0]:.1f} deg from KMLB (expect ~55 km, ~7 deg)")
h, r, w = K.beam(6.4, 55000.0, 30.0)
check(abs(h / 1000 - 6.36) < 0.1, f"6.4 deg beam centre at 55 km: {h/1000:.2f} km (probe said ~6.3)")
h, r, w = K.beam(19.5, 55000.0, 30.0)
check(abs(h / 1000 - 19.7) < 0.3, f"19.5 deg beam centre at 55 km: {h/1000:.1f} km")

print("== full volume")
v = synth_volume(FULL)
Z, H, W = K.sample_sweeps(v, LA, LO)
for hgt in (3000.0, 4800.0, 7900.0, 10000.0):
    got = K.at_heights(Z, H, W, hgt)[j, i]
    exp = 50 - 2.5 * 0 - 2.0 * hgt / 1000
    r_ = np.hypot((LA[j]-STORM[0])*111.2, (LO[i]-STORM[1])*97.7); exp = 50 - 2.5*r_ - 2*hgt/1000
    check(abs(got - exp) < 2.5, f"storm core at {hgt/1000:.1f} km: {got:.1f} dBZ vs truth {exp:.1f}")
top = K.top_edge(Z, H, W)
check(top[j, i] / 1000 > 18.0, f"full volume samples the storm column to {top[j,i]/1000:.1f} km")
ka = len(FULL) - 1
LOg, LAg = np.meshgrid(LO, LA)
rs = np.hypot((LAg - STORM[0]) * 111.2, (LOg - STORM[1]) * 97.7)
inside = (rs > 16) & (rs < 29) & (H[ka] > 10200) & (H[ka] < 11300)   # beam in the sheet, off the core
ja, ia = [int(x[0]) for x in np.nonzero(inside)]
check(inside.sum() > 0 and abs(Z[ka, ja, ia] - 2.0) < 0.01,
      f"weak anvil (2 dBZ, RHOHV 0.6) on the 19.5 deg beam at {H[ka, ja, ia]/1000:.1f} km kept "
      f"({int(inside.sum())} such columns) - QC only acts below 3 km")
check(abs(K.at_heights(Z, H, W, float(H[ka, ja, ia]))[ja, ia] - 2.0) < 0.01, "and reconstructed at that height")
check(K.at_heights(Z, H, W, 600.0)[jc, ic] == K.NO_ECHO, f"low clutter (15 dBZ, RHOHV 0.5) removed: {K.at_heights(Z, H, W, 600.0)[jc, ic]}")
check(np.isnan(K.at_heights(Z, H, W, 16000.0)[jc, ic]), "cone of silence: 16 km over the radar not sampled (MRMS fills)")

print("== truncated volume (AVSET at 6.4 deg)")
vt = synth_volume(TRUNC)
Zt, Ht, Wt = K.sample_sweeps(vt, LA, LO)
tt = K.top_edge(Zt, Ht, Wt)
check(6.5 < tt[j, i] / 1000 < 7.2, f"storm column (~54 km) sampled only to {tt[j,i]/1000:.2f} km")
check(np.isnan(K.at_heights(Zt, Ht, Wt, 7900.0)[j, i]), "-20 C height (7.9 km) at the storm: not sampled -> MRMS")
check(np.isfinite(K.at_heights(Zt, Ht, Wt, 4800.0)[j, i]), "0 C height (4.8 km) at the storm: KMLB")

print("== VII")
lev = np.array(K.STD_LEVELS_KM) * 1000
Kc = np.full((len(lev), 1, 1), 40.0, np.float32)
vii = K.vii_from(lev, Kc, np.full((1, 1), 6300.0), np.full((1, 1), 10100.0))[0, 0]
check(3.0 < vii < 6.5, f"40 dBZ through a 3.8 km ice layer: VII {vii:.2f} kg/m2 (hand calc ~4.4)")
Kw = np.full((len(lev), 1, 1), 10.0, np.float32)
viw = K.vii_from(lev, Kw, np.full((1, 1), 6300.0), np.full((1, 1), 10100.0))[0, 0]
check(viw < K.VII_FLOOR, f"10 dBZ ice layer: VII {viw:.3f} kg/m2, below the {K.VII_FLOOR} floor - not a core")

print("== timing")
import time
t0 = time.time(); fr = K.KmlbFrame(v, LA, LO, None, np.full((200, 200), 4800.0), "2025-07-31T16:08:00Z"); dt_ = time.time() - t0
check(dt_ < 10, f"sample + 33 levels for a 15-sweep volume: {dt_:.2f} s")
check(fr.info["mode"] == "full" and fr.info["lc39a_top_km"] > 18, f"frame info: {fr.info['mode']}, LC-39A to {fr.info['lc39a_top_km']} km, offset {fr.info['offset_min']} min")
frt = K.KmlbFrame(vt, LA, LO, None, np.full((200, 200), 4800.0), "2025-07-31T16:08:00Z")
check(frt.info["mode"] == "partial" and 6.5 < frt.info["lc39a_top_km"] < 7.2, f"truncated frame: {frt.info['mode']}, LC-39A to {frt.info['lc39a_top_km']} km")

print("== field merge: what comes from where")
def mrms_F():
    shp = (200, 200)
    LOg, LAg = np.meshgrid(LO, LA)
    r = np.hypot((LAg - STORM[0]) * 111.2, (LOg - STORM[1]) * 97.7)
    st = r < 15
    f = lambda v, none: np.where(st, v, none).astype(np.float32)
    return {"comp": f(45.0, -99.0), "r0": f(30.0, -99.0), "r5": f(28.0, -99.0), "r10": f(25.0, -99.0),
            "r15": f(22.0, -99.0), "r20": f(20.0, -99.0), "high": f(18.0, -99.0), "super": f(15.0, -99.0),
            "hmax": f(3000.0, -1.0), "et18": f(11.0, -1.0), "vii": f(2.0, -1.0), "cg": np.full(shp, -1.0, np.float32)}
z0 = np.full((200, 200), 4800.0)
h20 = 4800 + 20 / 6.5 * 1000
r0_ = np.hypot((LA[j]-STORM[0])*111.2, (LO[i]-STORM[1])*97.7)
F = mrms_F(); fr.apply_fields(F)
check(abs(F["r20"][j, i] - (50 - 2.5*r0_ - 2*h20/1000)) < 2.5, f"full: r20 from KMLB {F['r20'][j,i]:.1f} (truth {50-2.5*r0_-2*h20/1000:.1f}; MRMS said 20)")
check(abs(F["comp"][j, i] - 49.0) < 2.5, f"full: comp from KMLB alone {F['comp'][j,i]:.1f} (MRMS said 45)")
check(F["vii"][j, i] >= K.VII_FLOOR and F["vii"][j, i] != 2.0, f"full: VII from KMLB {F['vii'][j,i]:.2f} kg/m2")
check(F["super"][j, i] > K.NO_ECHO, f"full: 33-60 kft layer from KMLB {F['super'][j,i]:.1f} dBZ")
check(F["hmax"][j, i] <= 1000.0, f"full: height of max {F['hmax'][j,i]:.0f} m (lowest level - the analytic storm peaks at the ground)")
F = mrms_F(); frt.apply_fields(F)
check(F["r20"][j, i] == 20.0, f"truncated: r20 from MRMS {F['r20'][j,i]:.1f}")
check(abs(F["r0"][j, i] - (50 - 2.5*r0_ - 2*4.8)) < 2.5, f"truncated: r0 from KMLB {F['r0'][j,i]:.1f}")
check(abs(F["comp"][j, i] - max(49.0, 45.0)) < 2.5, f"truncated: comp is the larger of KMLB and MRMS {F['comp'][j,i]:.1f}")
check(F["vii"][j, i] == 2.0, "truncated: VII from MRMS (KMLB cannot see -10 to -40 C here)")
check(F["super"][j, i] == 15.0, "truncated: 33-60 kft layer from MRMS")
check(F["r0"][jc, ic] == -99.0 or not np.isfinite(F["r0"][jc, ic]) or F["r0"][jc, ic] <= -99, "clutter column: no echo at 0 C")
print(f"\n{len(FAILS)} failure(s)"); sys.exit(1 if FAILS else 0)
