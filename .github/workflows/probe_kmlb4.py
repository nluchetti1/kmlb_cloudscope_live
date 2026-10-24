#!/usr/bin/env python3
"""
Probe 4: the unvalidated pieces of the hybrid KMLB page, on real volumes from 20 Oct 2026
(10:02-11:15Z full VCP 215 to 19.5 deg - AVSET off; VCP 35 to 6.4 deg afterwards).

  K. Bucket latency: S3 LastModified minus volume start, for every volume that day.
  L. Full 18-sweep volume vs a 12-sweep one: read time, and Py-ART grid time for reflectivity
     alone and for Z + ZDR + RHOHV + KDP on the 200 x 200 x 33 grid.
  M. KDP from PhiDP: Py-ART kdp_vulpiani and CSU calc_kdp_bringi - time and sanity.
  N. CSU hydrometeor ID (summer, S band) with a standard temperature profile: time, class counts,
     graupel gates. Plus a ZDR-column count (ZDR >= 1 dB and Z >= 20 dBZ above the 0 C level).

    python probe_kmlb4.py                          # the 20 Oct 2026 volumes (default)
    python probe_kmlb4.py --at 2025-07-31T16:05     # the volume nearest a time (e.g. Crew-11 scrub)

Needs probe_kmlb.py beside it. pip install arm_pyart csu_radartools requests
Writes probe_kmlb4_report.txt.
"""
import datetime as dt
import os
import sys
import tempfile
import time

import numpy as np

import probe_kmlb as P

BUCKET = "unidata-nexrad-level2"
DAY = "2026/10/20/KMLB/"
OUT = []
FREEZE_KM = 4.6          # standard Florida autumn 0 C height for the test only
LC39A = (28.6084, -80.6043)


def say(s=""):
    print(s, flush=True)
    OUT.append(str(s))


def read(key):
    import pyart
    r = P.get(f"https://{BUCKET}.s3.amazonaws.com/{key}", timeout=(8, 120))
    with tempfile.NamedTemporaryFile(suffix=os.path.basename(key)) as fp:
        fp.write(r.content)
        fp.flush()
        t0 = time.time()
        radar = pyart.io.read_nexrad_archive(fp.name)
        return radar, time.time() - t0


def gate_temps(radar):
    """Standard lapse-rate temperature at every gate (deg C) from beam height - test profile only."""
    z = radar.gate_z["data"] / 1000.0 + 0.011          # km above MSL (KMLB ~11 m)
    return 25.0 - 6.5 * z, z


def section_k(keys):
    say("\n" + "=" * 78 + "\nK. Bucket latency (LastModified - volume start)\n" + "=" * 78)
    lags = []
    for k, sz, lm in keys:
        vt = P.vol_time(k)
        if vt and lm and not k.endswith("_MDM"):
            lags.append((lm_dt := dt.datetime.strptime(lm[:19], "%Y-%m-%dT%H:%M:%S"), (lm_dt - vt).total_seconds() / 60, k))
    if lags:
        v = sorted(x[1] for x in lags)
        say(f"   {len(v)} volumes: lag median {v[len(v) // 2]:.1f} min, p90 {v[int(0.9 * len(v))]:.1f}, max {v[-1]:.1f}")
        full = [x[1] for x in lags if P.vol_time(x[2]).hour in (10, 11)] if "--at" not in sys.argv else []
        if full:
            say(f"   10-11Z (full volumes): median {sorted(full)[len(full) // 2]:.1f} min")


def section_l(full_key, short_key):
    say("\n" + "=" * 78 + "\nL. Read and grid: full vs truncated volume\n" + "=" * 78)
    import pyart
    out = {}
    for label, key in (("full 19.5 deg", full_key), ("VCP 35 6.4 deg", short_key)):  # labels are keys; --at uses the same two slots
        radar, rt = read(key)
        el = sorted({round(float(a), 1) for a in radar.fixed_angle["data"]})
        say(f"   {label}: {os.path.basename(key)}, {radar.nsweeps} sweeps, top {max(el)} deg, read {rt:.1f} s")
        kw = dict(grid_shape=(33, 200, 200), grid_limits=((500, 19000), (-60e3, 165e3), (-90e3, 110e3)),
                  grid_origin=P.KMLB)
        t0 = time.time()
        pyart.map.grid_from_radars(radar, fields=["reflectivity"], **kw)
        g1 = time.time() - t0
        flds = [f for f in ("reflectivity", "differential_reflectivity", "cross_correlation_ratio") if f in radar.fields]
        t0 = time.time()
        pyart.map.grid_from_radars(radar, fields=flds, **kw)
        g3 = time.time() - t0
        say(f"     grid reflectivity only {g1:.1f} s; {len(flds)} fields {g3:.1f} s")
        out[label] = radar
    return out


def section_m(radar):
    say("\n" + "=" * 78 + "\nM. KDP from PhiDP (full volume)\n" + "=" * 78)
    import pyart
    phi = radar.fields["differential_phase"]["data"]
    say(f"   PhiDP range {float(np.ma.min(phi)):.0f} to {float(np.ma.max(phi)):.0f} deg; "
        f"{radar.nrays} rays x {radar.ngates} gates")
    t0 = time.time()
    try:
        kdp, _ = pyart.retrieve.kdp_vulpiani(radar, psidp_field="differential_phase", windsize=10, band="S")
        k = kdp["data"]
        say(f"   Py-ART kdp_vulpiani: {time.time() - t0:.1f} s; KDP p99 {float(np.nanpercentile(np.ma.filled(k, np.nan), 99)):.2f} deg/km")
        radar.add_field("kdp", kdp, replace_existing=True)
    except Exception as e:
        say(f"   kdp_vulpiani failed after {time.time() - t0:.1f} s: {type(e).__name__}: {str(e)[:120]}")
    try:
        from csu_radartools import csu_kdp
        dz = np.ma.filled(radar.fields["reflectivity"]["data"], -32768.0)
        dp = np.ma.filled(phi, -32768.0)
        rng = np.broadcast_to(radar.range["data"] / 1000.0, dz.shape)
        n = min(360, radar.nrays)                       # time a sweep's worth, extrapolate
        t0 = time.time()
        for i in range(n):
            csu_kdp.calc_kdp_bringi(dp=dp[i], dz=dz[i], rng=rng[i], thsd=12, gs=250.0, window=5)
        per = (time.time() - t0) / n
        say(f"   CSU calc_kdp_bringi: {per * 1000:.1f} ms/ray -> {per * radar.nrays:.0f} s for the volume")
    except Exception as e:
        say(f"   CSU KDP: {type(e).__name__}: {str(e)[:120]}")


def section_n(radar):
    say("\n" + "=" * 78 + "\nN. Hydrometeor ID and ZDR column (full volume)\n" + "=" * 78)
    T, zkm = gate_temps(radar)
    dz = np.ma.filled(radar.fields["reflectivity"]["data"], np.nan)
    zdr = np.ma.filled(radar.fields["differential_reflectivity"]["data"], np.nan)
    rho = np.ma.filled(radar.fields["cross_correlation_ratio"]["data"], np.nan)
    kdp = (np.ma.filled(radar.fields["kdp"]["data"], np.nan) if "kdp" in radar.fields
           else np.zeros_like(dz))
    col = (zdr >= 1.0) & (dz >= 20.0) & (zkm > FREEZE_KM) & (rho >= 0.8)
    if "--at" in sys.argv:
        say(f"   (0 C height assumed {FREEZE_KM} km; summer is nearer 4.8-5.0 km, so this slightly overcounts)")
    say(f"   ZDR-column gates (ZDR >= 1 dB, Z >= 20 dBZ, above {FREEZE_KM} km, RHOHV >= 0.8): {int(col.sum())}")
    try:
        from csu_radartools import csu_fhc
        t0 = time.time()
        hid = csu_fhc.csu_fhc_summer(dz=dz, zdr=zdr, rho=rho, kdp=kdp, use_temp=True, band="S", T=T)
        ht = time.time() - t0
        names = ["unclassified", "drizzle", "rain", "ice crystals", "aggregates", "wet snow",
                 "vertical ice", "low-density graupel", "high-density graupel", "hail", "big drops"]
        valid = np.isfinite(dz) & (dz > 0)
        wx = valid & (dz >= 5) & (rho >= 0.8)       # weather-only: drops insects, clutter, clear air
        say(f"   csu_fhc_summer: {ht:.1f} s for the volume")
        for label, m in (("Z > 0 dBZ (all echo)", valid), ("weather only (Z >= 5, RHOHV >= 0.8)", wx)):
            counts = {names[c]: int(((hid == c) & m).sum()) for c in range(11) if ((hid == c) & m).any()}
            say(f"   class counts, {label}: {counts}")
        g = ((hid == 7) | (hid == 8)) & wx & (T <= 0) & (T >= -20)
        say(f"   graupel gates in the 0 to -20 C layer (weather only): {int(g.sum())}")
        lat, lon = radar.gate_latitude["data"], radar.gate_longitude["data"]
        near = np.hypot((lat - LC39A[0]) * 60, (lon - LC39A[1]) * 60 * np.cos(np.radians(LC39A[0]))) <= 10
        gn = g & near
        if gn.any():
            say(f"   graupel within 10 nmi of LC-39A: {int(gn.sum())} gates, heights "
                f"{float(zkm[gn].min()):.1f}-{float(zkm[gn].max()):.1f} km, max Z {float(dz[gn].max()):.0f} dBZ")
        else:
            say("   graupel within 10 nmi of LC-39A: none")
        cn = col & near
        say(f"   ZDR-column gates within 10 nmi of LC-39A: {int(cn.sum())}")
    except Exception as e:
        say(f"   CSU HID: {type(e).__name__}: {str(e)[:160]}")


def pick(vols, at):
    """Volume starting nearest `at`."""
    return min(vols, key=lambda v: abs((v[2] - at).total_seconds()))[0] if vols else None


if __name__ == "__main__":
    say(f"KMLB probe 4 {P.now():%Y-%m-%d %H:%MZ}")
    at = dt.datetime.fromisoformat(sys.argv[sys.argv.index("--at") + 1]) if "--at" in sys.argv else None
    if at:
        DAY = at.strftime("%Y/%m/%d/KMLB/")
    try:
        keys = P.s3_list(BUCKET, DAY)
    except Exception as e:
        say(f"!! cannot list {BUCKET}/{DAY}: {type(e).__name__}: {str(e)[:160]}")
        open("probe_kmlb4_report.txt", "w").write("\n".join(OUT) + "\n")
        raise SystemExit(1)
    vols = P.volumes(keys)
    if at:
        full = pick(vols, at)
        short = pick(vols, at - dt.timedelta(minutes=30))     # a second volume for comparison
    else:
        full = next((k for k, _, t in vols if t.hour == 10 and t.minute >= 2), None)
        short = next((k for k, _, t in vols if t.hour == 12), None)
    say(f"   full volume: {full}\n   truncated volume: {short}")
    radars = {}
    for name, fn in (("K", lambda: section_k(keys)), ("L", lambda: radars.update(section_l(full, short))),
                     ("M", lambda: section_m(radars["full 19.5 deg"])),
                     ("N", lambda: section_n(radars["full 19.5 deg"]))):
        try:
            fn()
        except Exception as e:
            say(f"\n!! section {name} aborted: {type(e).__name__}: {str(e)[:200]}")
    open("probe_kmlb4_report.txt", "w").write("\n".join(OUT) + "\n")
    say("\nwrote probe_kmlb4_report.txt")
