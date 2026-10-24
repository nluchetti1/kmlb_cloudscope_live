#!/usr/bin/env python3
"""
Probe 3: do AVSET-off (full) volumes show up in the KMLB archive during launch windows?

    python probe_kmlb3.py 2026-10-20T14:00/2026-10-20T18:00 2026-10-22T02:00/2026-10-22T05:00
    python probe_kmlb3.py --days 14      # no windows known: one volume per hour, last 14 days

Lists every volume in each window (or the hourly sample) with VCP, sweeps and top tilt, and flags
full volumes (top >= 14 deg). Needs probe_kmlb.py beside it. pip install arm_pyart requests
Writes probe_kmlb3_report.txt.
"""
import datetime as dt
import os
import sys
import tempfile

import probe_kmlb as P

BUCKET = "unidata-nexrad-level2"
FULL_DEG = 14.0
OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(str(s))


def windows_from_args(argv):
    """['A/B', ...] -> [(datetime, datetime)]; '--days N' -> hourly sample over the last N days."""
    if "--days" in argv:
        n = int(argv[argv.index("--days") + 1])
        end = P.now().replace(minute=0, second=0, microsecond=0)
        return [(end - dt.timedelta(hours=h + 1), end - dt.timedelta(hours=h)) for h in range(24 * n)][::-1], True
    out = []
    for a in argv:
        if "/" in a:
            s, e = a.split("/")
            out.append((dt.datetime.fromisoformat(s), dt.datetime.fromisoformat(e)))
    return out, False


def volumes_between(t0, t1):
    out, day = [], t0.date()
    while day <= t1.date():
        out += [v for v in P.volumes(P.s3_list(BUCKET, day.strftime("%Y/%m/%d/KMLB/"))) if t0 <= v[2] < t1]
        day += dt.timedelta(days=1)
    return out


def top_of(key):
    import pyart
    r = P.get(f"https://{BUCKET}.s3.amazonaws.com/{key}", timeout=(8, 120))
    with tempfile.NamedTemporaryFile(suffix=os.path.basename(key)) as fp:
        fp.write(r.content)
        fp.flush()
        radar = pyart.io.read_nexrad_archive(fp.name)
    el = sorted({round(float(a), 1) for a in radar.fixed_angle["data"]})
    return radar.metadata.get("vcp_pattern"), radar.nsweeps, max(el)


if __name__ == "__main__":
    wins, hourly = windows_from_args(sys.argv[1:])
    if not wins:
        sys.exit("give windows like 2026-10-20T14:00/2026-10-20T18:00, or --days N")
    say(f"KMLB probe 3 {P.now():%Y-%m-%d %H:%MZ}: {'hourly sample' if hourly else f'{len(wins)} window(s)'}")
    listing = {}
    n_full = n_all = 0
    for t0, t1 in wins:
        if hourly:
            day = t0.date()
            if day not in listing:
                listing[day] = P.volumes(P.s3_list(BUCKET, day.strftime("%Y/%m/%d/KMLB/")))
            vols = [v for v in listing[day] if t0 <= v[2] < t1][:1]
        else:
            say(f"\n== {t0:%Y-%m-%d %H:%M}Z to {t1:%H:%M}Z")
            vols = volumes_between(t0, t1)
        for k, sz, vt in vols:
            try:
                vcp, ns, top = top_of(k)
            except Exception as e:
                say(f"   {vt:%m-%d %H:%M}Z  {type(e).__name__}: {str(e)[:80]}")
                continue
            n_all += 1
            full = top >= FULL_DEG
            n_full += full
            if not hourly or full:
                say(f"   {vt:%m-%d %H:%M}Z  VCP {vcp}  {ns:2d} sweeps  top {top:4.1f} deg"
                    f"{'   <-- FULL VOLUME (AVSET off?)' if full else ''}")
    say(f"\n=> {n_full} of {n_all} volumes reached {FULL_DEG:.0f} deg or higher")
    open("probe_kmlb3_report.txt", "w").write("\n".join(OUT) + "\n")
