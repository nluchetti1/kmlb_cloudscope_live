#!/usr/bin/env python3
"""
Follow-up KMLB probe. Settles the four open questions from probe_kmlb.py:

  G. Top elevation per volume. The newest volume on 24 Oct stopped at 8.0 deg (13 sweeps) - VCP 215
     normally reaches 19.5 deg. Was that volume incomplete, or is KMLB really topping out low? Reads
     the last 8 volumes now and 8 from a convective afternoon (2026-07-15 18-23Z).
  H. IGRA2 XMR soundings: launch hours (the first probe read the wrong columns), soundings per day
     by year since 2014, and the period of record.
  I. When codiapi history starts: LPLWS FM14 and MERLIN CG, month by month from Jan 2025.
  J. When the GOES-16 GLM archive on AWS starts - the decade's only lightning before NLDN (2020-10).

Needs probe_kmlb.py beside it (shared helpers). pip install arm_pyart requests
Writes probe_kmlb2_report.txt and probe_kmlb2.json.
"""
import datetime as dt
import io
import json
import os
import tempfile
import time
import zipfile

import probe_kmlb as P

OUT, J = [], {}
BUCKET = "unidata-nexrad-level2"


def say(s=""):
    print(s, flush=True)
    OUT.append(str(s))


def igra_headers(txt):
    """IGRA2 header records: '#' + 11-char id, then year (cols 14-17), month (19-20), day (22-23),
    hour (25-26) - 1-based columns per the IGRA2 format document. Returns [(datetime, raw)]."""
    out = []
    for ln in txt.splitlines():
        if not ln.startswith("#"):
            continue
        try:
            y, mo, d, h = int(ln[13:17]), int(ln[18:20]), int(ln[21:23]), int(ln[24:26])
        except ValueError:
            continue
        out.append((dt.datetime(y, mo, d, 0 if h == 99 else h), ln))
    return out


def section_g():
    say("\n" + "=" * 78 + "\nG. Top elevation per volume\n" + "=" * 78)
    import pyart
    J["volumes"] = []
    t = P.now()
    sets = [("now", t.strftime("%Y/%m/%d/KMLB/"), None, 8),
            ("2026-07-15 18-23Z", "2026/07/15/KMLB/", (18, 23), 8)]
    for label, pfx, hours, n in sets:
        v = P.volumes(P.s3_list(BUCKET, pfx))
        if hours:
            v = [x for x in v if hours[0] <= x[2].hour <= hours[1]]
            v = v[:: max(1, len(v) // n)][:n]
        else:
            v = v[-n:]
        say(f"   {label}: {len(v)} volumes")
        for k, sz, vt in v:
            try:
                r = P.get(f"https://{BUCKET}.s3.amazonaws.com/{k}", timeout=(8, 120))
                with tempfile.NamedTemporaryFile(suffix=os.path.basename(k)) as fp:
                    fp.write(r.content)
                    fp.flush()
                    radar = pyart.io.read_nexrad_archive(fp.name)
                el = sorted({round(float(a), 1) for a in radar.fixed_angle["data"]})
                t_end = radar.time["data"].max() / 60.0
                row = {"key": k, "vcp": radar.metadata.get("vcp_pattern"), "sweeps": radar.nsweeps,
                       "max_el": max(el), "scan_min": round(float(t_end), 1), "mb": round(sz / 1e6, 1)}
                J["volumes"].append(row)
                say(f"     {vt:%m-%d %H:%M}Z  VCP {row['vcp']}  {row['sweeps']:2d} sweeps  top {row['max_el']:4.1f} deg  "
                    f"scan {row['scan_min']:.1f} min  {row['mb']} MB  tilts {el}")
            except Exception as e:
                say(f"     {vt:%m-%d %H:%M}Z  {type(e).__name__}: {str(e)[:100]}")
    tops = [r["max_el"] for r in J["volumes"]]
    if tops:
        say(f"   => top elevation: min {min(tops)}, max {max(tops)}; "
            f"{sum(1 for x in tops if x < 10)} of {len(tops)} volumes stop below 10 deg")


def section_h():
    say("\n" + "=" * 78 + "\nH. IGRA2 XMR soundings\n" + "=" * 78)
    base = "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/"
    for sub, name in (("data-por/", "USM00074794-data.txt.zip"), ("data-y2d/", "USM00074794-data-beg2026.txt.zip")):
        t0 = time.time()
        r = P.get(base + sub + name, timeout=(8, 300))
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        txt = z.read(z.namelist()[0]).decode("ascii", "replace")
        heads = igra_headers(txt)
        say(f"   {sub}{name}: {len(r.content) / 1e6:.1f} MB zipped, {len(heads):,} soundings, "
            f"{heads[0][0]:%Y-%m-%d} to {heads[-1][0]:%Y-%m-%d %HZ} ({time.time() - t0:.0f} s)")
        if sub == "data-por/":
            by_year = {}
            for t, _ in heads:
                by_year.setdefault(t.year, []).append(t)
            for y in range(2014, P.now().year + 1):
                ts = by_year.get(y, [])
                hrs = {}
                for t in ts:
                    hrs[t.hour] = hrs.get(t.hour, 0) + 1
                top = sorted(hrs.items(), key=lambda x: -x[1])[:4]
                days = len({t.date() for t in ts})
                say(f"     {y}: {len(ts):4d} soundings on {days:3d} days; most common hours (UTC) {top}")
            J["igra_por"] = {"first": str(heads[0][0]), "last": str(heads[-1][0]), "n": len(heads)}
        else:
            lag = (P.now() - heads[-1][0]).total_seconds() / 3600
            say(f"   newest sounding is {lag:.0f} h old")
            J["igra_lag_h"] = round(lag)


def section_i():
    say("\n" + "=" * 78 + "\nI. When codiapi history starts\n" + "=" * 78)
    J["codiapi"] = {}
    m = dt.date(2025, 1, 1)
    while m <= P.now().date():
        a = dt.datetime(m.year, m.month, 15, 12, 0)
        row = {}
        try:
            r = P.get(P.LPLWS + "/range", params={"date>": a.strftime("%Y-%m-%dT%H:%M:%S"),
                      "date<": (a + dt.timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%S"),
                      "field_mill": 14}, timeout=(6, 40))
            row["fm14"] = len(r.json()) if r.status_code == 200 else f"HTTP {r.status_code}"
        except Exception as e:
            row["fm14"] = type(e).__name__
        w0 = dt.datetime(m.year, m.month, 1)
        try:
            r = P.get(P.MERLIN["cg"] + "/range", params={"date>": w0.strftime("%Y-%m-%dT%H:%M:%S"),
                      "date<": (w0 + dt.timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S"), **P.BOX},
                      timeout=(6, 60))
            row["cg_week"] = len(r.json()) if r.status_code == 200 else f"HTTP {r.status_code}: {r.text[:80]}"
        except Exception as e:
            row["cg_week"] = type(e).__name__
        say(f"   {m:%Y-%m}: FM14 rows in 30 min on the 15th: {row['fm14']};  MERLIN CG in days 1-7: {row['cg_week']}")
        J["codiapi"][f"{m:%Y-%m}"] = row
        m = dt.date(m.year + (m.month == 12), m.month % 12 + 1, 1)


def section_j():
    say("\n" + "=" * 78 + "\nJ. GOES-16 GLM archive start (noaa-goes16 GLM-L2-LCFA)\n" + "=" * 78)
    years = [p.strip("/").split("/")[-1] for p, _, _ in P.s3_list("noaa-goes16", "GLM-L2-LCFA/", delimiter="/")]
    say(f"   years: {years}")
    for y in years[:2]:
        days = sorted(p.strip("/").split("/")[-1] for p, _, _ in
                      P.s3_list("noaa-goes16", f"GLM-L2-LCFA/{y}/", delimiter="/"))
        if days:
            first = dt.datetime.strptime(f"{y} {days[0]}", "%Y %j")
            say(f"   {y}: {len(days)} day folders, first {first:%Y-%m-%d} (day {days[0]})")
            J.setdefault("glm", {})[y] = {"days": len(days), "first": str(first.date())}


if __name__ == "__main__":
    say(f"KMLB probe 2 {P.now():%Y-%m-%d %H:%MZ}")
    for name, fn in (("G", section_g), ("H", section_h), ("I", section_i), ("J", section_j)):
        try:
            fn()
        except Exception as e:
            say(f"\n!! section {name} aborted: {type(e).__name__}: {str(e)[:200]}")
    open("probe_kmlb2_report.txt", "w").write("\n".join(OUT) + "\n")
    json.dump(J, open("probe_kmlb2.json", "w"), indent=1, default=str)
    say("\nwrote probe_kmlb2_report.txt and probe_kmlb2.json")
