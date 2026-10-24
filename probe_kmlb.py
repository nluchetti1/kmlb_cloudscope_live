#!/usr/bin/env python3
"""
Probe for a KMLB Level II version of CloudScope Radar (live page + 10-year climatology).
Validates every source before any fetch code is written.

Run in GitHub Actions (probe_kmlb.yml - the real environment, and the only one that can reach the
codiapi APIs) or in Colab (sections A-E; F will time out there as before).

    pip install arm_pyart requests      # Py-ART for section C; everything else is stdlib+requests
    python probe_kmlb.py [--grid]       # --grid also times Py-ART's 3-D gridder (slow)

Writes probe_kmlb_report.txt and probe_kmlb.json.

  A. Which AWS bucket holds Level II now (unidata-nexrad-level2 vs the older noaa-nexrad-level2),
     newest KMLB volume, latency, files per hour.
  B. Real-time chunks bucket: newest chunk age (chunks beat full files by minutes).
  C. Newest volume decoded with Py-ART: size, VCP, elevation angles, fields (dual-pol?), gate
     spacing, read time; optional gridder timing.
  D. Archive depth: one July day per year 2014-2026 - files, MB, dual-pol fields present.
  E. XMR soundings (74794): which archive answers (UWyo, IEM, IGRA2) and how often XMR launches.
  F. codiapi MERLIN / LPLWS archive depth: one hour per year 2014-2025 with a bounding box.
"""
import datetime as dt
import io
import json
import os
import re
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import requests

SITE = "KMLB"
KMLB = (28.1133, -80.6542)
BUCKETS = ["unidata-nexrad-level2", "noaa-nexrad-level2"]
CHUNK_BUCKETS = ["unidata-nexrad-level2-chunks", "noaa-nexrad-level2-chunks"]
BOX = {"lat>": 27.18, "lat<": 30.02, "lon>": -82.08, "lon<": -79.12}
MERLIN = {"cg": "https://codiapi-lightningc2g.rad.spaceforce.mil/lightning_c2g",
          "cc": "https://codiapi-lightningc2c.rad.spaceforce.mil/lightning_c2c"}
LPLWS = "https://codiapi-lplws1.rad.spaceforce.mil/lplws_1min"
OUT, J = [], {}
S = requests.Session()
S.headers["User-Agent"] = "CloudScope-KMLB-probe/1.0"


def say(s=""):
    print(s, flush=True)
    OUT.append(str(s))


def now():
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def get(url, params=None, timeout=(8, 40), stream=False):
    return S.get(url, params=params, timeout=timeout, stream=stream)


# ------------------------------------------------------------------ S3 helpers ----------
def s3_list(bucket, prefix, max_keys=1000, delimiter=None):
    """List keys with the public S3 REST API (no boto). Returns [(key, size, last_modified)]."""
    out, token = [], None
    while True:
        p = {"list-type": "2", "prefix": prefix, "max-keys": str(max_keys)}
        if delimiter:
            p["delimiter"] = delimiter
        if token:
            p["continuation-token"] = token
        r = get(f"https://{bucket}.s3.amazonaws.com/", params=p)
        r.raise_for_status()
        keys, token, prefixes = parse_s3_xml(r.content)
        out += keys if not delimiter else [(pfx, 0, None) for pfx in prefixes] + keys
        if not token:
            return out


def parse_s3_xml(content):
    root = ET.fromstring(content)
    ns = {"s": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
    q = (lambda t: f"s:{t}") if ns else (lambda t: t)
    keys = [(c.findtext(q("Key"), namespaces=ns), int(c.findtext(q("Size"), "0", namespaces=ns)),
             c.findtext(q("LastModified"), namespaces=ns)) for c in root.findall(q("Contents"), ns)]
    prefixes = [c.findtext(q("Prefix"), namespaces=ns) for c in root.findall(q("CommonPrefixes"), ns)]
    trunc = (root.findtext(q("IsTruncated"), "false", namespaces=ns) or "").lower() == "true"
    token = root.findtext(q("NextContinuationToken"), namespaces=ns) if trunc else None
    return keys, token, prefixes


VOL_RE = re.compile(r"KMLB(\d{8})_(\d{6})(_V\d\d)?(\.gz)?$")


def vol_time(key):
    m = VOL_RE.search(key)
    return dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") if m else None


def volumes(keys):
    """Real volume files only (drop *_MDM metadata files and anything unparseable)."""
    return sorted([(k, sz, vol_time(k)) for k, sz, _ in keys if vol_time(k) and not k.endswith("_MDM")],
                  key=lambda x: x[2])


# ------------------------------------------------------------------ A: archive bucket ----
def section_a():
    say("\n" + "=" * 78 + "\nA. Level II archive bucket - newest KMLB volume\n" + "=" * 78)
    t = now()
    for b in BUCKETS:
        found = None
        for day in (t, t - dt.timedelta(days=1)):
            pfx = day.strftime(f"%Y/%m/%d/{SITE}/")
            try:
                v = volumes(s3_list(b, pfx))
            except Exception as e:
                say(f"   {b}: {pfx} -> {type(e).__name__}: {str(e)[:120]}")
                break
            say(f"   {b}: {pfx} -> {len(v)} volumes")
            if v:
                found = v
                break
        if found:
            k, sz, vt = found[-1]
            last_hour = [x for x in found if x[2] > vt - dt.timedelta(hours=1)]
            gaps = [(b2[2] - a2[2]).total_seconds() / 60 for a2, b2 in zip(found, found[1:])]
            say(f"   newest {k} ({sz / 1e6:.1f} MB), volume start {vt:%H:%M:%SZ}, "
                f"{(t - vt).total_seconds() / 60:.1f} min before now")
            say(f"   {len(last_hour)} volumes in the last hour; spacing median "
                f"{sorted(gaps)[len(gaps) // 2]:.1f} min, max {max(gaps):.1f} min" if gaps else "")
            J["archive_bucket"] = b
            J["newest_key"] = k
            J["newest_age_min"] = round((t - vt).total_seconds() / 60, 1)
            return b, k
    say("   no bucket answered with KMLB volumes")
    return None, None


# ------------------------------------------------------------------ B: real-time chunks --
def section_b():
    say("\n" + "=" * 78 + "\nB. Real-time chunks bucket\n" + "=" * 78)
    for b in CHUNK_BUCKETS:
        try:
            vols = s3_list(b, f"{SITE}/", delimiter="/")
            nums = sorted(int(p.strip("/").split("/")[-1]) for p, _, _ in vols
                          if p.strip("/").split("/")[-1].isdigit())
            say(f"   {b}: {len(nums)} volume folders")
            if not nums:
                continue
            best = None
            for n in nums[-6:]:     # folder numbers wrap (1-999); find the newest by chunk time
                ch = s3_list(b, f"{SITE}/{n}/")
                if ch:
                    last = max(ch, key=lambda c: c[2] or "")
                    if best is None or (last[2] or "") > (best[2] or ""):
                        best = last
            if best:
                lm = dt.datetime.strptime(best[2][:19], "%Y-%m-%dT%H:%M:%S")
                say(f"   newest chunk {best[0]} ({best[1] / 1e3:.0f} kB) written "
                    f"{(now() - lm).total_seconds() / 60:.1f} min ago")
                J["chunk_bucket"] = b
                J["chunk_age_min"] = round((now() - lm).total_seconds() / 60, 1)
                return
        except Exception as e:
            say(f"   {b}: {type(e).__name__}: {str(e)[:120]}")


# ------------------------------------------------------------------ C: decode ------------
def section_c(bucket, key, do_grid):
    say("\n" + "=" * 78 + "\nC. Decode the newest volume with Py-ART\n" + "=" * 78)
    if not bucket:
        say("   skipped - no volume found in A")
        return
    t0 = time.time()
    r = get(f"https://{bucket}.s3.amazonaws.com/{key}", timeout=(8, 120))
    r.raise_for_status()
    say(f"   downloaded {len(r.content) / 1e6:.1f} MB in {time.time() - t0:.1f} s")
    try:
        import pyart
    except Exception as e:
        say(f"   Py-ART not importable ({e}); pip install arm_pyart")
        return
    with tempfile.NamedTemporaryFile(suffix=os.path.basename(key)) as fp:
        fp.write(r.content)
        fp.flush()
        t0 = time.time()
        radar = pyart.io.read_nexrad_archive(fp.name)
        rt = time.time() - t0
    el = [round(float(a), 2) for a in radar.fixed_angle["data"]]
    vcp = radar.metadata.get("vcp_pattern")
    gate = float(radar.range["data"][1] - radar.range["data"][0])
    say(f"   read in {rt:.1f} s; VCP {vcp}; {radar.nsweeps} sweeps; {radar.nrays} rays; "
        f"{radar.ngates} gates at {gate:.0f} m to {radar.range['data'][-1] / 1e3:.0f} km")
    say(f"   elevation angles: {el}")
    say(f"   fields: {sorted(radar.fields)}")
    dual = all(f in radar.fields for f in ("differential_reflectivity", "cross_correlation_ratio"))
    say(f"   dual-pol fields present: {dual}")
    J.update({"read_s": round(rt, 1), "vcp": vcp, "elevations": el, "fields": sorted(radar.fields),
              "gate_m": gate, "max_el": max(el)})
    # Beam heights over the Cape (LC-39A ~55 km) and the north edge of the map (~165 km), 4/3 earth
    import math
    ae = 6371e3 * 4 / 3
    for name, km in (("LC-39A", 55), ("map north edge", 165)):
        lo = math.sqrt(km * km * 1e6 + ae * ae + 2 * km * 1e3 * ae * math.sin(math.radians(min(el)))) - ae
        hi = math.sqrt(km * km * 1e6 + ae * ae + 2 * km * 1e3 * ae * math.sin(math.radians(max(el)))) - ae
        say(f"   beam centre at {name} ({km} km): lowest {lo / 1e3:.2f} km, highest {hi / 1e3:.1f} km")
    if do_grid:
        say("   timing pyart.map.grid_from_radars on 200 x 200 x 33 (the MRMS grid) ...")
        t0 = time.time()
        try:
            pyart.map.grid_from_radars(
                radar, grid_shape=(33, 200, 200),
                grid_limits=((500, 19000), (-220e3, 0), (-90e3, 110e3)),
                fields=["reflectivity"], grid_origin=KMLB)
            say(f"   Py-ART gridder: {time.time() - t0:.1f} s per volume")
            J["pyart_grid_s"] = round(time.time() - t0, 1)
        except Exception as e:
            say(f"   gridder failed: {type(e).__name__}: {e}")


# ------------------------------------------------------------------ D: archive depth -----
def section_d(bucket):
    say("\n" + "=" * 78 + "\nD. Archive depth - 15 July of each year\n" + "=" * 78)
    bucket = bucket or BUCKETS[0]
    J["archive"] = {}
    for y in range(2014, now().year + 1):
        day = dt.date(y, 7, 15)
        if day > now().date():
            break
        try:
            v = volumes(s3_list(bucket, day.strftime(f"%Y/%m/%d/{SITE}/")))
        except Exception as e:
            say(f"   {y}: {type(e).__name__}: {str(e)[:100]}")
            continue
        mb = sum(sz for _, sz, _ in v) / 1e6
        hours = len({t.hour for _, _, t in v})
        say(f"   {y}-07-15: {len(v):4d} volumes, {mb:7.0f} MB, {hours:2d}/24 hours with data, "
            f"median {mb / max(len(v), 1):.1f} MB/volume")
        J["archive"][y] = {"volumes": len(v), "mb": round(mb), "hours": hours}
    tot = [d for d in J["archive"].values() if d["volumes"]]
    if tot:
        per_vol = sum(d["mb"] for d in tot) / sum(d["volumes"] for d in tot)
        say(f"   => hourly sampling over 10 years: ~87,600 volumes x {per_vol:.1f} MB = "
            f"{87600 * per_vol / 1e6:.2f} TB to download")


# ------------------------------------------------------------------ E: XMR soundings -----
def section_e():
    say("\n" + "=" * 78 + "\nE. XMR soundings (WMO 74794)\n" + "=" * 78)
    t = now()
    J["soundings"] = {}
    # 1) University of Wyoming - new wsgi interface, then the classic cgi
    for label, url, params in [
        ("UWyo wsgi", "https://weather.uwyo.edu/wsgi/sounding",
         {"datetime": (t - dt.timedelta(days=1)).strftime("%Y-%m-%d 12:00:00"), "id": "74794",
          "type": "TEXT:LIST", "src": "UNKNOWN"}),
        ("UWyo cgi", "https://weather.uwyo.edu/cgi-bin/sounding",
         {"region": "naconf", "TYPE": "TEXT:LIST", "YEAR": t.strftime("%Y"), "MONTH": t.strftime("%m"),
          "FROM": "0100", "TO": t.strftime("%d") + "23", "STNM": "74794"})]:
        try:
            r = get(url, params=params)
            n = r.text.count("Station number: 74794") or r.text.count("74794 XMR")
            say(f"   {label}: HTTP {r.status_code}, {len(r.text):,} chars, soundings found: {n}")
            J["soundings"][label] = {"status": r.status_code, "found": n}
        except Exception as e:
            say(f"   {label}: {type(e).__name__}: {str(e)[:120]}")
    # 2) Iowa Environmental Mesonet RAOB JSON - one call per synoptic time over the last 3 days
    hits = []
    for h in range(0, 72, 6):
        ts = (t - dt.timedelta(hours=h)).replace(minute=0, second=0)
        ts = ts.replace(hour=ts.hour - ts.hour % 6)
        for stn in ("KXMR", "XMR"):
            try:
                r = get("https://mesonet.agron.iastate.edu/json/raob.py",
                        params={"ts": ts.strftime("%Y%m%d%H%M"), "station": stn})
                prof = r.json().get("profiles", [])
                if prof:
                    hits.append((ts.strftime("%m-%d %HZ"), stn, len(prof[0].get("profile", []))))
                    break
            except Exception:
                pass
    say(f"   IEM RAOB JSON, synoptic times in the last 72 h with an XMR profile: "
        f"{len(hits)} of 12 -> {sorted(set(hits))}")
    J["soundings"]["IEM"] = hits
    # 3) IGRA2 - the long archive; list the y2d / por directories for the station file
    for sub in ("data-y2d", "data-por"):
        url = f"https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/{sub}/"
        try:
            r = get(url)
            m = re.findall(r'href="(USM000747(?:94)[^"]*)"', r.text)
            say(f"   IGRA2 {sub}: HTTP {r.status_code}; XMR files: {m or 'none listed'}")
            J["soundings"][f"IGRA2 {sub}"] = m
            if sub == "data-y2d" and m:
                z = get(url + m[0], timeout=(8, 120))
                import zipfile
                txt = zipfile.ZipFile(io.BytesIO(z.content)).read(zipfile.ZipFile(io.BytesIO(z.content)).namelist()[0]).decode()
                heads = [ln for ln in txt.splitlines() if ln.startswith("#USM00074794")]
                hrs = {}
                for ln in heads:
                    hrs[ln[22:24]] = hrs.get(ln[22:24], 0) + 1
                say(f"   IGRA2 year-to-date XMR soundings: {len(heads)}; by launch hour (UTC): "
                    f"{dict(sorted(hrs.items()))}; newest header {heads[-1][:40] if heads else '-'}")
                J["soundings"]["IGRA2 y2d count"] = len(heads)
        except Exception as e:
            say(f"   IGRA2 {sub}: {type(e).__name__}: {str(e)[:120]}")


# ------------------------------------------------------------------ F: codiapi depth -----
def section_f():
    say("\n" + "=" * 78 + "\nF. MERLIN and LPLWS archive depth (codiapi)\n" + "=" * 78)
    J["codiapi"] = {}
    for y in range(2014, now().year):
        a = dt.datetime(y, 7, 15, 20, 0)
        win = {"date>": a.strftime("%Y-%m-%dT%H:%M:%S"),
               "date<": (a + dt.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")}
        row = {}
        for kind, base in MERLIN.items():
            try:
                r = get(base + "/range", params={**win, **BOX}, timeout=(6, 40))
                n = len(r.json()) if r.status_code == 200 else f"HTTP {r.status_code}"
            except Exception as e:
                n = type(e).__name__
            row[kind] = n
        try:
            r = get(LPLWS + "/range", params={**win, "field_mill": 14}, timeout=(6, 40))
            row["fm14_minutes"] = len(r.json()) if r.status_code == 200 else f"HTTP {r.status_code}"
        except Exception as e:
            row["fm14_minutes"] = type(e).__name__
        say(f"   {y}-07-15 20-21Z: MERLIN CG {row['cg']}, cloud-to-cloud {row['cc']}, "
            f"FM14 minutes {row['fm14_minutes']}")
        J["codiapi"][y] = row


if __name__ == "__main__":
    say(f"KMLB probe {now():%Y-%m-%d %H:%MZ}, python {sys.version.split()[0]}")
    b = k = None
    for name, fn in (("A", section_a), ("B", section_b), ("C", None), ("D", None), ("E", section_e),
                     ("F", section_f)):
        try:
            if name == "A":
                b, k = fn()
            elif name == "C":
                section_c(b, k, "--grid" in sys.argv)
            elif name == "D":
                section_d(b)
            else:
                fn()
        except Exception as e:
            say(f"\n!! section {name} aborted: {type(e).__name__}: {str(e)[:200]}")
    open("probe_kmlb_report.txt", "w").write("\n".join(OUT) + "\n")
    json.dump(J, open("probe_kmlb.json", "w"), indent=1, default=str)
    say("\nwrote probe_kmlb_report.txt and probe_kmlb.json")
