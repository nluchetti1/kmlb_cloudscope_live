"""
CloudScope Radar - KMLB hybrid
==============================

The MRMS classifier (mrms_classify.py, pinned and unchanged in its rules) with KMLB Level II merged
in: wherever KMLB's beams sample a height over a map column, that height comes from KMLB; above
KMLB's highest beam (and in the cone of silence over the radar) it comes from MRMS. Every frame
uses the volume whose scan midpoint is nearest the frame time, and records which it was.

WHY A HYBRID
    Measured Oct 2026 (probes 1-4): KMLB runs AVSET most of the time, and AVSET ends a volume early
    when the upper tilts lack significant (>= ~18 dBZ) echo - 15 of 16 sampled volumes stopped at
    6.4 deg, a beam top of ~6.8 km over LC-39A: about -10 C in summer, below -20 C and far below
    anvil altitude. During launch countdowns the 45 WS asks MLB to turn AVSET off (31 Jul 2025, the
    Crew-11 scrub: full volumes to 19.5 deg from 13:01 to 16:09Z). So the depth KMLB covers changes
    volume to volume, and the page says, for every frame, how far up KMLB reached.

SINGLE-SOURCE COLUMNS (COLUMN_MODE = "single", since kmlb-v2)
    A map column is ALL KMLB where KMLB samples it from 3 km or lower up to FULL_KM (15 km) or
    higher, and ALL MRMS everywhere else - slices, composite, hmax, the upper layer, VII, the Steiner
    level and the 3-D volume alike. The per-field merge below ("mixed") let one column mix radars:
    KMLB's empty slices with MRMS's composite and hmax made the classifier's gap rule see "echo, no
    slice hits, peak above -10 C" and call cumulus colder than -10 C that neither radar showed - 6
    of the 10 extra violations a case probe could reproduce (Oct 2026), concentrated 50-65 km from
    KMLB where the 6.4 deg beam reaches -10 C but not -20 C: right over the pads. With AVSET on, pad
    columns are therefore MRMS; with AVSET off (launch countdowns) they are KMLB.

WHAT COMES FROM WHERE (COLUMN_MODE = "mixed", the kmlb-v1 behaviour)
    r0 ... r20      KMLB at the XMR (BUFKIT) isotherm heights where its beams reach them, else MRMS
    comp, hmax      KMLB where it samples the column to 15 km; elsewhere the larger of KMLB and MRMS
    super           MRMS 33-60 kft layer composite (definition believed, not verified) unless KMLB
                    samples the whole layer; partial coverage takes the larger
    vii             KMLB (Mosier et al. 2011, -10 to -40 C) where it samples that layer, floored at
                    VII_FLOOR so weak ice does not read as a core; else MRMS. The floor approximates
                    the MRMS product and needs tuning against overlapping cases.
    cg, lightning, field mills, satellite    unchanged (MRMS NLDN, MERLIN, GLM, LPLWS, GOES)
    3-D volume      KMLB below its top beam, MRMS above (newest frame); the Steiner level likewise

QUALITY CONTROL
    MRMS's merged reflectivity is quality-controlled; raw KMLB is not. Three screens, in order:
      1. Below QC_MAX_KM, Z under QC_KEEP_DBZ and RHOHV under QC_RHO: insects, birds, sea clutter,
         anomalous propagation.
      2. Below QC_MAX_KM, Z under QC_NORHO_DBZ and RHOHV MISSING. NEXRAD withholds RHOHV at
         low-SNR gates, and a missing value never compares as "below 0.8" - on 24 Oct 2026 the
         first live frames were full of weak clear-air speckle over the pads that passed screen 1.
      3. Speckle, at every height: 8-connected echo patches of fewer than SPECKLE_GATES gates whose
         peak is under SPECKLE_KEEP_DBZ. Small but strong cells are kept; broad weak anvils are
         never small.
    All three run on each sweep's own ray x gate array, BEFORE sampling onto the map: far out a
    beam is wider than a map cell, so one stray gate maps onto a 2 x 2 block of cells and a
    cell-count threshold misses it. Above QC_MAX_KM only screen 3 acts: weak anvil edges have
    noisy, low RHOHV and must stay.

Run: OUT_DIR=site python kmlb_classify.py
"""
import datetime
import logging
import os
import re
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import requests

import mrms_classify as M

KMLB_VERSION = "kmlb-v2"
COLUMN_MODE = "single"                # "single": whole columns from one radar; "mixed": kmlb-v1
COLUMN_BOTTOM_M = 3000.0              # a KMLB column must be sampled from here (and to FULL_KM)
SITE = "KMLB"
RADAR = (28.1133, -80.6542)           # overridden by the volume's own latitude/longitude/altitude
BUCKET = "https://unidata-nexrad-level2.s3.amazonaws.com"
TOL_MIN = 6.0                         # frame may use a volume whose midpoint is this close
MID_EST_MIN = 2.5                     # scan midpoint ~ start + 2.5 min (VCP 212/215: 4.4-6 min scans)
AE = 6371000.0 * 4.0 / 3.0            # 4/3-earth radius, m
HALF_BEAM_DEG = 0.475                 # half the 0.95 deg half-power beamwidth
QC_RHO, QC_MAX_KM, QC_KEEP_DBZ = 0.80, 3.0, 35.0
QC_NORHO_DBZ = 20.0                   # screen 2: weaker than this with no RHOHV, below QC_MAX_KM
SPECKLE_GATES, SPECKLE_KEEP_DBZ = 20, 20.0  # screen 3: < 20 gates (~2.4 km2 at the pads), peak < 20 dBZ
FULL_KM = 15.0                        # KMLB "samples the column" if its top beam reaches this
VII_FLOOR = 0.5                       # kg/m2
LAYER_SUPER_M = (10058.0, 18288.0)    # 33-60 kft
STD_LEVELS_KM = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0,
                 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                 16.0, 17.0, 18.0, 19.0]             # the MRMS 3-D levels
ISO_KEYS = {"r0": 0.0, "r5": -5.0, "r10": -10.0, "r15": -15.0, "r20": -20.0}
LC39A = (28.6084, -80.6043)
NO_ECHO = -99.0
_VOL_RE = re.compile(r"KMLB(\d{8})_(\d{6})_V\d\d(\.gz)?$")


# --------------------------------------------------------------------------------------
# Archive listing and reading
# --------------------------------------------------------------------------------------
def vol_start(key):
    m = _VOL_RE.search(key)
    return datetime.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") if m else None


def list_day(sess, day):
    """[(start datetime, key)] for one UTC day, oldest first (public S3 REST, no boto)."""
    out, token = [], None
    while True:
        p = {"list-type": "2", "prefix": day.strftime(f"%Y/%m/%d/{SITE}/"), "max-keys": "1000"}
        if token:
            p["continuation-token"] = token
        r = sess.get(BUCKET + "/", params=p, timeout=(8, 40))
        r.raise_for_status()
        root = ET.fromstring(r.content)
        ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
        for c in root.findall(f"{ns}Contents"):
            k = c.findtext(f"{ns}Key")
            t = vol_start(k) if k and not k.endswith("_MDM") else None
            if t:
                out.append((t, k))
        trunc = (root.findtext(f"{ns}IsTruncated") or "").lower() == "true"
        token = root.findtext(f"{ns}NextContinuationToken") if trunc else None
        if not token:
            return sorted(out)


def _units_epoch(units):
    """'seconds since 2025-07-31T16:05:02Z' -> datetime."""
    m = re.search(r"since\s+(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)", units or "")
    return datetime.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S") if m else None


def read_volume(raw, key):
    """Download bytes -> sweeps dict (see from_radar)."""
    import pyart
    with tempfile.NamedTemporaryFile(suffix=os.path.basename(key)) as fp:
        fp.write(raw)
        fp.flush()
        # Only the two fields the hybrid uses: a full NEXRAD volume with all seven fields is
        # ~450 MB in memory, and the climatology runs four processes per runner.
        radar = pyart.io.read_nexrad_archive(
            fp.name, include_fields=["reflectivity", "cross_correlation_ratio"])
    return from_radar(radar, key)


def from_radar(radar, key):
    """Py-ART Radar -> {"sweeps": [{el, az, rng, z, rho, n}], lat, lon, alt, vcp, top_deg, nsweeps,
    mid, key}. One sweep per elevation: of split cuts and SAILS repeats, the one with the most
    reflectivity."""
    zf = radar.fields["reflectivity"]["data"]
    rf = radar.fields.get("cross_correlation_ratio", {}).get("data")
    rng = np.asarray(radar.range["data"], np.float64)
    best = {}
    for i in range(radar.nsweeps):
        s, e = radar.get_start_end(i)
        el = round(float(radar.fixed_angle["data"][i]), 1)
        z = np.ma.filled(np.ma.asarray(zf[s:e + 1]).astype(np.float32), np.nan)
        n = int(np.isfinite(z).sum())
        if el not in best or n > best[el]["n"]:
            best[el] = {"el": el, "az": np.asarray(radar.azimuth["data"][s:e + 1], np.float64),
                        "rng": rng, "z": z, "n": n,
                        "rho": (np.ma.filled(np.ma.asarray(rf[s:e + 1]).astype(np.float32), np.nan)
                                if rf is not None else None)}
    t0 = _units_epoch(radar.time.get("units")) or vol_start(key)
    tt = np.asarray(radar.time["data"], np.float64)
    mid = t0 + datetime.timedelta(seconds=float((tt.min() + tt.max()) / 2.0))
    sweeps = [best[k] for k in sorted(best)]
    return {"sweeps": sweeps, "lat": float(np.ravel(radar.latitude["data"])[0]),
            "lon": float(np.ravel(radar.longitude["data"])[0]),
            "alt": float(np.ravel(radar.altitude["data"])[0]),
            "vcp": (radar.metadata or {}).get("vcp_pattern"), "top_deg": sweeps[-1]["el"],
            "nsweeps": int(radar.nsweeps), "mid": mid, "key": key}


# --------------------------------------------------------------------------------------
# Beam geometry and sampling
# --------------------------------------------------------------------------------------
def column_geometry(la, lo, rlat, rlon):
    """Great-circle ground distance (m) and azimuth (deg) from the radar to every map cell."""
    LO, LA = np.meshgrid(np.radians(lo), np.radians(la))
    p0, l0 = np.radians(rlat), np.radians(rlon)
    d = 2 * np.arcsin(np.sqrt(np.sin((LA - p0) / 2) ** 2
                              + np.cos(p0) * np.cos(LA) * np.sin((LO - l0) / 2) ** 2))
    az = np.degrees(np.arctan2(np.sin(LO - l0) * np.cos(LA),
                               np.cos(p0) * np.sin(LA) - np.sin(p0) * np.cos(LA) * np.cos(LO - l0)))
    return d * 6371000.0, np.mod(az, 360.0)


def beam(el_deg, s_m, alt_m):
    """(height MSL m, slant range m, half beam width m) at ground distance s for elevation el,
    4/3-earth refraction."""
    e = np.radians(el_deg)
    a = s_m / AE
    h = AE * (np.cos(e) / np.cos(e + a) - 1.0) + alt_m
    r = AE * np.sin(a) / np.cos(e + a)
    return h, r, r * np.tan(np.radians(HALF_BEAM_DEG))


def qc_sweep(sw, alt_m):
    """The three QC screens on one sweep's ray x gate reflectivity. Returns (clean z, tally of
    gates removed by each screen)."""
    from scipy import ndimage
    z = sw["z"].copy()
    rng = sw["rng"]
    e = np.radians(sw["el"])
    h_agl = np.sqrt(rng ** 2 + AE ** 2 + 2 * rng * AE * np.sin(e)) - AE     # per gate, m above radar
    low = np.broadcast_to(h_agl < QC_MAX_KM * 1000.0, z.shape)
    echo = np.isfinite(z)
    if sw["rho"] is not None:
        rho = sw["rho"]
        j1 = echo & low & (z < QC_KEEP_DBZ) & (rho < QC_RHO)
        j2 = echo & low & (z < QC_NORHO_DBZ) & ~np.isfinite(rho)
    else:
        j1 = np.zeros(z.shape, bool)
        j2 = echo & low & (z < QC_NORHO_DBZ)
    z[j1 | j2] = np.nan
    lab, n = ndimage.label(np.isfinite(z), structure=np.ones((3, 3), bool))
    drop = np.zeros(z.shape, bool)
    if n:
        size = np.bincount(lab.ravel())[1:]
        peak = np.asarray(ndimage.maximum(np.nan_to_num(z, nan=-999.0), lab, np.arange(1, n + 1)))
        small = (size < SPECKLE_GATES) & (peak < SPECKLE_KEEP_DBZ)
        drop = (lab > 0) & small[np.maximum(lab - 1, 0)]
        z[drop] = np.nan
    return z, {"echo": int(echo.sum()), "low_rhohv": int(j1.sum()), "no_rhohv": int(j2.sum()),
               "speckle": int(drop.sum())}


def sample_sweeps(vol, la, lo):
    """Each sweep, QC'd in its own ray x gate space (qc_sweep), then sampled at every map column
    (nearest ray and gate). Returns Z, H, W of shape (K, ny, nx) and a QC tally in GATES summed
    over sweeps: dBZ (NO_ECHO = sampled, nothing there; NaN = this sweep does not reach the
    column), beam-centre height MSL and half beam width, both m."""
    s, az = column_geometry(la, lo, vol["lat"], vol["lon"])
    Z, H, W = [], [], []
    qc = {"echo": 0, "low_rhohv": 0, "no_rhohv": 0, "speckle": 0}
    for sw in vol["sweeps"]:
        zc, t = qc_sweep(sw, vol["alt"])
        for k_, v_ in t.items():
            qc[k_] += v_
        h, r, w = beam(sw["el"], s, vol["alt"])
        order = np.argsort(sw["az"])
        azs = sw["az"][order]
        k = np.searchsorted(azs, az) % azs.size
        km1 = (k - 1) % azs.size
        dk = np.abs(((azs[k] - az) + 180) % 360 - 180)
        dkm = np.abs(((azs[km1] - az) + 180) % 360 - 180)
        ray = order[np.where(dkm < dk, km1, k)]
        rng = sw["rng"]
        dr = float(rng[1] - rng[0])
        g = np.round((r - rng[0]) / dr).astype(np.int64)
        ok = (g >= 0) & (g < rng.size)
        z = zc[ray, np.clip(g, 0, rng.size - 1)].astype(np.float32)
        z = np.where(np.isfinite(z), z, NO_ECHO)
        Z.append(np.where(ok, z, np.nan))
        H.append(h)
        W.append(w)
    return np.stack(Z), np.stack(H), np.stack(W), qc


def at_heights(Z, H, W, target):
    """dBZ at target height(s) MSL (m; scalar or (ny, nx)): linear in dB between the two sweeps
    that bracket it, the nearer sweep's value when one of them has no echo, the end sweep within
    half a beam width beyond the lowest / highest beam. NaN where KMLB does not sample it."""
    K = Z.shape[0]
    t = np.broadcast_to(np.asarray(target, np.float64), Z.shape[1:])
    idx = (H <= t[None]).sum(axis=0) - 1                    # sweep at or below the target
    lo_i = np.clip(idx, 0, K - 1)
    hi_i = np.clip(idx + 1, 0, K - 1)
    take = lambda A, i: np.take_along_axis(A, i[None], axis=0)[0]
    z1, z2, h1, h2 = take(Z, lo_i), take(Z, hi_i), take(H, lo_i), take(H, hi_i)
    f = np.clip((t - h1) / np.maximum(h2 - h1, 1.0), 0.0, 1.0)
    both = (z1 > NO_ECHO) & (z2 > NO_ECHO) & np.isfinite(z1) & np.isfinite(z2)
    near = np.where(f < 0.5, z1, z2)
    near = np.where(np.isfinite(near), near, np.where(np.isfinite(z1), z1, z2))
    out = np.where(both, z1 + f * (z2 - z1), near)
    below = idx < 0
    out = np.where(below, np.where(t >= H[0] - W[0], Z[0], np.nan), out)
    above = idx >= K - 1
    out = np.where(above, np.where(t <= H[K - 1] + W[K - 1], Z[K - 1], np.nan), out)
    return out.astype(np.float32)


def top_edge(Z, H, W):
    """Highest height KMLB samples in each column (upper edge of the highest sweep that reaches
    it), m MSL; NaN where no sweep reaches the column."""
    reach = np.isfinite(Z)
    edge = np.where(reach, H + W, -np.inf).max(axis=0)
    return np.where(np.isfinite(edge), edge, np.nan)


def vii_from(levels_m, K, h10, h40):
    """Vertically integrated ice (kg/m2) between the -10 and -40 C heights, Mosier et al. 2011."""
    rho_i, n0 = 917.0, 4.0e6
    zlin = np.where(K > NO_ECHO, 10.0 ** (np.nan_to_num(K, nan=NO_ECHO) / 10.0), 0.0)
    m = np.pi * rho_i * n0 ** (3.0 / 7.0) * (5.28e-18 * zlin / 720.0) ** (4.0 / 7.0)
    tot = np.zeros(K.shape[1:], np.float64)
    for i in range(len(levels_m) - 1):
        a, b = levels_m[i], levels_m[i + 1]
        lo_ = np.clip(h10, a, b)
        hi_ = np.clip(h40, a, b)
        tot += 0.5 * (m[i] + m[i + 1]) * np.maximum(hi_ - lo_, 0.0)
    return tot.astype(np.float32)


# --------------------------------------------------------------------------------------
# One frame's hybrid
# --------------------------------------------------------------------------------------
class KmlbFrame:
    def __init__(self, vol, la, lo, iso, z0, valid_iso):
        self.vol, self.la, self.lo = vol, la, lo
        self.Z, self.H, self.W, self.qc = sample_sweeps(vol, la, lo)
        self.top = top_edge(self.Z, self.H, self.W)
        self.bottom = np.where(np.isfinite(self.Z), self.H - self.W, np.inf).min(axis=0)
        # Columns that come wholly from KMLB (single mode); in mixed mode every sampled height does.
        self.kcol = ((np.nan_to_num(self.top, nan=-1.0) >= FULL_KM * 1000.0)
                     & (self.bottom <= COLUMN_BOTTOM_M))
        self.levels_m = np.array(STD_LEVELS_KM) * 1000.0
        self.Kstd = np.stack([at_heights(self.Z, self.H, self.W, h) for h in self.levels_m])
        self.iso_m = self._isotherms(iso, z0)
        vt = datetime.datetime.strptime(valid_iso[:19], "%Y-%m-%dT%H:%M:%S")
        j = int(np.argmin(np.abs(la - LC39A[0])))
        i = int(np.argmin(np.abs(lo - LC39A[1])))
        top39 = float(self.top[j, i]) / 1000.0 if np.isfinite(self.top[j, i]) else None
        self.info = {
            "mode": "full" if (top39 or 0) >= FULL_KM else "partial",
            "volume": os.path.basename(vol["key"]), "vcp": vol["vcp"], "sweeps": vol["nsweeps"],
            "top_deg": vol["top_deg"], "lc39a_top_km": round(top39, 1) if top39 else None,
            "mid": vol["mid"].strftime("%H:%M:%SZ"),
            "offset_min": round((vol["mid"] - vt).total_seconds() / 60.0, 1),
            "column_mode": COLUMN_MODE,
            "kmlb_columns": round(float(self.kcol.mean()), 3),
            "qc": (f"below {QC_MAX_KM:g} km: RHOHV < {QC_RHO} under {QC_KEEP_DBZ:g} dBZ, or no RHOHV "
                   f"under {QC_NORHO_DBZ:g} dBZ; everywhere: patches < {SPECKLE_GATES} gates under "
                   f"{SPECKLE_KEEP_DBZ:g} dBZ"),
            "qc_removed": self.qc,
        }
        self.used = {}

    def _isotherms(self, iso, z0):
        """Isotherm heights (m MSL, per column) from the XMR profile, else the MRMS freezing level
        with a 6.5 C/km lapse rate. -40 C extends the -20 C height at 7 C/km."""
        out = {}
        for k, tc in ISO_KEYS.items():
            if iso and tc in iso:
                out[tc] = np.full(z0.shape, float(iso[tc]), np.float64)
            else:
                out[tc] = z0.astype(np.float64) + (-tc) / 6.5 * 1000.0
        out[-40.0] = out[-20.0] + 20.0 / 7.0 * 1000.0
        return out

    def apply_fields(self, F):
        if COLUMN_MODE == "single":
            return self._apply_single(F)
        sampled = lambda A: np.isfinite(A)
        for key, tc in ISO_KEYS.items():
            K = at_heights(self.Z, self.H, self.W, self.iso_m[tc])
            F[key] = np.where(sampled(K), K, F[key]).astype(np.float32)
            self.used[key] = float(sampled(K).mean())
        Ks = self.Kstd
        anyk = np.isfinite(Ks).any(axis=0)
        kmax = np.where(anyk, np.nanmax(np.where(np.isfinite(Ks), Ks, -np.inf), axis=0), NO_ECHO)
        kmax = np.where(np.isfinite(kmax), kmax, NO_ECHO)
        full = np.nan_to_num(self.top, nan=-1.0) >= FULL_KM * 1000.0
        mrms_comp = F["comp"].copy()
        F["comp"] = np.where(full, kmax, np.maximum(kmax, mrms_comp)).astype(np.float32)
        kidx = np.argmax(np.where(np.isfinite(Ks), Ks, -np.inf), axis=0)
        kh = np.where(kmax > NO_ECHO, self.levels_m[kidx], -1.0)
        F["hmax"] = np.where(full | (kmax > mrms_comp), kh, F["hmax"]).astype(np.float32)
        lay = (self.levels_m >= LAYER_SUPER_M[0]) & (self.levels_m <= LAYER_SUPER_M[1])
        L = np.where(np.isfinite(Ks[lay]), Ks[lay], -np.inf).max(axis=0)
        L = np.where(np.isfinite(L), L, NO_ECHO)
        whole = np.nan_to_num(self.top, nan=-1.0) >= LAYER_SUPER_M[1]
        part = np.isfinite(Ks[lay]).any(axis=0)
        F["super"] = np.where(whole, L, np.where(part, np.maximum(L, F["super"]), F["super"])
                              ).astype(np.float32)
        h10, h40 = self.iso_m[-10.0], self.iso_m[-40.0]
        kv = vii_from(self.levels_m, Ks, h10, h40)
        kv = np.where(kv >= VII_FLOOR, kv, 0.0)
        covers = np.nan_to_num(self.top, nan=-1.0) >= h40
        F["vii"] = np.where(covers, kv, F["vii"]).astype(np.float32)
        self.used.update({"comp_full": float(full.mean()), "vii": float(covers.mean())})
        self.info["kmlb_share"] = {k: round(v, 3) for k, v in self.used.items()}

    def _apply_single(self, F):
        """Every rule input from one radar per column: KMLB in self.kcol, MRMS (untouched) elsewhere."""
        kc = self.kcol
        for key, tc in ISO_KEYS.items():
            K = at_heights(self.Z, self.H, self.W, self.iso_m[tc])
            F[key] = np.where(kc & np.isfinite(K), K, F[key]).astype(np.float32)
        Ks = self.Kstd
        fin = np.isfinite(Ks)
        kmax = np.where(fin.any(axis=0), np.where(fin, Ks, -np.inf).max(axis=0), NO_ECHO)
        kmax = np.where(np.isfinite(kmax), kmax, NO_ECHO)
        kh = np.where(kmax > NO_ECHO, self.levels_m[np.argmax(np.where(fin, Ks, -np.inf), axis=0)], -1.0)
        lay = (self.levels_m >= LAYER_SUPER_M[0]) & (self.levels_m <= LAYER_SUPER_M[1])
        L = np.where(np.isfinite(Ks[lay]), Ks[lay], -np.inf).max(axis=0)
        L = np.where(np.isfinite(L), L, NO_ECHO)
        kv = vii_from(self.levels_m, Ks, self.iso_m[-10.0], self.iso_m[-40.0])
        kv = np.where(kv >= VII_FLOOR, kv, 0.0)
        F["comp"] = np.where(kc, kmax, F["comp"]).astype(np.float32)
        F["hmax"] = np.where(kc, kh, F["hmax"]).astype(np.float32)
        F["super"] = np.where(kc, L, F["super"]).astype(np.float32)
        F["vii"] = np.where(kc, kv, F["vii"]).astype(np.float32)
        share = float(kc.mean())
        self.used = {k: share for k in list(ISO_KEYS) + ["comp_full", "vii"]}
        self.info["kmlb_share"] = {k: round(v, 3) for k, v in self.used.items()}

    def _use(self, K):
        """Where a KMLB value replaces MRMS's: KMLB columns (single) or any sampled height (mixed)."""
        return (self.kcol & np.isfinite(K)) if COLUMN_MODE == "single" else np.isfinite(K)

    def merge_volume(self, vol, heights_km):
        out = vol.copy()
        for i, h in enumerate(heights_km):
            K = at_heights(self.Z, self.H, self.W, float(h) * 1000.0)
            out[i] = np.where(self._use(K), K, out[i])
        return out

    def merge_level(self, lvl, h_km):
        K = at_heights(self.Z, self.H, self.W, float(h_km) * 1000.0)
        return np.where(self._use(K), K, lvl).astype(np.float32)

    def extra_planes(self, shape):
        """Plane 14: highest height KMLB samples here, units of 100 m; 255 = not sampled."""
        t = self.top
        return [np.where(np.isfinite(t), np.clip(np.round(t / 100.0), 0, 253), 255).astype(np.uint8)]


# --------------------------------------------------------------------------------------
# The hook object
# --------------------------------------------------------------------------------------
class Hybrid:
    def __init__(self):
        self.sess = requests.Session()
        self.sess.headers["User-Agent"] = "CloudScope-KMLB/1.0 (launch weather nowcast)"
        self.days, self.vols, self.last_note = {}, {}, "not attempted"

    def _listing(self, t):
        out = []
        for d in {(t - datetime.timedelta(minutes=15)).date(), t.date()}:
            if d not in self.days:
                try:
                    self.days[d] = list_day(self.sess, d)
                except Exception as e:
                    logging.warning(f"KMLB listing {d}: {type(e).__name__}: {e}")
                    self.days[d] = []
            out += self.days[d]
        return sorted(out)

    def _fetch(self, key):
        if key not in self.vols:
            r = self.sess.get(f"{BUCKET}/{key}", timeout=(8, 60))
            r.raise_for_status()
            self.vols[key] = read_volume(r.content, key)
        return self.vols[key]

    def prepare(self, sess, valid_iso, la, lo, iso, z0):
        vt = datetime.datetime.strptime(valid_iso[:19], "%Y-%m-%dT%H:%M:%S")
        cand = [(abs((t + datetime.timedelta(minutes=MID_EST_MIN) - vt).total_seconds()), t, k)
                for t, k in self._listing(vt)]
        cand = [c for c in cand if c[0] <= TOL_MIN * 60]
        if not cand:
            self.last_note = f"no KMLB volume within {TOL_MIN:g} min of this frame"
            logging.info(f"KMLB: {self.last_note}")
            return None
        _, t, key = min(cand)
        try:
            vol = self._fetch(key)
        except Exception as e:
            self.last_note = f"{os.path.basename(key)}: {type(e).__name__}: {e}"
            logging.warning(f"KMLB: {self.last_note}")
            return None
        fr = KmlbFrame(vol, la, lo, iso, z0, valid_iso)
        logging.info(f"KMLB: {fr.info['volume']} VCP {fr.info['vcp']} to {fr.info['top_deg']} deg, "
                     f"LC-39A sampled to {fr.info['lc39a_top_km']} km ({fr.info['mode']}), "
                     f"midpoint {fr.info['offset_min']:+.1f} min from the frame; QC removed "
                     f"{fr.qc['low_rhohv']} low-RHOHV, {fr.qc['no_rhohv']} no-RHOHV, "
                     f"{fr.qc['speckle']} speckle of {fr.qc['echo']} echo gates")
        return fr

    @staticmethod
    def empty_planes(shape):
        return [np.full(shape, 255, np.uint8)]


def main():
    M.VIEWER_VERSION_EXPECTED = KMLB_VERSION
    M.HYBRID = Hybrid()
    M.main()


if __name__ == "__main__":
    main()
