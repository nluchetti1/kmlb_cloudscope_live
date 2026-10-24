"""Synthetic KMLB volumes from an analytic 3-D storm, shared by the KMLB tests."""
import datetime as dt
import numpy as np
import kmlb_classify as K

LA = np.round(29.595 - 0.01 * np.arange(200), 3)
LO = np.round(-81.595 + 0.01 * np.arange(200), 3)
STORM = (28.60, -80.60)          # ~54 km from KMLB, beside the pads
CLUTTER = (28.20, -80.65)        # ~10 km north of KMLB
STRONG = (2838, -8071)           # map cell (floor of lat*100, lon*100): a lone 30 dBZ low cell

def truth(lat, lon, h):
    """Storm: 50 dBZ core, -2.5 dBZ/km outward, -2 dBZ per km up, top 12 km, radius 15 km.
    Plus low clutter near KMLB: 15 dBZ below 800 m within 6 km (RHOHV 0.5), and a weak anvil
    sheet 10-11.5 km at 2 dBZ over the storm out to 30 km (RHOHV 0.6 - noisy but real)."""
    r = np.hypot((lat - STORM[0]) * 111.2, (lon - STORM[1]) * 97.7)
    z = np.where((r < 15) & (h < 12000), 50 - 2.5 * r - 2.0 * h / 1000.0, np.nan)
    z = np.where(np.isfinite(z) & (z < 0), np.nan, z)
    anvil = (r < 30) & (h > 10000) & (h < 11500)
    z = np.where(anvil & ~np.isfinite(z), 2.0, z)
    rc = np.hypot((lat - CLUTTER[0]) * 111.2, (lon - CLUTTER[1]) * 97.7)
    clut = (rc < 6) & (h < 800)
    z = np.where(clut, 15.0, z)
    rho = np.where(clut, 0.5, np.where(anvil, 0.6, 0.98))
    # 24 Oct 2026 speckle: isolated single map cells of weak echo near the pads with RHOHV
    # withheld (NaN), weak isolated cells aloft, and one lone strong low cell that must survive.
    ci, cj = np.floor(lat * 100).astype(int), np.floor(lon * 100).astype(int)   # = map cells
    iso = ((ci * 7 + cj * 13) % 23 == 0)
    pads = (lat > 28.30) & (lat < 28.45) & (lon > -80.75) & (lon < -80.55) & (h < 1500) & iso
    z = np.where(pads, 8.0, z)
    rho = np.where(pads, np.nan, rho)
    aloft = (lat > 29.20) & (lat < 29.50) & (lon > -81.20) & (lon < -80.90) & (h > 7000) & (h < 9000) & iso
    z = np.where(aloft & ~np.isfinite(z), 5.0, z)
    lone = (ci == STRONG[0]) & (cj == STRONG[1]) & (h < 2000)
    z = np.where(lone, 30.0, z)
    rho = np.where(lone, 0.98, rho)
    return z, rho

def synth_volume(elevs, start=dt.datetime(2025, 7, 31, 16, 5, 2)):
    az = np.arange(0, 360, 0.5) + 0.25
    rng = 2125.0 + 250.0 * np.arange(1000)       # to ~252 km
    sweeps = []
    lat0, lon0, alt = 28.1133, -80.6542, 30.0
    for el in elevs:
        e = np.radians(el)
        R, A = np.meshgrid(rng, np.radians(az))
        h = np.sqrt(R**2 + K.AE**2 + 2 * R * K.AE * np.sin(e)) - K.AE + alt
        s = K.AE * np.arcsin(R * np.cos(e) / (K.AE + h - alt))
        d = s / 6371000.0
        p0, l0 = np.radians(lat0), np.radians(lon0)
        lat = np.arcsin(np.sin(p0) * np.cos(d) + np.cos(p0) * np.sin(d) * np.cos(A))
        lon = l0 + np.arctan2(np.sin(A) * np.sin(d) * np.cos(p0), np.cos(d) - np.sin(p0) * np.sin(lat))
        z, rho = truth(np.degrees(lat), np.degrees(lon), h)
        sweeps.append({"el": el, "az": az, "rng": rng, "z": z.astype(np.float32), "rho": rho.astype(np.float32), "n": 0})
    return {"sweeps": sweeps, "lat": lat0, "lon": lon0, "alt": alt, "vcp": 215, "top_deg": elevs[-1],
            "nsweeps": len(elevs), "mid": start + dt.timedelta(minutes=2.5), "key": "x/KMLB20250731_160502_V06"}

FULL = [0.5, 0.9, 1.3, 1.8, 2.4, 3.1, 4.0, 5.1, 6.4, 8.0, 10.0, 12.0, 14.0, 16.7, 19.5]
TRUNC = FULL[:9]
