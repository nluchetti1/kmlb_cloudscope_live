# CloudScope Radar · KMLB hybrid

KMLB Level II merged with MRMS and scored against the lightning launch commit criteria with the
same rules as the MRMS page. Wherever KMLB's beams sample a height over a map column, that height
comes from KMLB; above its highest beam (and in the cone of silence) it comes from MRMS. Each frame
uses the KMLB volume whose scan midpoint is nearest the frame time, and says how far up KMLB
reached ("full depth over the pads" with AVSET off, or "sampled to 6.8 km over LC-39A, MRMS above").

## Files
| File | What |
|---|---|
| `kmlb_classify.py` | KMLB listing (unidata-nexrad-level2), Py-ART read, beam sampling, the field merge; runs `mrms_classify.main()` with the hybrid hook set |
| `mrms_classify.py` | The MRMS classifier, v23 rules + inert hybrid hooks (identical behaviour with no hook set) |
| `web/index.html` | The page (kmlb-v1). Set `MRMS_PAGE_URL` near the top to link the MRMS page |
| `.github/workflows/kmlb.yml` | The live job: dispatch every 5 min from cron-job.org, publishes `gh-pages` |
| `.github/workflows/kmlb_validate.yml` + `kmlb_validate.py` | One-off: a real volume through the page's code |
| `test_kmlb_unit.py`, `kmlb_synth.py`, `test_kmlb_harness.py`, `test_kmlb_page.js` | Offline tests (`python test_kmlb_unit.py`, `python test_kmlb_harness.py`, `node test_kmlb_page.js`) |

## Setup
1. Commit everything to `kmlb_cloudscope_live` (replace the probe files or keep them - nothing here uses them).
2. Run **KMLB validate** from the Actions tab (blank time = the Crew-11 scrub volume). Check the read
   time, the 15-17 elevations, "sampled 100%" at -20 C near LC-39A, and the QC fraction.
3. Settings -> Pages -> deploy from the `gh-pages` branch (created by the first live run).
4. cron-job.org: a second job POSTing to `.../actions/workflows/kmlb.yml/dispatches` every 5 min
   (fine-grained token for this repo, Actions read & write; body `{"ref":"main"}`).

## Tunables (top of kmlb_classify.py)
`TOL_MIN` (6 min), `QC_RHO` / `QC_MAX_KM` / `QC_KEEP_DBZ` (0.80, 3 km, 35 dBZ), `FULL_KM` (15 km),
`VII_FLOOR` (0.5 kg/m2 - approximates MRMS VII; tune on overlapping cases), `LAYER_SUPER_M`
(33-60 kft - the MRMS layer-composite definition as I understand it; verify).
