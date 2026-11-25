"""Weather Scoreboard ensemble job (the snow and rain odds, and the ensembles on the models map).

Twice a day (see .github/workflows/ensembles.yml) this script:
  1. downloads the two big ensembles: the European ENS (ECMWF open data, 51 runs: a control and 50 variations) and the
     American GEFS (NOAA, 31 runs);
  2. adds up, for every run, total snow and total rain every 12 hours out to 7 days;
  3. saves for every Weather Scoreboard city each run's totals over 3 and 7 days (the website works out the odds,
     ranges and charts from them), and, for the models map, on the same grid as the other models:
       <e>-snow.bin, <e>-rain.bin             the share of runs at or above each amount, every 12 hours (whole
                                              percents, 255 = no data; layers: each time × each amount);
       <e>-mean-snow.bin, <e>-mean-rain.bin   the average of all the runs, every 12 hours (snow in tenths of an inch,
                                              rain in hundredths, 65535 = no data);
  4. leaves everything in out/ens/, which the workflow publishes to this repository's ens-data branch. The website
     collects it from there every 10 minutes, so nothing is sent to the site and no key is needed.

Snow, as on the other model maps, is snowfall as liquid times 10 (the usual 10-to-1 ratio). Rain is rain only (the
precipitation that isn't snow). Totals count from the start of each ensemble's run. Uses the helpers in models.py.
"""
import datetime as dt, gzip, json, math, os, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests
import models as M   # shared helpers: get, log, subset, step_hours, sampler, grid_axes, GRID, CITIES, nearest

log = M.log
EOUT = os.path.join(M.OUT, "ens"); os.makedirs(EOUT, exist_ok=True)
STEPS = list(range(12, 169, 12))                      # hours after the run: every 12 hours to 7 days
WINDOWS = [72, 168]                                   # the city totals: next 3 days, next 7 days
THR = {"snow": [1, 3, 6, 12], "rain": [0.5, 1, 2]}    # inches

def runs(lag, every):
    """The most recent runs, newest first, allowing `lag` hours for a run to be published."""
    now = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=lag)
    c = now.replace(minute=0, second=0, microsecond=0, hour=now.hour // every * every)
    return [c - dt.timedelta(hours=every * i) for i in range(3)]

# ---------------------------------------------------------------- European ENS (ECMWF open data, CC BY 4.0)
# Snowfall (sf) and total precipitation (tp) add up from the start of the run, so only the 12-hourly times are needed.
# Only the 00 and 12 UTC runs go past day 6, so those are the ones used. The control run (cf) and the 50 variations (pf)
# are separate downloads, so all 51 are counted.
def eens(cyc):
    from ecmwf.opendata import Client
    import pygrib
    tp, sf = {}, {}; lat = lon = None
    for kind in ("cf", "pf"):
        target = os.path.join(M.OUT, f"eens-{kind}.grib2"); err = None
        for src in ("ecmwf", "azure", "aws"):   # ECMWF's own server first; the cloud copies can answer "slow down" (503)
            try:
                t0 = time.time()
                # At most 15 minutes per source, then the next one.
                M.limited(900, Client(source=src).retrieve, date=cyc.strftime("%Y%m%d"), time=cyc.hour, stream="enfo",
                          type=kind, step=STEPS, param=["tp", "sf"], target=target)
                log(f"  ENS {kind} from {src}: {os.path.getsize(target) // (1 << 20)} MB in {time.time() - t0:.0f}s"); err = None; break
            except Exception as e:
                err = e; log(f"  ENS {kind} source {src} failed: {type(e).__name__} {str(e)[:160]}")
        if err: raise RuntimeError(f"ENS {kind} not available")
        with pygrib.open(target) as g:
            for m in g:
                try: n = int(m["perturbationNumber"])
                except Exception: n = 0
                if kind == "cf": n = 0
                try: s = M.step_hours(m.endStep)
                except Exception: s = M.step_hours(m.step)
                lats, lons = m.latlons()
                lat, lon, v = M.subset(lats, lons, np.array(m.values, dtype="float32"))
                (tp if m.shortName == "tp" else sf)[(n, s)] = v * 1000 / 25.4      # metres of water → inches of water
        os.remove(target)
    members = sorted(n for n in {n for n, _ in tp} if all((n, s) in tp and (n, s) in sf for s in STEPS))
    log(f"  ENS: {len(members)} complete runs")
    snow = {s: np.stack([sf[(n, s)] * 10 for n in members]) for s in STEPS}                      # inches of snow at 10:1
    rain = {s: np.stack([np.clip(tp[(n, s)] - sf[(n, s)], 0, None) for n in members]) for s in STEPS}
    return dict(lat=lat, lon=lon, snow=snow, rain=rain, members=len(members), run=cyc)

# ---------------------------------------------------------------- American GEFS (NOAA, public domain)
# From NOAA's copy on Amazon's cloud (no rate limits), 0.5-degree files every 6 hours. Only the three fields needed
# (precipitation, the snow flag and 2-metre temperature) are downloaded from each file, using its index.
GEFS = "https://noaa-gefs-pds.s3.amazonaws.com/"
def gefs_fields(url):
    idx = M.get(url + ".idx", tries=3, timeout=60, ok404=True)
    if not idx: return None
    lines = idx.decode("utf-8", "ignore").strip().splitlines()
    offs = [int(l.split(":")[1]) for l in lines]
    want = []
    for i, l in enumerate(lines):
        p = l.split(":")
        if (p[3] == "APCP" and p[4] == "surface") or (p[3] == "CSNOW" and p[4] == "surface") or (p[3] == "TMP" and p[4] == "2 m above ground"):
            want.append((offs[i], offs[i + 1] - 1 if i + 1 < len(offs) else None))
    blob = b""
    for a, b in want:
        for k in range(3):
            try:
                r = requests.get(url, headers={**M.UA, "Range": f"bytes={a}-{'' if b is None else b}"}, timeout=60)
                if r.status_code in (200, 206): blob += r.content; break
            except Exception:
                pass
            time.sleep(2 * (k + 1))
        else:
            return None
    return blob

def gens(cyc):
    import pygrib
    ymd, hh = cyc.strftime("%Y%m%d"), cyc.strftime("%H")
    steps = list(range(6, STEPS[-1] + 1, 6))
    names = ["gec00"] + [f"gep{i:02d}" for i in range(1, 31)]
    url = lambda mem, f: f"{GEFS}gefs.{ymd}/{hh}/atmos/pgrb2ap5/{mem}.t{hh}z.pgrb2a.0p50.f{f:03d}"
    # Is the run complete? Its last file for every member must be there.
    probe = M.get(url("gep30", steps[-1]) + ".idx", tries=1, timeout=30, ok404=True)
    if not probe: raise RuntimeError(f"GEFS {ymd} {hh}Z not complete yet")
    jobs = [(mem, f) for mem in names for f in steps]
    t0 = time.time(); got = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for job, blob in zip(jobs, ex.map(lambda j: gefs_fields(url(*j)), jobs)): got[job] = blob
    log(f"  GEFS: {sum(1 for b in got.values() if b)} of {len(jobs)} files in {time.time() - t0:.0f}s, "
        f"{sum(len(b) for b in got.values() if b) // (1 << 20)} MB")
    tmp = os.path.join(M.OUT, "gens.grib2"); lat = lon = None
    snow = {s: [] for s in STEPS}; rain = {s: [] for s in STEPS}; kept = 0
    for mem in names:
        tot = {0: 0}; S = R = 0; frac_ok = True; done = True; P_prev = 0; mine_s = {}; mine_r = {}
        for f in steps:
            blob = got.get((mem, f))
            if not blob: done = False; break
            open(tmp, "wb").write(blob)
            temp = csnow = None; ap = []
            with pygrib.open(tmp) as g:
                for m in g:
                    lats, lons = m.latlons()
                    la, lo, v = M.subset(lats, lons, np.array(m.values, dtype="float32"))
                    lat, lon = la, lo
                    if m.shortName == "tp":
                        try: s0, s1 = M.step_hours(m.startStep), M.step_hours(m.endStep)
                        except Exception: s0, s1 = f - 6, f
                        ap.append((s0, s1, v))
                    elif m.shortName == "csnow": csnow = v
                    elif m.shortName == "2t": temp = v
            for s0, s1, v in sorted(ap, key=lambda x: x[0], reverse=True):   # chain the buckets: 0-6, 6-12, …
                if s0 in tot and s1 not in tot: tot[s1] = tot[s0] + v
            if f not in tot: done = False; break
            inc = np.clip(tot[f] - P_prev, 0, None) / 25.4; P_prev = tot[f]       # mm → inches of water, this 6 hours
            if csnow is not None: frac = np.clip(csnow, 0, 1)
            elif temp is not None: frac = (temp <= 273.15 + 0.5).astype("float32"); frac_ok = False
            else: done = False; break
            S = S + inc * frac * 10; R = R + inc * (1 - frac)
            if f in STEPS: mine_s[f] = S; mine_r[f] = R
        if done:   # only runs that came in whole
            kept += 1
            for s in STEPS: snow[s].append(mine_s[s]); rain[s].append(mine_r[s])
        if not frac_ok: log(f"  GEFS {mem}: no snow flag, snow from temperature")
    if os.path.exists(tmp): os.remove(tmp)
    if kept < 20: raise RuntimeError(f"GEFS: only {kept} complete runs")
    return dict(lat=lat, lon=lon, snow={s: np.stack(snow[s]) for s in STEPS}, rain={s: np.stack(rain[s]) for s in STEPS}, members=kept, run=cyc)

# ---------------------------------------------------------------- outputs
def city_totals(e):
    """Each city's totals from every run: [snow 3 days, snow 7 days, rain 3 days, rain 7 days], each a sorted list."""
    out = {}
    for c in M.CITIES:
        i, j = M.nearest(e, c["lat"], c["lon"])
        row = []
        for var, nd in (("snow", 1), ("rain", 2)):
            for w in WINDOWS: row.append(sorted(round(float(x), nd) for x in e[var][w][:, i, j]))
        out[c["k"]] = row
    return out

def grid_sampler(e):
    lats, lons = M.grid_axes(); tlon, tlat = np.meshgrid(lons, lats)
    return M.sampler(e, tlat, tlon)

def chance_grid(e, var, smp):
    """The share of runs at or above each amount, as whole percents (255 = no data), on the models map's grid:
    layers in order: each time (STEPS) × each amount (THR)."""
    layers = []
    for s in STEPS:
        for t in THR[var]:
            p = smp((e[var][s] >= t).mean(axis=0).astype("float32") * 100)
            layers.append(np.where(np.isnan(p), 255, np.clip(np.round(p), 0, 100)).astype("u1"))
    return np.stack(layers)

def mean_grid(e, var, smp):
    """The average of all the runs at each time (STEPS): snow in tenths of an inch, rain in hundredths (65535 = no data)."""
    sc = 10 if var == "snow" else 100; layers = []
    for s in STEPS:
        v = smp(e[var][s].mean(axis=0).astype("float32"))
        layers.append(np.where(np.isnan(v), 65535, np.clip(np.round(v * sc), 0, 65534)).astype("<u2"))
    return np.stack(layers)

def main():
    got = {}
    for key, fn, lag, every in (("eens", eens, 9, 12), ("gens", gens, 6, 6)):
        for cyc in runs(lag, every):
            try:
                log(f"{key}: trying the {cyc:%Y-%m-%d %HZ} run"); t0 = time.time()
                got[key] = fn(cyc); log(f"{key}: ok, {got[key]['members']} runs, in {time.time() - t0:.0f}s"); break
            except Exception as e:
                log(f"{key}: {type(e).__name__}: {str(e)[:200]}")
        else:
            log(f"{key}: skipped this time")
    if not got:
        log("No ensemble could be downloaded."); json.dump({"ok": False, "log": M.LOG}, open(os.path.join(EOUT, "latest.json"), "w")); sys.exit(1)
    files = []
    cities = {k: city_totals(e) for k, e in got.items()}
    open(os.path.join(EOUT, "cities.json.gz"), "wb").write(gzip.compress(json.dumps(cities, separators=(",", ":")).encode(), 6)); files.append("cities.json.gz")
    for k, e in got.items():
        smp = grid_sampler(e)
        for var in ("snow", "rain"):
            n = f"{k}-{var}.bin"; open(os.path.join(EOUT, n), "wb").write(gzip.compress(chance_grid(e, var, smp).tobytes(), 6)); files.append(n)
            n = f"{k}-mean-{var}.bin"; open(os.path.join(EOUT, n), "wb").write(gzip.compress(mean_grid(e, var, smp).tobytes(), 6)); files.append(n)
    made = dt.datetime.now(dt.timezone.utc)
    meta = {"ok": True, "v": 2, "made": made.isoformat(), "stamp": made.strftime("%Y%m%d%H%M"), "steps": STEPS, "windows": WINDOWS, "thr": THR,
            "runs": {k: e["run"].isoformat() for k, e in got.items()}, "members": {k: e["members"] for k, e in got.items()},
            "grid": {k: M.GRID[k] for k in ("w", "e", "s", "n", "cols", "rows")}, "files": files, "log": M.LOG[-40:]}
    json.dump(meta, open(os.path.join(EOUT, "latest.json"), "w"))
    open(os.path.join(EOUT, "README.md"), "w").write("Ensemble odds for WeatherScoreboard.com, made by ensembles.py. Not offered for reuse. "
        "Data: NOAA GEFS (public domain); ECMWF ENS forecast data © ECMWF, CC BY 4.0.\n")
    log(f"done: {', '.join(files)}")

if __name__ == "__main__":
    main()
