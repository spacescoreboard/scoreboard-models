"""Weather Scoreboard model job.

Twice a day (see .github/workflows/models.yml) this script:
  1. downloads the next 72 hours of three forecast models: the American GFS (NOAA), the European IFS (ECMWF open
     data) and the Canadian GDPS/GEM (Environment and Climate Change Canada);
  2. adds the National Weather Service's own forecast grids and NOAA's HRRR (both about 2 miles apart);
  3. reads each model's forecast for every Weather Scoreboard city for 7 days (temperature, rain and melted snow,
     snow; every 3 hours, then every 6 on day 7), which the website keeps to grade the models later, and saves
     each model's grids for the lower 48 for the website's interactive map;
  3. when a storm is coming (4 inches or more for a big city in any model), draws the animated "model fight"
     snowfall map as an MP4 video, a GIF and a still image;
  4. sends everything to the website, which checks the secret key.

Snow is the model's snowfall as liquid, times 10 (the usual 10-to-1 ratio), so all three models are measured the
same way. Where a model's snowfall isn't available, snow is the precipitation that falls at or below about 33°F.

Settings come from GitHub secrets: SITE_URL (https://www.weatherscoreboard.com) and WIS_MODELS_KEY (from the
plugin's settings page). Run it by hand from the Actions tab with "Run workflow"; "force_map" draws a map even
with no storm, for testing.
"""
import datetime as dt, io, json, math, os, re, subprocess, sys, time, traceback
import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
CITIES = json.load(open(os.path.join(HERE, "cities.json")))
STEPS = list(range(0, 145, 3)) + list(range(150, 169, 6))   # forecast hours: every 3 hours to day 6, then every 6 to day 7
PREV = {f: STEPS[i - 1] for i, f in enumerate(STEPS) if i}
MAP_STEPS = [f for f in STEPS if f <= 72]   # the storm animation covers 3 days
I72 = STEPS.index(72)
REGION = dict(w=-127.0, e=-65.0, s=23.0, n=51.0)
UA = {"User-Agent": "WeatherScoreboard models job (weatherscoreboard.com)"}
LOG = []
def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.append(m)

def get(url, tries=4, timeout=120, ok404=False):
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code == 200: return r.content
            if r.status_code == 404 and ok404: return None
            log("  HTTP", r.status_code, url[:140])
        except Exception as e:
            log("  error", type(e).__name__, url[:140])
        time.sleep(5 * (i + 1))
    return None

def cycle_candidates():
    """Most recent 00/12 UTC runs, newest first, allowing about 5 hours for a run to be published."""
    now = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=5)
    c = now.replace(minute=0, second=0, microsecond=0, hour=0 if now.hour < 12 else 12)
    return [c, c - dt.timedelta(hours=12), c - dt.timedelta(hours=24)]

def subset(lats, lons, *fields):
    """Cut a global grid to North America; longitudes as -180..180."""
    lons = np.where(lons > 180, lons - 360, lons)
    if lats.ndim == 1:
        lons2, lats2 = np.meshgrid(lons, lats)
    else:
        lats2, lons2 = lats, lons
    rows = np.where((lats2[:, 0] >= REGION["s"] - 1) & (lats2[:, 0] <= REGION["n"] + 1))[0] if lats2.ndim == 2 else None
    cols = np.where((lons2[0, :] >= REGION["w"] - 1) & (lons2[0, :] <= REGION["e"] + 1))[0]
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    return (lats2[r0:r1, c0:c1], lons2[r0:r1, c0:c1]) + tuple(f[r0:r1, c0:c1] for f in fields)

def grib_fields(blob):
    """All messages in a GRIB2 blob as dicts: shortName, step range, values, lats, lons."""
    import pygrib
    path = os.path.join(OUT, "tmp.grib2"); open(path, "wb").write(blob)
    out = []
    with pygrib.open(path) as g:
        for m in g:
            lats, lons = m.latlons()
            try: s0, s1 = int(m.startStep), int(m.endStep)
            except Exception: s0 = s1 = int(getattr(m, "step", 0))
            out.append(dict(name=m.shortName, s0=s0, s1=s1, values=np.array(m.values, dtype="float32"), lats=lats, lons=lons))
    return out

# ---------------------------------------------------------------- GFS (NOAA, public domain)
def gfs(cyc):
    ymd, hh = cyc.strftime("%Y%m%d"), cyc.strftime("%H")
    T, P, S = {}, {}, {}; lat = lon = None; tot = {0: 0}
    for f in STEPS:
        if f == 0: continue
        url = ("https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?dir=%2Fgfs." + ymd + "%2F" + hh + "%2Fatmos"
               f"&file=gfs.t{hh}z.pgrb2.0p25.f{f:03d}&var_TMP=on&var_APCP=on&var_CSNOW=on&lev_2_m_above_ground=on&lev_surface=on"
               f"&subregion=&toplat={REGION['n']+1}&bottomlat={REGION['s']-1}&leftlon={360+REGION['w']-1}&rightlon={360+REGION['e']+1}")
        blob = get(url, ok404=True)
        if not blob: raise RuntimeError(f"GFS f{f:03d} not available for {ymd} {hh}Z")
        msgs = grib_fields(blob); time.sleep(1)  # NOMADS asks for gentle use
        temp = next(m for m in msgs if m["name"] == "2t")
        lat, lon, tv = subset(temp["lats"], temp["lons"], temp["values"])
        T[f] = (tv - 273.15) * 9 / 5 + 32
        snowfrac = None
        for m in msgs:
            if m["name"] == "csnow": snowfrac = subset(m["lats"], m["lons"], m["values"])[2]
        for m in msgs:  # rain and melted snow, mm; GFS files hold 6-hour buckets
            if m["name"] == "tp":
                v = subset(m["lats"], m["lons"], m["values"])[2]
                if m["s0"] in tot: tot[m["s1"]] = tot[m["s0"]] + v
        if f not in tot: raise RuntimeError(f"GFS f{f:03d}: no precipitation total")
        P[f] = tot[f]
        inc = P[f] - P[PREV[f]] if PREV[f] in P else P[f]
        frac = snowfrac if snowfrac is not None else (T[f] <= 33).astype("float32")
        S[f] = (S.get(PREV[f], 0) + np.clip(inc, 0, None) * frac / 25.4 * 10)
    T[0] = T[3]; P[0] = np.zeros_like(P[3]); S[0] = np.zeros_like(P[3])
    return dict(lat=lat, lon=lon, T=T, P={k: v / 25.4 for k, v in P.items()}, S=S, run=cyc, snow_note="GFS snow where its precipitation type is snow")

# ---------------------------------------------------------------- ECMWF IFS open data (CC BY 4.0)
def ecmwf(cyc):
    from ecmwf.opendata import Client
    target = os.path.join(OUT, "ec.grib2"); snow_native = True
    for src in ("ecmwf", "aws"):
        try:
            c = Client(source=src)
            try:
                c.retrieve(date=cyc.strftime("%Y%m%d"), time=cyc.hour, type="fc", step=STEPS, param=["2t", "tp", "sf"], target=target)
            except Exception as e:
                log("  ECMWF snowfall not available (", type(e).__name__, "); using temperature")
                snow_native = False
                c.retrieve(date=cyc.strftime("%Y%m%d"), time=cyc.hour, type="fc", step=STEPS, param=["2t", "tp"], target=target)
            break
        except Exception as e:
            log("  ECMWF source", src, "failed:", type(e).__name__, str(e)[:160])
    else:
        raise RuntimeError("ECMWF not available")
    msgs = grib_fields(open(target, "rb").read())
    T, P, SF = {}, {}, {}; lat = lon = None
    for m in msgs:
        lat, lon, v = subset(m["lats"], m["lons"], m["values"])
        if m["name"] == "2t": T[m["s1"]] = (v - 273.15) * 9 / 5 + 32
        elif m["name"] == "tp": P[m["s1"]] = v * 1000 / 25.4          # m → inches, accumulated from the start
        elif m["name"] == "sf": SF[m["s1"]] = v * 1000 / 25.4 * 10    # m of water → inches of snow at 10:1
    if snow_native and len(SF) == len(STEPS):
        S = SF
    else:
        S = {0: np.zeros_like(P[0])}
        for f in STEPS[1:]: S[f] = S[PREV[f]] + np.clip(P[f] - P[PREV[f]], 0, None) * (T[f] <= 33) * 10
    if 0 not in P: P[0] = np.zeros_like(next(iter(P.values()))); S[0] = np.zeros_like(P[0])
    return dict(lat=lat, lon=lon, T=T, P=P, S=S, run=cyc, snow_note="ECMWF snowfall" if snow_native else "ECMWF snow from temperature")

# ---------------------------------------------------------------- Canadian GDPS / GEM (ECCC Data Server End-use Licence)
def gem(cyc):
    hh = cyc.strftime("%H"); base = f"https://dd.weather.gc.ca/today/model_gdps/15km/{hh}/"
    listing = get(base + "024/", ok404=True)
    if not listing: raise RuntimeError("GEM run not listed")
    names = re.findall(r'href="([^"]+\.grib2)"', listing.decode("utf-8", "ignore"))
    stamp = cyc.strftime("%Y%m%dT%HZ")
    def pick(pattern):
        hits = [n for n in names if n.startswith(stamp) and re.search(pattern, n)]
        return hits[0] if hits else None
    # Environment Canada renamed its variables in 2025 (TMP → AirTemp, APCP → Precip-Accum); accept either.
    tname = pick(r"_MSC_GDPS_AirTemp_AGL-2m_") or pick(r"_MSC_GDPS_TMP_AGL-2m_")
    pname = pick(r"_MSC_GDPS_Precip-Accum_Sfc_") or pick(r"_MSC_GDPS_APCP_Sfc_") or pick(r"_MSC_GDPS_(?:Total)?Precip(?:itation)?-Accum[^_]*_Sfc_")
    if not tname or not pname:
        vars_ = sorted(set(re.sub(r"^.*_MSC_GDPS_|_LatLon.*$", "", n) for n in names))
        log("  GEM variables with Temp, Precip or Snow in the name:", [v for v in vars_ if re.search("Temp_AGL-2m|Precip|Snow", v)])
        raise RuntimeError("GEM temperature or precipitation file not found")
    log("  GEM files:", re.sub(r"^.*_MSC_GDPS_|_LatLon.*$", "", tname), "and", re.sub(r"^.*_MSC_GDPS_|_LatLon.*$", "", pname))
    # About 108 large files: fetched 8 at a time, and a run with a few hours missing is still used (filled in between).
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(f, name, which) for f in STEPS if f for name, which in ((tname, "T"), (pname, "P"))]
    def fetch(job):   # downloads run side by side; decoding stays one at a time (it uses one temporary file)
        f, name, which = job
        return job, get(base + f"{f:03d}/" + re.sub(r"_PT\d{3}H", f"_PT{f:03d}H", name), tries=3, timeout=90)
    def decode(blob):
        if not blob: return None
        m = grib_fields(blob)[0]
        return subset(m["lats"], m["lons"], m["values"])
    got = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for job, blob in ex.map(fetch, jobs): got[job] = decode(blob)
    missing = [j for j in jobs if got[j] is None]
    for j in missing: got[j] = decode(fetch(j)[1])   # one more try, one at a time
    missing = [j for j in jobs if got[j] is None]
    if missing: log("  GEM hours missing:", sorted(set(f for f, _, _ in missing)))
    if len(missing) > 8 or any(f == STEPS[-1] for f, _, _ in missing): raise RuntimeError(f"GEM: {len(missing)} files missing")
    T, P = {}, {}; lat = lon = None
    for (f, name, which), res in got.items():
        if res is None: continue
        lat, lon, v = res
        (T if which == "T" else P)[f] = (v - 273.15) * 9 / 5 + 32 if which == "T" else v / 25.4   # K → °F; kg/m² (mm) → in, accumulated
    for store in (T, P):   # fill a missing hour from the hours on either side
        have = sorted(store)
        for f in STEPS[1:]:
            if f in store: continue
            lo = max((h for h in have if h < f), default=None); hi = min((h for h in have if h > f), default=None)
            if lo is None or hi is None: raise RuntimeError(f"GEM hour {f} missing")
            w = (f - lo) / (hi - lo); store[f] = store[lo] * (1 - w) + store[hi] * w
    T[0] = T[3]; P[0] = np.zeros_like(P[3]); S = {0: np.zeros_like(P[3])}
    for f in STEPS[1:]: S[f] = S[PREV[f]] + np.clip(P[f] - P[PREV[f]], 0, None) * (T[f] <= 33) * 10
    return dict(lat=lat, lon=lon, T=T, P=P, S=S, run=cyc, snow_note="GEM snow from temperature")

# ---------------------------------------------------------------- high-detail sources: NWS forecast grids and HRRR
# Both come on projected grids (Lambert conformal, about 2.5 and 3 km), kept whole, with the projection, so they can be
# sampled at any point. Values are lined up with STEPS (hours after `start`, the global models' run); where a source
# has no forecast for a step (HRRR past 48 hours, NWS snow and rain past 72), the value is missing (NaN).
def grib_proj(blob):
    """Messages of a GRIB2 blob on one projected grid: (lats, lons, projparams, [message dicts with values])."""
    import pygrib
    path = os.path.join(OUT, "tmp.grib2"); open(path, "wb").write(blob)
    out = []; lats = lons = proj = None
    with pygrib.open(path) as g:
        for m in g:
            if lats is None: lats, lons = m.latlons(); proj = m.projparams
            try: s0, s1 = int(m.startStep), int(m.endStep)
            except Exception: s0 = s1 = int(getattr(m, "step", 0))
            ana = m.analDate.replace(tzinfo=dt.timezone.utc)
            out.append(dict(name=m.shortName, s0=s0, s1=s1, t0=ana + dt.timedelta(hours=s0), t1=ana + dt.timedelta(hours=s1),
                            values=np.array(np.ma.filled(m.values.astype("float32"), np.nan))))
    return lats, lons, proj, out

def nws(start):
    """The National Weather Service's own forecast grids (NDFD, CONUS 2.5 km, public domain): temperature for 7 days,
    rain and melted snow (QPF) and snow for 3 days. Snow is the NWS forecasters' own snow amount."""
    base = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/"
    def read(vp, el):
        blob = get(base + vp + "/ds." + el + ".bin", timeout=300)
        if not blob: raise RuntimeError(f"NDFD {vp} {el} not available")
        return grib_proj(blob)
    lats, lons, proj, temp = read("VP.001-003", "temp")
    try: temp += read("VP.004-007", "temp")[3]
    except Exception as e: log("  NWS days 4-7 temperature:", str(e)[:120])
    qpf, snow = read("VP.001-003", "qpf")[3], read("VP.001-003", "snow")[3]
    T, P, S = {}, {}, {}
    tv = sorted(((m["t1"], m["values"]) for m in temp), key=lambda x: x[0])
    times = [t for t, _ in tv]
    for f in STEPS:
        t = start + dt.timedelta(hours=f)
        if t < times[0] or t > times[-1]: T[f] = None; continue
        j = next(i for i, x in enumerate(times) if x >= t)
        if times[j] == t or j == 0: T[f] = (tv[j][1] - 273.15) * 9 / 5 + 32
        else:  # between two forecast times: blend
            a = (t - times[j - 1]).total_seconds() / (times[j] - times[j - 1]).total_seconds()
            T[f] = ((tv[j - 1][1] * (1 - a) + tv[j][1] * a) - 273.15) * 9 / 5 + 32
    def accum(periods, scale, store):
        end = max(m["t1"] for m in periods)
        for f in STEPS:
            t = start + dt.timedelta(hours=f)
            if t > end: store[f] = None; continue
            tot = np.zeros_like(periods[0]["values"])
            for m in periods:  # the part of each period that falls between `start` and t
                a, b = max(m["t0"], start), min(m["t1"], t)
                if b <= a: continue
                tot += np.nan_to_num(m["values"]) * ((b - a).total_seconds() / (m["t1"] - m["t0"]).total_seconds())
            store[f] = tot * scale
    accum(qpf, 1 / 25.4, P)          # kg/m² (mm) → inches
    accum(snow, 39.37, S)            # metres of snow → inches
    return dict(lat=lats, lon=lons, proj=proj, T=T, P=P, S=S, run=max(m["t0"] for m in temp[:1]), snow_note="NWS forecast snow amount")

def hrrr(start):
    """NOAA's HRRR (3 km, public domain), the newest 48-hour run (00, 06, 12 or 18Z). Snow is precipitation the HRRR
    calls snow, at 10:1, like the GFS. Totals count from the HRRR run's own start."""
    now = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=3)
    c0 = now.replace(minute=0, second=0, microsecond=0, hour=now.hour // 6 * 6)
    for cyc in (c0, c0 - dt.timedelta(hours=6), c0 - dt.timedelta(hours=12)):
        try:
            ymd, hh = cyc.strftime("%Y%m%d"), cyc.strftime("%H")
            T, P, S = {}, {}, {}; tot = {0: 0}; lats = lons = proj = None
            for fh in range(3, 49, 3):
                url = ("https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl?dir=%2Fhrrr." + ymd + "%2Fconus"
                       f"&file=hrrr.t{hh}z.wrfsfcf{fh:02d}.grib2&var_TMP=on&var_APCP=on&var_CSNOW=on&lev_2_m_above_ground=on&lev_surface=on")
                blob = get(url, ok404=True)
                if not blob: raise RuntimeError(f"HRRR {hh}Z f{fh:02d} not available")
                lats, lons, proj, msgs = grib_proj(blob); time.sleep(1)
                frac = None
                for m in msgs:
                    if m["name"] == "2t": T[fh] = (m["values"] - 273.15) * 9 / 5 + 32
                    elif m["name"] == "csnow": frac = m["values"]
                    elif m["name"] == "tp" and m["s0"] in tot: tot[m["s1"]] = tot[m["s0"]] + m["values"]
                if fh not in tot or fh not in T: raise RuntimeError(f"HRRR f{fh:02d}: missing fields")
                P[fh] = tot[fh]
                inc = np.clip(P[fh] - (P.get(fh - 3, 0)), 0, None)
                S[fh] = S.get(fh - 3, 0) + inc * (frac if frac is not None else (T[fh] <= 33)) / 25.4 * 10
            T[0] = T[3]; P[0] = np.zeros_like(P[3]); S[0] = np.zeros_like(P[3])
            off = int(round((cyc - start).total_seconds() / 3600))
            line = lambda d, conv: {f: (conv(d[f - off]) if (f - off) in d else None) for f in STEPS}
            log(f"  HRRR run {cyc:%Y-%m-%d %HZ}, {off:+d} hours from the global models' run")
            return dict(lat=lats, lon=lons, proj=proj, run=cyc, snow_note="HRRR snow from its precipitation type",
                        T=line(T, lambda x: x), P=line(P, lambda x: x / 25.4), S=line(S, lambda x: x))
        except Exception as e:
            log(f"  HRRR {cyc:%HZ}: {str(e)[:160]}")
    raise RuntimeError("no HRRR run available")

def sampler(md, tlat, tlon):
    """A function that resamples one of the model's fields to the points tlat/tlon (2-D or 1-D arrays), blending the
    four nearest grid points; NaN outside the model's grid or where the model has no value."""
    if md.get("proj"):
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", md["proj"], always_xy=True)
        xs, _ = tr.transform(md["lon"][0, :], md["lat"][0, :]); _, ys = tr.transform(md["lon"][:, 0], md["lat"][:, 0])
        px, py = tr.transform(tlon, tlat); colax, rowax, wx, wy = xs, ys, px, py
    else:
        colax, rowax, wx, wy = md["lon"][0, :], md["lat"][:, 0], tlon, tlat
    def frac(axis, want):
        order = np.argsort(axis); a = axis[order]
        f = np.interp(want, a, np.arange(len(a)), left=np.nan, right=np.nan)
        ok = ~np.isnan(f); f = np.where(ok, f, 0)
        i0 = np.clip(np.floor(f).astype(int), 0, len(a) - 2); t = (f - i0).astype("float32")
        return order[i0], order[i0 + 1], t, ok
    c0, c1, tx, okx = frac(np.asarray(colax), np.asarray(wx)); r0, r1, ty, oky = frac(np.asarray(rowax), np.asarray(wy))
    bad = ~(okx & oky)
    def apply(a):
        if a is None: return np.full(np.shape(wx), np.nan, dtype="float32")
        v = (a[r0, c0] * (1 - tx) + a[r0, c1] * tx) * (1 - ty) + (a[r1, c0] * (1 - tx) + a[r1, c1] * tx) * ty
        return np.where(bad, np.nan, v).astype("float32")
    return apply

# ---------------------------------------------------------------- city forecasts
def nearest(md, la, lo):
    d = (md["lat"] - la) ** 2 + ((md["lon"] - lo) * math.cos(math.radians(la))) ** 2
    return np.unravel_index(np.argmin(d), d.shape)

def city_series(md):
    if md.get("proj"):  # high-detail grids: blend the nearest points; missing values become null
        la = np.array([c["lat"] for c in CITIES]); lo = np.array([c["lon"] for c in CITIES]); smp = sampler(md, la, lo)
        cols = {k: {f: smp(md[k][f]) for f in STEPS} for k in ("T", "P", "S")}
        r = lambda v, n: None if v != v else round(float(v), n)
        return {c["k"]: [[r(cols["T"][f][i], 1), r(cols["P"][f][i], 2), r(cols["S"][f][i], 1)] for f in STEPS] for i, c in enumerate(CITIES)}
    out = {}
    for c in CITIES:
        i, j = nearest(md, c["lat"], c["lon"])
        out[c["k"]] = [[round(float(md["T"][f][i, j]), 1), round(float(md["P"][f][i, j]), 2), round(float(md["S"][f][i, j]), 1)] for f in STEPS]
    return out

# ---------------------------------------------------------------- the map
def draw(models, totals, force=False):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Polygon as MPoly
    from pyproj import Transformer
    from PIL import Image
    # The site's fonts: from the fonts folder here, else from the plugin on the website, else a standard font.
    fdir = os.path.join(HERE, "fonts"); os.makedirs(fdir, exist_ok=True); have = 0
    site = os.environ.get("SITE_URL", "").rstrip("/")
    for f in ("AtkinsonHyperlegible-Regular.ttf", "AtkinsonHyperlegible-Bold.ttf", "BricolageGrotesque-ExtraBold.ttf"):
