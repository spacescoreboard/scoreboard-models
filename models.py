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

# Time budgets. A slow or stuck data server used to hold the whole job until GitHub stopped it at 75 minutes, and then
# nothing reached the site. Now each source gets its own share of time (DEADLINE, set in main), each download its own
# limit for the whole file (not just between bytes), and the job sends whatever it has by about 55 minutes.
JOB_START = time.time()
JOB_LIMIT = 55 * 60
DEADLINE = None

class OutOfTime(Exception):
    pass

# Whether the website answers GitHub at all (checked once at the start). When it doesn't, the job skips everything that
# talks to the site (its fonts, the pushes) instead of waiting minutes on each, and the site collects this run from
# GitHub Pages instead (out/pages/delivery, fetched by the plugin's server task).
SITE_OK = None
def site_ok():
    global SITE_OK
    if SITE_OK is None:
        url = os.environ.get("SITE_URL", "").rstrip("/")
        # For the host, if the site can't be reached: this runner's public IP, the site's IP, and the time.
        try:
            import socket, urllib.parse
            host = urllib.parse.urlparse(url).hostname or ""
            dest = sorted({a[4][0] for a in socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)}) if host else []
        except Exception as e:
            dest = [f"lookup failed: {type(e).__name__}"]
        try:
            mine = requests.get("https://api.ipify.org", timeout=10).text.strip()
        except Exception:
            mine = "unknown"
        log(f"  connection check {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S} UTC: runner IP {mine}, {host or 'site'} at {', '.join(dest) or 'unknown'}, port 443")
        try:
            r = requests.get(url + "/wp-json/wis/v1", headers=UA, timeout=(15, 30)) if url else None
            SITE_OK = bool(r is not None and r.status_code == 200 and "wis\\/v1" in r.text[:5000])
            if r is not None and not SITE_OK: log("  the website answered", r.status_code, "instead of its models address")
        except Exception as e:
            log("  the website didn't answer from GitHub:", type(e).__name__); SITE_OK = False
        if url and not SITE_OK: log("  sending this run through GitHub Pages only; the site picks it up from there")
    return SITE_OK

def get(url, tries=4, timeout=120, ok404=False):
    for i in range(tries):
        if DEADLINE and time.time() > DEADLINE:
            raise OutOfTime("this source's time is up")
        try:
            t0 = time.time(); r = requests.get(url, headers=UA, timeout=min(timeout, 60), stream=True)
            if r.status_code == 200:
                buf = bytearray()
                for chunk in r.iter_content(1 << 14):
                    buf += chunk
                    if time.time() - t0 > timeout * 2 or (DEADLINE and time.time() > DEADLINE):
                        r.close(); raise TimeoutError(f"download too slow ({len(buf) // 1024} KB in {time.time() - t0:.0f}s)")
                return bytes(buf)
            if r.status_code == 404 and ok404: return None
            log("  HTTP", r.status_code, url[:140])
        except OutOfTime:
            raise
        except Exception as e:
            log("  error", type(e).__name__, str(e)[:80], url[:140])
        time.sleep(5 * (i + 1))
    return None

def step_hours(v):
    """A GRIB step as hours: 6, "6", "6h" or "30m" (the NWS grids sometimes count in minutes)."""
    t = str(v).strip().lower()
    if t.endswith("m"): h = int(t[:-1]) / 60
    elif t.endswith("h"): h = int(t[:-1])
    else: h = int(float(t))
    return int(h) if h == int(h) else h

def cycle_candidates(lag=5):
    """Most recent 00/12 UTC runs, newest first, allowing `lag` hours for a run to be published (the European's open
    data comes out about 7 to 8 hours after its start time, the others about 4 to 5)."""
    now = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=lag)
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
            try: s0, s1 = step_hours(m.startStep), step_hours(m.endStep)
            except Exception: s0 = s1 = step_hours(getattr(m, "step", 0))
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
                limited(left(600), c.retrieve, date=cyc.strftime("%Y%m%d"), time=cyc.hour, type="fc", step=STEPS, param=["2t", "tp", "sf"], target=target)
            except Exception as e:
                log("  ECMWF snowfall not available (", type(e).__name__, "); using temperature")
                snow_native = False
                limited(left(600), c.retrieve, date=cyc.strftime("%Y%m%d"), time=cyc.hour, type="fc", step=STEPS, param=["2t", "tp"], target=target)
            break
        except Exception as e:
            log("  ECMWF source", src, "failed:", type(e).__name__, str(e)[:160])
            if DEADLINE and time.time() > DEADLINE - 30: raise OutOfTime("ECMWF out of time")
            # Not on ECMWF's own server yet: the cloud copies come later still, so go straight to the run before.
            if src == "ecmwf" and "404" in str(e): raise RuntimeError("this ECMWF run isn't published yet")
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
    # Each run is kept in a folder named for its date (UTC), for 30 days. ("today" only holds the current UTC day, so
    # after 7 PM Eastern it no longer has that day's runs.) The "today" folder is the fallback.
    hh = cyc.strftime("%H"); listing = None
    for base in (f"https://dd.weather.gc.ca/{cyc:%Y%m%d}/WXO-DD/model_gdps/15km/{hh}/", f"https://dd.weather.gc.ca/today/model_gdps/15km/{hh}/"):
        listing = get(base + "024/", tries=2, ok404=True)
        if listing and cyc.strftime("%Y%m%dT%HZ").encode() in listing: break
        listing = None
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
            try: s0, s1 = step_hours(m.startStep), step_hours(m.endStep)
            except Exception: s0 = s1 = step_hours(getattr(m, "step", 0))
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
        p = os.path.join(fdir, f)
        if not os.path.exists(p) and site and site_ok():
            blob = get(site + "/wp-content/plugins/will-it-snow/assets/fonts/" + f, tries=2, timeout=30, ok404=True)
            if blob and blob[:4] in (b"\x00\x01\x00\x00", b"true", b"OTTO"): open(p, "wb").write(blob)
        if os.path.exists(p): fm.fontManager.addfont(p); have += 1
    log(f"  fonts: {have} of 3 site fonts found" + ("" if have == 3 else " (upload the fonts folder to this repo so the job doesn't need the website for them)"))
    BODY, HEAD = ("Atkinson Hyperlegible", "Bricolage Grotesque") if have == 3 else ("DejaVu Sans", "DejaVu Sans")
    INK, SLATE, LINE, WATER, LAND = "#14233A", "#4A5A70", "#B9C6D3", "#DCE8F2", "#F4F6F8"

    # Where: around the big cities with 4+ inches in any model (or the snowiest cities when forced).
    big = [c for c in CITIES if c["rank"] < 200]
    hit = [c for c in big if max(totals[m][c["k"]] for m in totals) >= 4] or sorted(big, key=lambda c: -max(totals[m][c["k"]] for m in totals))[:6]
    la0, la1 = min(c["lat"] for c in hit) - 4, max(c["lat"] for c in hit) + 4
    lo0, lo1 = min(c["lon"] for c in hit) - 7, max(c["lon"] for c in hit) + 7
    if la1 - la0 < 12: m = (la0 + la1) / 2; la0, la1 = m - 6, m + 6
    if lo1 - lo0 < 22: m = (lo0 + lo1) / 2; lo0, lo1 = m - 11, m + 11
    clat, clon = (la0 + la1) / 2, (lo0 + lo1) / 2
    proj = f"+proj=lcc +lat_1={clat-6} +lat_2={clat+6} +lat_0={clat} +lon_0={clon} +units=km"
    tr = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
    xs, ys = tr.transform([lo0, lo1, lo0, lo1, clon, clon], [la0, la0, la1, la1, la0, la1])
    X0, X1, Y0, Y1 = min(xs), max(xs), min(ys), max(ys)
    # keep a 1.5 : 1 frame (panel shape)
    w, h = X1 - X0, Y1 - Y0
    if w / h > 1.5: d = (w / 1.5 - h) / 2; Y0, Y1 = Y0 - d, Y1 + d
    else: d = (h * 1.5 - w) / 2; X0, X1 = X0 - d, X1 + d

    ne = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
    def polys(name, keep=lambda p: True):
        data = json.loads(get(ne + name)); out = []
        for f in data["features"]:
            if not keep(f["properties"]): continue
            g = f["geometry"]; parts = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
            for part in parts:
                ring = np.array(part[0]); x, y = tr.transform(ring[:, 0], ring[:, 1])
                if x.max() < X0 - 500 or x.min() > X1 + 500 or y.max() < Y0 - 500 or y.min() > Y1 + 500: continue
                out.append(np.column_stack([x, y]))
        return out
    LANDP = polys("ne_50m_land.geojson")
    LAKES = polys("ne_50m_lakes.geojson", lambda p: p.get("scalerank", 9) <= 2)
    STATES = polys("ne_50m_admin_1_states_provinces_lakes.geojson", lambda p: p.get("adm0_a3") in ("USA", "CAN", "MEX"))

    LEVELS = [0.1, 1, 2, 3, 4, 6, 8, 12, 18, 24, 99]
    COLORS = ["#C6DBEF", "#9ECAE1", "#6BAED6", "#4292C6", "#2171B5", "#08519C", "#6A51A3", "#8E3FA8", "#C2378F", "#E7298A"]
    cmap = ListedColormap(COLORS); norm = BoundaryNorm(LEVELS, cmap.N)
    mx = lambda c: max(totals[m][c["k"]] for m in totals)
    inside = [c for c in big if la0 <= c["lat"] <= la1 and lo0 <= c["lon"] <= lo1]
    def visible(c):
        x, y = tr.transform(c["lon"], c["lat"])
        return X0 + (X1 - X0) * 0.03 < x < X1 - (X1 - X0) * 0.14 and Y0 + (Y1 - Y0) * 0.04 < y < Y1 - (Y1 - Y0) * 0.16
    def spaced(cands, n, km):
        out = []
        for c in cands:
            if all(math.hypot((c["lat"] - o["lat"]) * 111, (c["lon"] - o["lon"]) * 111 * math.cos(math.radians(c["lat"]))) > km for o in out): out.append(c)
            if len(out) >= n: break
        return out
    # Labels: the biggest cities in the snow first, then other big cities, spread out and away from the edges.
    labels = spaced(sorted([c for c in inside if c["rank"] < 120 and visible(c)], key=lambda c: (mx(c) < 1, c["rank"])), 9, 160)
    # Totals under the maps: three well-known cities with the most snow in any model.
    show = spaced(sorted([c for c in inside if c["rank"] < 80 and mx(c) >= 1], key=lambda c: -mx(c)), 3, 250)
    if len(show) < 3:  # a quiet test map: the snowiest well-known cities instead of an empty city
        more = spaced(sorted([c for c in inside if c["rank"] < 150 and c not in show], key=lambda c: -mx(c)), 3, 250)
        show += [c for c in more if all(math.hypot((c["lat"] - o["lat"]) * 111, (c["lon"] - o["lon"]) * 85) > 250 for o in show)][: 3 - len(show)]
    show = sorted(show, key=lambda c: c["lon"])
    labels = show + [c for c in labels if c not in show and all(math.hypot((c["lat"] - o["lat"]) * 111, (c["lon"] - o["lon"]) * 85) > 160 for o in show)][: 9 - len(show)]
    spreads = [max(totals[m][c["k"]] for m in totals) - min(totals[m][c["k"]] for m in totals) for c in show]
    headline = "Snowfall forecast: the models disagree" if spreads and max(spreads) >= 4 else "Snowfall forecast: the models agree"
    names = {"gfs": "American (GFS)", "ecmwf": "European (ECMWF)", "gem": "Canadian (GEM)"}
    order = [k for k in ("gfs", "ecmwf", "gem") if k in models]
    proj_grid = {k: tr.transform(models[k]["lon"], models[k]["lat"]) for k in order}
    et = dt.timezone(dt.timedelta(hours=-5))  # Eastern Standard Time label (approximate in summer)
    start = max(models[k]["run"] for k in order)
    def clock(f):
        t = (start + dt.timedelta(hours=f)).astimezone(et); hr = t.hour
        return t.strftime("%a ") + f"{(hr % 12) or 12} {'AM' if hr < 12 else 'PM'}"
    def total_at(k, c, f):
        i, j = nearest(models[k], c["lat"], c["lon"]); return float(models[k]["S"][f][i, j])

    frames = []
    for f in MAP_STEPS[1:]:
        fig = plt.figure(figsize=(12, 5.4), dpi=100, facecolor="white")
        hl = fig.text(0.025, 0.93, headline, fontfamily=HEAD, fontsize=21, color=INK, fontweight="heavy")
        fig.text(0.025, 0.885, f"Total snow since {clock(0)} ET", fontfamily=BODY, fontsize=12, color=SLATE)
        runs = " · ".join(f"{names[k].split(' ')[0]} {models[k]['run'].strftime('%HZ %b %-d')}" for k in order)
        mr = fig.text(0.975, 0.925, "Model runs: " + runs, fontfamily=BODY, fontsize=9.5, color=SLATE, ha="right")
        rd = fig.canvas.get_renderer()  # never let the headline run into the model runs (it's wider in the standard font)
        while hl.get_window_extent(rd).x1 > mr.get_window_extent(rd).x0 - 24 and hl.get_fontsize() > 14: hl.set_fontsize(hl.get_fontsize() - 1)
        fig.text(0.975, 0.872, clock(f) + " ET", fontfamily=HEAD, fontsize=16, fontweight="heavy", color=INK, ha="right")
        tb = fig.add_axes([0.66, 0.862, 0.17, 0.012]); tb.set_axis_off(); tb.set_xlim(0, 72); tb.set_ylim(0, 1)
        tb.add_patch(plt.Rectangle((0, 0), 72, 1, color="#E6EDF4")); tb.add_patch(plt.Rectangle((0, 0), f, 1, color="#2E5EAA"))
        width = 0.975 / len(order)
        for n, k in enumerate(order):
            ax = fig.add_axes([0.012 + n * width, 0.17, width - 0.015, 0.66]); ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_axis_off()
            ax.add_patch(plt.Rectangle((X0, Y0), X1 - X0, Y1 - Y0, color=WATER, zorder=0))
            for pg in LANDP: ax.add_patch(MPoly(pg, closed=True, facecolor=LAND, edgecolor="none", zorder=1))
            gx, gy = proj_grid[k]
            ax.contourf(gx, gy, np.ma.masked_less(models[k]["S"][f], 0.1), levels=LEVELS, cmap=cmap, norm=norm, zorder=2, alpha=0.95)
            for pg in LAKES: ax.add_patch(MPoly(pg, closed=True, facecolor=WATER, edgecolor="#9DB3C7", linewidth=0.4, zorder=3))
            for pg in STATES: ax.add_patch(MPoly(pg, closed=True, facecolor="none", edgecolor=LINE, linewidth=0.45, zorder=4))
            ax.add_patch(plt.Rectangle((X0, Y0), X1 - X0, Y1 - Y0, fill=False, edgecolor="#D3DEE9", linewidth=1, zorder=6))
            for c in labels:
                x, y = tr.transform(c["lon"], c["lat"]); ax.plot(x, y, "o", ms=3.2, color=INK, zorder=5)
                ax.text(x + (X1 - X0) * 0.012, y + (Y1 - Y0) * 0.015, c["name"], fontsize=8, fontfamily=BODY, color=INK, zorder=5, clip_on=True)
            ax.text(X0 + (X1 - X0) * 0.02, Y1 - (Y1 - Y0) * 0.08, names[k], fontfamily=HEAD, fontsize=13, fontweight="heavy", color=INK, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#D3DEE9"))
            tot = "   ".join(f"{c['name']}: {total_at(k, c, f):.0f} in" for c in show)
            fig.text(0.012 + n * width + (width - 0.015) / 2, 0.135, tot, fontfamily=BODY, fontsize=9.5, color=INK, ha="center")
        lx = fig.add_axes([0.25, 0.06, 0.5, 0.028]); lx.set_axis_off()
        for i, col in enumerate(COLORS): lx.add_patch(plt.Rectangle((i, 0), 1, 1, color=col))
        lx.set_xlim(0, len(COLORS)); lx.set_ylim(0, 1)
        for i, v in enumerate(LEVELS[:-1]):
            fig.text(0.25 + 0.5 * i / len(COLORS), 0.035, "0.1" if v == 0.1 else str(v), fontsize=8.5, fontfamily=BODY, color=SLATE, ha="center")
        fig.text(0.755, 0.064, "inches of snow (10:1)", fontsize=9, fontfamily=BODY, color=SLATE)
        fig.text(0.025, 0.02, "WeatherScoreboard.com", fontfamily=HEAD, fontsize=11, fontweight="heavy", color=INK)
        fig.text(0.975, 0.02, "Data: NOAA GFS · © ECMWF, CC BY 4.0 · Environment and Climate Change Canada · snow totals by Weather Scoreboard", fontfamily=BODY, fontsize=8.5, color=SLATE, ha="right")
        fig.canvas.draw(); im = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
        im.save(os.path.join(OUT, f"frame{len(frames):03d}.png")); frames.append(im); plt.close(fig)

    frames[-1].save(os.path.join(OUT, "map.png"))
    pal = frames[-1].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    q = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
    q[0].save(os.path.join(OUT, "map.gif"), save_all=True, append_images=q[1:], duration=[260] * (len(q) - 1) + [2600], loop=0, optimize=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "4", "-i", os.path.join(OUT, "frame%03d.png"),
                    "-vf", "tpad=stop_mode=clone:stop_duration=2.5,scale=1200:-2,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "20", "-movflags", "+faststart", os.path.join(OUT, "map.mp4")], check=True)
    return dict(headline=headline, region=[la0, la1, lo0, lo1], cities=[c["k"] for c in CITIES if max(totals[m][c["k"]] for m in totals) >= 1],
                show=[{"k": c["k"], "totals": {k: round(total_at(k, c, 72), 1) for k in order}} for c in show])

# ---------------------------------------------------------------- map grids for the website's interactive map
# One grid for the lower 48, 0.25° wide columns, rows evenly spaced in Web Mercator (so the browser map can stretch it
# straight onto its map), filled from each model by blending its four nearest grid points.
GRID = dict(w=-127.0, e=-65.0, s=23.0, n=51.0, cols=248, rows=144)
def grid_axes():
    my = lambda la: math.log(math.tan(math.pi / 4 + math.radians(la) / 2))
    y0, y1 = my(GRID["n"]), my(GRID["s"])
    ys = y0 + (np.arange(GRID["rows"]) + 0.5) * (y1 - y0) / GRID["rows"]
    lats = np.degrees(2 * np.arctan(np.exp(ys)) - math.pi / 2)
    lons = GRID["w"] + (np.arange(GRID["cols"]) + 0.5) * (GRID["e"] - GRID["w"]) / GRID["cols"]
    return lats, lons
def nearest_index(axis, want):
    order = np.argsort(axis); a = axis[order]
    i = np.clip(np.searchsorted(a, want), 1, len(a) - 1)
    i = np.where(np.abs(a[i - 1] - want) <= np.abs(a[i] - want), i - 1, i)
    return order[i]
def encode(md, smp, shape):
    """Each variable for every step as packed numbers: snow in tenths of an inch and rain in hundredths (uint16,
    65535 = no forecast), temperature in °F + 80 (uint8, 0 = no forecast)."""
    snow = np.full((len(STEPS),) + shape, 65535, dtype="<u2"); rain = snow.copy(); temp = np.zeros((len(STEPS),) + shape, dtype="u1")
    for n, f in enumerate(STEPS):
        for arr, k, sc in ((snow, "S", 10), (rain, "P", 100)):
            v = smp(md[k].get(f)); arr[n] = np.where(np.isnan(v), 65535, np.clip(np.round(v * sc), 0, 65534))
        v = smp(md["T"].get(f)); temp[n] = np.where(np.isnan(v), 0, np.clip(np.round(v) + 80, 1, 255))
    return {"snow": snow, "rain": rain, "temp": temp}

def export_grids(md, key):
    import gzip
    lats, lons = grid_axes(); tlon, tlat = np.meshgrid(lons, lats)
    out = {}
    for var, arr in encode(md, sampler(md, tlat, tlon), tlat.shape).items():
        p = os.path.join(OUT, f"{key}-{var}.bin"); open(p, "wb").write(gzip.compress(arr.tobytes(), 6)); out[f"{key}-{var}.bin"] = p
    return out

# High-detail tiles for the NWS and HRRR (about 2 miles), published with GitHub Pages: the lower 48 on a 0.03°
# grid, rows evenly spaced in Web Mercator, cut into 256 × 256 tiles; one file per source, variable and tile, holding
# every step. The map loads only the tiles in view once someone zooms in.
HI = dict(w=-125.0, e=-66.5, s=24.0, n=50.0, cols=1950, tile=256)
def hires_axes():
    my = lambda la: math.log(math.tan(math.pi / 4 + math.radians(la) / 2))
    y0, y1 = my(HI["n"]), my(HI["s"]); rows = int(round((y0 - y1) / math.radians((HI["e"] - HI["w"]) / HI["cols"])))
    ys = y0 + (np.arange(rows) + 0.5) * (y1 - y0) / rows
    lats = np.degrees(2 * np.arctan(np.exp(ys)) - math.pi / 2)
    lons = HI["w"] + (np.arange(HI["cols"]) + 0.5) * (HI["e"] - HI["w"]) / HI["cols"]
    return lats, lons

def export_hires(sources, stamp):
    import gzip
    pages = os.path.join(OUT, "pages"); os.makedirs(os.path.join(pages, "hires"), exist_ok=True)
    lats, lons = hires_axes(); rows, cols, T = len(lats), len(lons), HI["tile"]
    tlon, tlat = np.meshgrid(lons, lats); done = []
    for key, md in sources.items():
        try:
            data = encode(md, sampler(md, tlat, tlon), tlat.shape)
            for var, arr in data.items():
                for ty in range(0, rows, T):
                    for tx in range(0, cols, T):
                        part = np.ascontiguousarray(arr[:, ty:ty + T, tx:tx + T])
                        open(os.path.join(pages, "hires", f"{key}-{var}-{tx // T}-{ty // T}.bin"), "wb").write(gzip.compress(part.tobytes(), 6))
            done.append(key); del data
        except Exception:
            log(f"hires {key} failed:", traceback.format_exc()[-400:])
    info = dict(HI, rows=rows, stamp=stamp, steps=STEPS, sources=done, runs={k: sources[k]["run"].isoformat() for k in done},
                vars={"snow": {"type": "u16", "scale": 0.1}, "rain": {"type": "u16", "scale": 0.01}, "temp": {"type": "u8", "offset": -80}})
    json.dump(info, open(os.path.join(pages, "hires.json"), "w"))
    log(f"hires: {len(done)} sources, {rows} x {cols} cells, tiles in out/pages")
    return done

# ---------------------------------------------------------------- the homepage picture (Model Watch)
PREVIEW_SCALES = {
    "snow": ([0.1, 1, 2, 3, 4, 6, 8, 12, 18, 24, 999], ["#C6DBEF", "#9ECAE1", "#6BAED6", "#4292C6", "#2171B5", "#08519C", "#6A51A3", "#8E3FA8", "#C2378F", "#E7298A"], ["0.1", "1", "2", "3", "4", "6", "8", "12", "18", "24+"]),
    "rain": ([0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 999], ["#D9F0D3", "#ADDDA3", "#78C679", "#41AB5D", "#238443", "#1D91C0", "#225EA8", "#253494", "#7A3E9D", "#B0358B", "#E0218A"], ["0.01", "0.1", "¼", "½", "¾", "1", "1½", "2", "3", "4", "6+"]),
}
def preview(models, var, name="preview"):
    """One still picture for the homepage: one model's total snow (or rain) over the next 7 days across the lower 48,
    the same colors as the interactive map. Saved as out/<name>.jpg."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib import font_manager as fm
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Polygon as MPoly
    from pyproj import Transformer
    from PIL import Image
    pick = next(k for k in ("ecmwf", "gfs", "gem") if k in models); md = models[pick]; last = STEPS[-1]
    fdir = os.path.join(HERE, "fonts"); os.makedirs(fdir, exist_ok=True); body = "DejaVu Sans"; site = os.environ.get("SITE_URL", "").rstrip("/")
    for f in ("AtkinsonHyperlegible-Regular.ttf", "AtkinsonHyperlegible-Bold.ttf"):  # the site's font, as for the storm map
        path = os.path.join(fdir, f)
        if not os.path.exists(path) and site and site_ok():
            blob = get(site + "/wp-content/plugins/will-it-snow/assets/fonts/" + f, tries=2, timeout=30, ok404=True)
            if blob and blob[:4] in (b"\x00\x01\x00\x00", b"true", b"OTTO"): open(path, "wb").write(blob)
        if os.path.exists(path): fm.fontManager.addfont(path); body = "Atkinson Hyperlegible"
    tr = Transformer.from_crs("EPSG:4326", "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=38 +lon_0=-96 +units=km", always_xy=True)
    lats = np.linspace(20, 54, 340); lons = np.linspace(-130, -60, 700); tlon, tlat = np.meshgrid(lons, lats)
    vals = sampler(md, tlat, tlon)((md["S"] if var == "snow" else md["P"]).get(last))
    X, Y = tr.transform(tlon, tlat)
    levels, colors, labels = PREVIEW_SCALES[var]
    W, H = 1200, 680; X0, X1, Y0, Y1 = -2450, 2350, -1500, 1250
    if (X1 - X0) / (Y1 - Y0) > W / H: d = ((X1 - X0) * H / W - (Y1 - Y0)) / 2; Y0, Y1 = Y0 - d, Y1 + d
    else: d = ((Y1 - Y0) * W / H - (X1 - X0)) / 2; X0, X1 = X0 - d, X1 + d
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.axis("off")
    fig.patch.set_facecolor("#DCE8F2")
    ne = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
    def polys(name, keep=lambda p: True):
        raw = get(ne + name); out = []
        if not raw: return out
        for f in json.loads(raw)["features"]:
            if not keep(f["properties"]): continue
            g = f["geometry"]; parts = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
            for part in parts:
                ring = np.array(part[0]); x, y = tr.transform(ring[:, 0], ring[:, 1])
                if x.max() < X0 - 300 or x.min() > X1 + 300 or y.max() < Y0 - 300 or y.min() > Y1 + 300: continue
                out.append(np.column_stack([x, y]))
        return out
    for poly in polys("ne_50m_land.geojson"): ax.add_patch(MPoly(poly, closed=True, facecolor="#F4F6F8", edgecolor="none", zorder=1))
    masked = np.ma.masked_where(~np.isfinite(vals) | (vals < levels[0]), vals)
    if masked.count(): ax.contourf(X, Y, masked, levels=levels, cmap=ListedColormap(colors), norm=BoundaryNorm(levels, len(colors)), zorder=2, alpha=0.92)
    for poly in polys("ne_50m_lakes.geojson", lambda p: p.get("scalerank", 9) <= 2): ax.add_patch(MPoly(poly, closed=True, facecolor="#DCE8F2", edgecolor="#9FB0C2", linewidth=0.5, zorder=3))
    for poly in polys("ne_50m_admin_1_states_provinces_lakes.geojson", lambda p: p.get("adm0_a3") in ("USA", "CAN", "MEX")):
        ax.add_patch(MPoly(poly, closed=True, facecolor="none", edgecolor="#8A9AAE", linewidth=0.6, zorder=4))
    halo = [pe.withStroke(linewidth=3, foreground="white")]
    show = ("Seattle", "San Francisco", "Los Angeles", "Phoenix", "Salt Lake City", "Denver", "Dallas", "Houston", "Minneapolis", "Kansas City",
            "Chicago", "St. Louis", "Detroit", "Nashville", "Atlanta", "Washington", "New York", "Boston")
    seen = set()
    for c in sorted([c for c in CITIES if c["name"] in show], key=lambda c: c["rank"]):
        if c["name"] in seen: continue  # Kansas City, MO and KS: one label
        seen.add(c["name"])
        x, y = tr.transform(c["lon"], c["lat"])
        if not (X0 + 60 < x < X1 - 260 and Y0 + 120 < y < Y1 - 60): continue
        ax.plot(x, y, "o", ms=3.5, color="#14233A", zorder=6)
        ax.text(x + 22, y + 10, c["name"], fontsize=11, family=body, color="#14233A", zorder=6, path_effects=halo)
    # Legend, bottom right
    lx, ly, lw, lh = X1 - (X1 - X0) * 0.36, Y0 + (Y1 - Y0) * 0.05, (X1 - X0) * 0.33, (Y1 - Y0) * 0.022
    n = len(colors); ax.add_patch(plt.Rectangle((lx - 25, ly - lh * 2.6), lw + 50, lh * 5.2, facecolor="white", edgecolor="#D3DEE9", zorder=7))
    for i, col in enumerate(colors): ax.add_patch(plt.Rectangle((lx + lw * i / n, ly), lw / n, lh, facecolor=col, edgecolor="none", zorder=8))
    for i, lab in enumerate(labels):
        if i % 2 and n > 10: continue
        ax.text(lx + lw * i / n, ly - lh * 0.4, lab, fontsize=9, family=body, color="#33445B", va="top", ha="left", zorder=8)
    ax.text(lx + lw, ly + lh * 1.5, "inches", fontsize=9, family=body, color="#33445B", ha="right", va="bottom", zorder=8)
    png = os.path.join(OUT, name + ".png"); fig.savefig(png, dpi=100, facecolor=fig.get_facecolor()); plt.close(fig)
    jpg = os.path.join(OUT, name + ".jpg"); Image.open(png).convert("RGB").save(jpg, "JPEG", quality=84, optimize=True, progressive=True)
    log(f"{name}: {pick} {var}, {os.path.getsize(jpg) // 1024} KB")
    return dict(model=pick, var=var, hours=last)

# ---------------------------------------------------------------- main
def limited(seconds, fn, *a, **k):
    """Run fn in a child process and stop it after `seconds` (the ECMWF client downloads on its own, not through get())."""
    import multiprocessing as mp
    ctx = mp.get_context("fork"); q = ctx.Queue()
    def run():
        try: fn(*a, **k); q.put("")
        except Exception as e: q.put(f"{type(e).__name__}: {str(e)[:150]}")
    p = ctx.Process(target=run); p.start(); p.join(max(30, seconds))
    if p.is_alive():
        p.terminate(); p.join(5); raise TimeoutError(f"no complete download in {max(30, seconds):.0f}s")
    err = q.get() if not q.empty() else "the download stopped"
    if err: raise RuntimeError(err)

def left(cap):
    """Seconds this source may still use, at most `cap`."""
    return min(cap, (DEADLINE - time.time()) if DEADLINE else cap)

def budget(minutes):
    """Start a source's time: its own minutes, but never past the job's limit."""
    global DEADLINE
    DEADLINE = min(time.time() + minutes * 60, JOB_START + JOB_LIMIT)
    return DEADLINE > time.time() + 30

def main():
    global DEADLINE
    force = os.environ.get("FORCE_MAP", "false").lower() == "true"
    models, notes = {}, {}
    for key, fn in (("gfs", gfs), ("ecmwf", ecmwf), ("gem", gem)):
        if not budget(15):
            log(f"{key}: skipped, out of time"); continue
        for cyc in cycle_candidates(8 if key == "ecmwf" else 5):
            try:
                log(f"{key}: trying the {cyc:%Y-%m-%d %HZ} run"); t0 = time.time()
                models[key] = fn(cyc); notes[key] = models[key]["snow_note"]
                log(f"{key}: ok in {time.time() - t0:.0f}s"); break
            except OutOfTime:
                log(f"{key}: out of time after {time.time() - t0:.0f}s, skipped this time"); break
            except Exception as e:
                log(f"{key}: {type(e).__name__}: {str(e)[:200]}")
        else:
            log(f"{key}: skipped this time")
    DEADLINE = None
    if not models:
        log("No model could be downloaded."); upload({"ok": False, "log": LOG}, None, {}); sys.exit(1)

    start = max(m["run"] for m in models.values())
    globals_ = list(models)
    for key, fn in (("nws", nws), ("hrrr", hrrr)):  # the high-detail sources
        if not budget(8):
            log(f"{key}: skipped, out of time"); continue
        try:
            log(f"{key}: downloading"); t0 = time.time()
            models[key] = fn(start); notes[key] = models[key]["snow_note"]; log(f"{key}: ok in {time.time() - t0:.0f}s")
        except Exception as e:
            log(f"{key}: skipped this time: {type(e).__name__}: {str(e)[:200]}")
    DEADLINE = None
    points = {k: city_series(m) for k, m in models.items()}
    totals = {k: {c: v[I72][2] for c, v in points[k].items()} for k in globals_}   # snow over 72 hours, global models
    big = [c for c in CITIES if c["rank"] < 200]
    storm = any(max(totals[m][c["k"]] for m in totals) >= 4 for c in big)
    log("storm:", storm, "· snowiest big cities:", sorted(((round(max(totals[m][c['k']] for m in totals), 1), c["k"]) for c in big), reverse=True)[:5])
    meta = {"ok": True, "made": dt.datetime.now(dt.timezone.utc).isoformat(), "steps": STEPS, "notes": notes,
            "runs": {k: m["run"].isoformat() for k, m in models.items()}, "storm": storm, "log": LOG[-60:]}
    files = {}
    if (storm or force) and len(globals_) >= 2:
        try:
            meta["map"] = draw({k: models[k] for k in globals_}, totals, force); meta["map"]["forced"] = force and not storm
            files = {n: os.path.join(OUT, n) for n in ("map.mp4", "map.gif", "map.png")}
        except Exception:
            log("map failed:", traceback.format_exc()[-800:])
    elif force or storm:
        log("map: needs at least two models")
    log("map:", "drawn" if files else "not drawn", "(test map requested)" if force else "")
    # The homepage picture, every run: 7-day snow if any big city gets an inch in any global model, else rain.
    try:
        week = max((v[-1][2] or 0) for k in globals_ for c, v in points[k].items() if c in {x["k"] for x in big} and v and v[-1][2] is not None)
        auto = "snow" if week >= 1 else "rain"
        meta["preview"] = preview({k: models[k] for k in globals_}, auto)
        files["preview.jpg"] = os.path.join(OUT, "preview.jpg")
        # Both pictures as well, so the site can show rain on a snowy week (or snow on a rainy one) when asked to.
        import shutil
        shutil.copy(os.path.join(OUT, "preview.jpg"), os.path.join(OUT, f"preview_{auto}.jpg")); files[f"preview_{auto}.jpg"] = os.path.join(OUT, f"preview_{auto}.jpg")
        meta["preview"][auto] = meta["preview"]["model"]
        other = "rain" if auto == "snow" else "snow"
        meta["preview"][other] = preview({k: models[k] for k in globals_}, other, f"preview_{other}")["model"]
        files[f"preview_{other}.jpg"] = os.path.join(OUT, f"preview_{other}.jpg")
    except Exception:
        log("preview failed:", traceback.format_exc()[-600:])
    # Grids for the interactive map, sent first; the meta that follows switches the site over to them.
    try:
        stamp = start.strftime("%Y%m%d%H")
        sent = 0
        for key, md in models.items():
            for name, path in export_grids(md, key).items():
                GRIDS[name] = path; sent += upload_grid(stamp, name, path)
        meta["grid"] = dict(GRID, stamp=stamp, steps=STEPS, models=list(models), runs=meta["runs"],
                            vars={"snow": {"type": "u16", "scale": 0.1}, "rain": {"type": "u16", "scale": 0.01}, "temp": {"type": "u8", "offset": -80}})
        log(f"grids: {sent} files sent")
        hi = export_hires({k: models[k] for k in ("nws", "hrrr") if k in models}, stamp)
        repo = os.environ.get("GITHUB_REPOSITORY", "")
        if hi and "/" in repo: meta["grid"]["hires"] = f"https://{repo.split('/')[0].lower()}.github.io/{repo.split('/')[1]}/hires.json"
        if sent < 3 * len(models) and len(GRIDS) < 3 * len(models): meta.pop("grid")
    except Exception:
        log("grids failed:", traceback.format_exc()[-600:])
    os.makedirs(os.path.join(OUT, "pages"), exist_ok=True)  # GitHub Pages needs the folder, even if empty
    open(os.path.join(OUT, "pages", "index.html"), "w").write("<!doctype html><title>Weather Scoreboard model data</title><p>Map data for <a href=\"https://www.weatherscoreboard.com/models/\">WeatherScoreboard.com/models</a>.</p>"
        "<p>These files are published for WeatherScoreboard.com and aren't offered for reuse. Images © Weather Scoreboard. Underlying data: NOAA (public domain); "
        "ECMWF IFS forecast data © ECMWF, CC BY 4.0; contains information licenced under the Data Server End-use Licence of Environment and Climate Change Canada.</p>")
    meta["log"] = LOG[-60:]  # including the map and grid steps
    pts = {"runs": meta["runs"], "steps": STEPS, "cities": points}
    deliver(meta, pts, files)
    upload(meta, pts, files)

GRIDS = {}
def deliver(meta, points, files):
    """The same delivery on GitHub Pages (out/pages/<folder>), for the site to fetch itself when GitHub can't reach it:
    meta.json, points.json.gz, the grid files, the pictures, and latest.json listing them. Public data only; no key."""
    import gzip, hashlib, shutil
    # The folder's name comes from the site's key (WIS_MODELS_KEY), so only the plugin, which has the same key, knows
    # where to look; the key itself is never published.
    key = os.environ.get("WIS_MODELS_KEY", "")
    folder = "d-" + hashlib.sha256(("wis-delivery:" + key).encode()).hexdigest()[:24] if key else "delivery"
    d = os.path.join(OUT, "pages", folder); os.makedirs(d, exist_ok=True)
    listed = {"made": meta.get("made"), "stamp": (meta.get("grid") or {}).get("stamp"), "grids": [], "files": []}
    m = dict(meta); m["via"] = "pages"
    json.dump(m, open(os.path.join(d, "meta.json"), "w"))
    open(os.path.join(d, "points.json.gz"), "wb").write(gzip.compress(json.dumps(points, separators=(",", ":")).encode(), 6))
    if listed["stamp"]:
        for n, p in GRIDS.items(): shutil.copy(p, os.path.join(d, n)); listed["grids"].append(n)
    for n, p in (files or {}).items():
        if os.path.exists(p): shutil.copy(p, os.path.join(d, n)); listed["files"].append(n)
    json.dump(listed, open(os.path.join(d, "latest.json"), "w"))
    log(f"delivery on GitHub Pages: {len(listed['grids'])} grids, {len(listed['files'])} pictures")

def upload_grid(stamp, name, path):
    url, key = os.environ.get("SITE_URL", "").rstrip("/"), os.environ.get("WIS_MODELS_KEY", "")
    if not url or not key or not site_ok(): return 0
    for i in range(3):
        try:
            r = requests.post(url + "/wp-json/wis/v1/models/grid", headers={**UA, "X-WIS-Key": key}, data={"stamp": stamp},
                              files={"grid": (name, open(path, "rb"), "application/octet-stream")}, timeout=(15, 120))
            if r.status_code == 200: return 1
            log("  grid", name, r.status_code, r.text[:160])
        except Exception as e:
            log("  grid", name, type(e).__name__)
        time.sleep(5)
    return 0

def upload(meta, points, files):
    url, key = os.environ.get("SITE_URL", "").rstrip("/"), os.environ.get("WIS_MODELS_KEY", "")
    if not url or not key: log("SITE_URL or WIS_MODELS_KEY not set; nothing sent."); return
    if not site_ok(): log("upload: skipped (the site isn't reachable from GitHub); it will fetch this run from GitHub Pages"); return
    send = {"meta": ("meta.json", json.dumps(meta), "application/json")}
    if points:
        import gzip
        send["points"] = ("points.json.gz", gzip.compress(json.dumps(points, separators=(",", ":")).encode(), 6), "application/gzip")
    for n, p in (files or {}).items(): send[n.replace(".", "_")] = (n, open(p, "rb"), "application/octet-stream")
    try:
        r = requests.post(url + "/wp-json/wis/v1/models", headers={**UA, "X-WIS-Key": key}, files=send, timeout=(15, 180))
        log("upload:", r.status_code, r.text[:300])
    except Exception as e:
        log("upload: failed,", type(e).__name__, "; the site will fetch this run from GitHub Pages")

if __name__ == "__main__":
    main()
