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
               f"&file=gfs.t{hh}z.pgrb2.0p25.f{f
