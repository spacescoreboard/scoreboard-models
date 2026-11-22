"""Weather Scoreboard model job.

Twice a day (see .github/workflows/models.yml) this script:
  1. downloads the next 72 hours of three forecast models: the American GFS (NOAA), the European IFS (ECMWF open
     data) and the Canadian GDPS/GEM (Environment and Climate Change Canada);
  2. reads each model's forecast for every Weather Scoreboard city, every 3 hours (temperature, rain and melted
     snow, snow), which the website keeps to grade the models later;
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
STEPS = list(range(0, 73, 3))            # forecast hours used
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
        inc = P[f] - P[f - 3] if f - 3 in P else P[f]
        frac = snowfrac if snowfrac is not None else (T[f] <= 33).astype("float32")
        S[f] = (S.get(f - 3, 0) + np.clip(inc, 0, None) * frac / 25.4 * 10)
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
        for f in STEPS[1:]: S[f] = S[f - 3] + np.clip(P[f] - P[f - 3], 0, None) * (T[f] <= 33) * 10
    if 0 not in P: P[0] = np.zeros_like(next(iter(P.values()))); S[0] = np.zeros_like(P[0])
    return dict(lat=lat, lon=lon, T=T, P=P, S=S, run=cyc, snow_note="ECMWF snowfall" if snow_native else "ECMWF snow from temperature")

# ---------------------------------------------------------------- Canadian GDPS / GEM (Open Government Licence – Canada)
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
    T, P = {}, {}; lat = lon = None
    for f in STEPS:
        if f == 0: continue
        for name, store in ((tname, T), (pname, P)):
            fn = re.sub(r"_PT\d{3}H", f"_PT{f:03d}H", name)
            blob = get(base + f"{f:03d}/" + fn)
            if not blob: raise RuntimeError(f"GEM {fn} missing")
            m = grib_fields(blob)[0]
            lat, lon, v = subset(m["lats"], m["lons"], m["values"])
            store[f] = (v - 273.15) * 9 / 5 + 32 if store is T else v / 25.4   # K → °F; kg/m² (mm) → in, accumulated
    T[0] = T[3]; P[0] = np.zeros_like(P[3]); S = {0: np.zeros_like(P[3])}
    for f in STEPS[1:]: S[f] = S[f - 3] + np.clip(P[f] - P[f - 3], 0, None) * (T[f] <= 33) * 10
    return dict(lat=lat, lon=lon, T=T, P=P, S=S, run=cyc, snow_note="GEM snow from temperature")

# ---------------------------------------------------------------- city forecasts
def nearest(md, la, lo):
    d = (md["lat"] - la) ** 2 + ((md["lon"] - lo) * math.cos(math.radians(la))) ** 2
    return np.unravel_index(np.argmin(d), d.shape)

def city_series(md):
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
    for f in ("AtkinsonHyperlegible-Regular.ttf", "AtkinsonHyperlegible-Bold.ttf", "BricolageGrotesque-ExtraBold.ttf"):
        fm.fontManager.addfont(os.path.join(HERE, "fonts", f))
    BODY, HEAD = "Atkinson Hyperlegible", "Bricolage Grotesque"
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
    if len(show) < 3: show += [c for c in labels if c not in show][: 3 - len(show)]
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
    for f in STEPS[1:]:
        fig = plt.figure(figsize=(12, 5.4), dpi=100, facecolor="white")
        fig.text(0.025, 0.93, headline, fontfamily=HEAD, fontsize=21, color=INK, fontweight="heavy")
        fig.text(0.025, 0.885, f"Total snow since {clock(0)} ET", fontfamily=BODY, fontsize=12, color=SLATE)
        runs = " · ".join(f"{names[k].split(' ')[0]} {models[k]['run'].strftime('%HZ %b %-d')}" for k in order)
        fig.text(0.975, 0.925, "Model runs: " + runs, fontfamily=BODY, fontsize=9.5, color=SLATE, ha="right")
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
        fig.text(0.975, 0.02, "Data: NOAA GFS · ECMWF open data (CC BY 4.0) · Environment and Climate Change Canada", fontfamily=BODY, fontsize=8.5, color=SLATE, ha="right")
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

# ---------------------------------------------------------------- main
def main():
    force = os.environ.get("FORCE_MAP", "false").lower() == "true"
    models, notes = {}, {}
    for key, fn in (("gfs", gfs), ("ecmwf", ecmwf), ("gem", gem)):
        for cyc in cycle_candidates():
            try:
                log(f"{key}: trying the {cyc:%Y-%m-%d %HZ} run"); t0 = time.time()
                models[key] = fn(cyc); notes[key] = models[key]["snow_note"]
                log(f"{key}: ok in {time.time() - t0:.0f}s"); break
            except Exception as e:
                log(f"{key}: {type(e).__name__}: {str(e)[:200]}")
        else:
            log(f"{key}: skipped this time")
    if not models:
        log("No model could be downloaded."); upload({"ok": False, "log": LOG}, None, {}); sys.exit(1)

    points = {k: city_series(m) for k, m in models.items()}
    totals = {k: {c: v[-1][2] for c, v in p.items()} for k, p in points.items()}
    big = [c for c in CITIES if c["rank"] < 200]
    storm = any(max(totals[m][c["k"]] for m in totals) >= 4 for c in big)
    log("storm:", storm, "· snowiest big cities:", sorted(((round(max(totals[m][c['k']] for m in totals), 1), c["k"]) for c in big), reverse=True)[:5])
    meta = {"ok": True, "made": dt.datetime.now(dt.timezone.utc).isoformat(), "steps": STEPS, "notes": notes,
            "runs": {k: m["run"].isoformat() for k, m in models.items()}, "storm": storm, "log": LOG[-60:]}
    files = {}
    if (storm or force) and len(models) >= 2:
        try:
            meta["map"] = draw(models, totals, force); meta["map"]["forced"] = force and not storm
            files = {n: os.path.join(OUT, n) for n in ("map.mp4", "map.gif", "map.png")}
        except Exception:
            log("map failed:", traceback.format_exc()[-800:])
    upload(meta, {"runs": meta["runs"], "steps": STEPS, "cities": points}, files)

def upload(meta, points, files):
    url, key = os.environ.get("SITE_URL", "").rstrip("/"), os.environ.get("WIS_MODELS_KEY", "")
    if not url or not key: log("SITE_URL or WIS_MODELS_KEY not set; nothing sent."); return
    send = {"meta": ("meta.json", json.dumps(meta), "application/json")}
    if points: send["points"] = ("points.json", json.dumps(points, separators=(",", ":")), "application/json")
    for n, p in (files or {}).items(): send[n.replace(".", "_")] = (n, open(p, "rb"), "application/octet-stream")
    r = requests.post(url + "/wp-json/wis/v1/models", headers={**UA, "X-WIS-Key": key}, files=send, timeout=180)
    log("upload:", r.status_code, r.text[:300])
    if r.status_code != 200: sys.exit(1)

if __name__ == "__main__":
    main()
