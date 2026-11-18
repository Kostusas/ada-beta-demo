"""Small HTML/SVG views rendered by the Jupyter widget frontend."""
from html import escape

STYLE = """
<style>
.adaPanel{background:#101f2d;color:#edf6f5;padding:20px 24px;border-radius:10px;font:14px/1.5 system-ui,sans-serif}
.adaPanel h2,.adaPanel h3{color:#f4f7fa;font-family:system-ui,sans-serif;margin:0 0 5px;font-weight:650}
.adaPanel h2{font-size:29px;letter-spacing:-.035em}.adaPanel h3{font-size:17px}
.adaMuted,.adaLabel{color:#afc3d0}.adaLabel{font-size:12px}.adaGlow{color:#5ce1c0}
.adaStats{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}
.adaCard{flex:1;min-width:115px;background:#1a3041;border-radius:7px;padding:10px 14px}
.adaValue{color:#f7fafb;font-size:23px;font-variant-numeric:tabular-nums;font-weight:650}
.adaColumns{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}
.adaThread{display:flex;align-items:center;gap:9px;font:12px/1.7 ui-monospace,monospace}
.adaTrack{display:block;flex:1;background:#294151;height:7px;border-radius:4px}
.adaTrack i{display:block;background:#5ce1c0;height:100%;border-radius:4px}
.adaPanel table{border-collapse:collapse;width:100%;color:inherit;font-size:13px}
.adaPanel td,.adaPanel th{padding:6px;text-align:right;border-bottom:1px solid #334957;background:#101f2d;color:inherit}
.adaPanel td:first-child,.adaPanel th:first-child{text-align:left}
</style>
"""


def panel(body):
    return f'{STYLE}<div class="adaPanel">{body}</div>'


def cards(items):
    return '<div class="adaStats">' + ''.join(
        f'<div class="adaCard"><div class="adaLabel">{label}</div><div class="adaValue">{value}</div></div>'
        for label, value in items) + '</div>'


def ram(size):
    return f"{size/2**30:.2f} GiB" if size >= 2**30 else f"{size/2**20:.1f} MiB"


def trace(samples, field, maximum, color):
    if len(samples) < 2:
        return '<div class="adaMuted">Collecting samples…</div>'
    w, h = 480, 65
    start = samples[0][0]
    duration = max(samples[-1][0] - start, .001)
    pts = " ".join(f"{w*(t-start)/duration:.1f},{h*(1-min(1,v[field]/maximum)):.1f}" for t,v in samples)
    return (f'<svg role="img" aria-label="{field} over {duration:.1f} seconds" width="100%" height="65" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
            f'<path d="M0 32H480M0 64H480" stroke="#354956"/><polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/></svg>'
            f'<span class="adaLabel">Past {duration:.1f} s</span>')


def telemetry(event, history, capacity):
    info = f'Configuration {event["index"]} of {event["count"]} · {event["n"]} loci · {event["threads"]} Julia compute threads'
    s = event["sample"]
    body = f'<h3>{escape(event["phase"])}</h3><div class="adaMuted">{info} · Julia PID {event["pid"]}</div>'
    if s is None:
        return panel(body + '<p>Reading worker counters…</p>')
    body += cards([("Busy CPU equivalents", f'{s["cores"]:.1f}'), ("Session CPU use", f'{100*s["cores"]/capacity:.0f}%'),
                   ("Resident memory", ram(s["rss"])), ("Finished predictions", str(event["passes"]))])
    body += '<div class="adaColumns"><div><b>Worker CPU</b>'
    body += trace(history, "cores", max(capacity, max(v["cores"] for _,v in history)), "#5ce1c0")
    body += f'<br><b>Worker RSS</b> <span class="adaLabel">· scale to {ram(max(v["rss"] for _,v in history))}</span>'
    body += trace(history, "rss", max(1,max(v["rss"] for _,v in history)), "#ffa873") + '</div><div>'
    if s["os_threads"] is None:
        return panel(body + '<b>OS threads</b><p>Per-thread counters are unavailable in this environment.</p></div></div>')
    body += f'<b>Busiest OS threads</b> <span class="adaLabel">· {s["os_threads"]} total</span>'
    for tid, cores in s["busiest"][:8]:
        cpu = s.get("last_cpu", {}).get(str(tid))
        location = "" if cpu is None else f" · CPU {cpu}"
        body += (f'<div class="adaThread"><span>TID {tid}{location}</span><span class="adaTrack">'
                 f'<i style="width:{min(100,100*cores):.1f}%"></i></span><span>{100*cores:.0f}%</span></div>')
    return panel(body + '</div></div>')


def comparison(run):
    rows = run.get("rows", [])
    if not rows:
        return panel('<h3>Scaling results</h3><p>Finish a configuration to see its measured throughput and speedup.</p>')
    body = f'<h3>Measured scaling</h3><div class="adaMuted">{escape(run["node"])} · {escape(rows[-1]["cpu"])} · Julia {escape(rows[-1]["julia"])}</div>'
    for n in dict.fromkeys(r["n"] for r in rows):
        group = [r for r in rows if r["n"] == n]
        best = max(n**3/r["median_s"] for r in group)
        body += f'<h3 style="margin-top:15px">{n:,} loci · {n**3:,} entries</h3>'
        body += '<table><tr><th>Threads</th><th>G entries/s</th><th>Median/pass</th><th>Speedup</th><th>Efficiency</th><th>Busy CPUs</th><th>Peak RSS</th></tr>'
        for r in group:
            rate = n**3/r["median_s"]
            speed = f'{r["speedup"]:.2f}×' if "speedup" in r else "—"
            efficiency = f'{r["efficiency"]:.0%}' if "efficiency" in r else "—"
            body += (f'<tr><td>{r["threads"]}</td><td>{rate/1e9:.3f}<span class="adaTrack"><i style="width:{100*rate/best:.1f}%"></i></span></td>'
                     f'<td>{r["median_s"]*1000:.2f} ms</td><td>{speed}</td><td>{efficiency}</td>'
                     f'<td>{r["busy_cores"]:.2f}</td><td>{ram(r["peak_rss"])}</td></tr>')
        body += '</table>'
    return panel(body)
