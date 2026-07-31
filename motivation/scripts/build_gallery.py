import base64, json, re
from pathlib import Path
SP = Path("/private/tmp/claude-501/-Users-junekwon-Desktop-Projects-robot-data/26f969a1-1b2a-4425-9a87-ed38d0b97b67/scratchpad")
CR = SP / "case_renders"
TASK_LABEL = {"stack":"Stack","threading":"Threading","square":"Square","stack_three":"Stack Three",
    "coffee":"Coffee","three_piece_assembly":"Three-Piece Assembly","hammer_cleanup":"Hammer Cleanup","mug_cleanup":"Mug Cleanup"}
ORDER = ["stack","threading","square","stack_three","coffee","three_piece_assembly","hammer_cleanup","mug_cleanup"]

def parse(fn):
    m = re.match(r"(hi|lo)_s(\d+)_DGR(\d+)_(OK|FAIL)_(near|far)_d([\d.]+)\.mp4", fn)
    role, src, dgr, ok, band, d = m.groups()
    return dict(role=role, src=int(src), dgr=int(dgr), ok=(ok=="OK"), band=band, d=float(d))

def datauri(p):
    return "data:video/mp4;base64," + base64.b64encode(p.read_bytes()).decode()

tasks = {}
for d in CR.iterdir():
    if not d.is_dir(): continue
    clips = []
    for f in d.glob("*.mp4"):
        c = parse(f.name); c["uri"] = datauri(f); clips.append(c)
    if clips: tasks[d.name] = clips

def cell(clips, role, ok, band):
    m = [c for c in clips if c["role"]==role and c["ok"]==ok and c["band"]==band]
    if not m:
        return '<div class="cell empty">—</div>'
    c = m[0]
    pill = 'ok' if c["ok"] else 'fail'
    txt = '성공' if c["ok"] else '실패'
    return (f'<figure class="cell">'
            f'<video src="{c["uri"]}" autoplay loop muted playsinline></video>'
            f'<figcaption><span class="pill {pill}">{txt}</span>'
            f'<span class="dpos">d={c["d"]:.2f}</span></figcaption></figure>')

def dgr_tier(v): return "high" if v>=50 else ("mid" if v>=25 else "low")

sections = []
for t in ORDER:
    if t not in tasks: continue
    clips = tasks[t]
    hi = next((c for c in clips if c["role"]=="hi"), None)
    lo = next((c for c in clips if c["role"]=="lo"), None)
    rows = []
    for role, meta in [("hi", hi), ("lo", lo)]:
        if meta is None:
            rows.append('<div class="srcrow missing"><div class="srchead">— (저DGR 소스 렌더 불가)</div></div>'); continue
        v = meta["dgr"]; tier = dgr_tier(v)
        role_txt = "많이 쓰인 소스" if role=="hi" else "적게 쓰인 소스"
        rows.append(
            f'<div class="srcrow">'
            f'<div class="srchead">'
            f'<span class="rolelab">{role_txt}</span>'
            f'<span class="srcid">source #{meta["src"]}</span>'
            f'<span class="dgr {tier}"><b>{v}%</b><small>DGR</small></span>'
            f'</div>'
            f'<div class="cells">'
            f'{cell(clips, role, True, "near")}{cell(clips, role, True, "far")}{cell(clips, role, False, "far")}'
            f'</div></div>')
    contrast = f'{hi["dgr"]}%' + (f' vs {lo["dgr"]}%' if lo else '')
    sections.append(
        f'<section class="task">'
        f'<div class="taskhead"><h2>{TASK_LABEL.get(t,t)}</h2>'
        f'<span class="contrast">생성 성공률 {contrast}</span></div>'
        f'<div class="colhead"><span>성공 · 가까운 변형</span><span>성공 · 먼 변형</span><span>실패 · 먼 변형</span></div>'
        + "".join(rows) + '</section>')

HTML = open(SP/"gallery_template.html").read().replace("{{SECTIONS}}", "\n".join(sections))
(SP/"gallery.html").write_text(HTML)
print(f"tasks={len(sections)} clips={sum(len(v) for v in tasks.values())} size={len(HTML)//1024}KB -> gallery.html")
