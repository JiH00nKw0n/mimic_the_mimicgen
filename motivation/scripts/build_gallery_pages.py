"""Build the GitHub Pages case-study gallery: ONE dated HTML for all tasks.
Videos are NOT embedded — they are referenced by absolute URLs under BASE
(mp4 files live in the same Pages repo) and lazy-loaded with an
IntersectionObserver so ~800 clips stay scrollable.
Usage: python3 build_gallery_pages.py <renders_dir> <out_html> <base_url>
"""
import json, sys
from pathlib import Path

RD = Path(sys.argv[1]); OUT = Path(sys.argv[2]); BASE = sys.argv[3].rstrip("/")
TASKS = ["stack", "square", "coffee", "threading", "three_piece_assembly",
         "stack_three", "hammer_cleanup", "mug_cleanup"]
NAME = {"stack": "Stack", "square": "Square", "coffee": "Coffee", "threading": "Threading",
        "three_piece_assembly": "Three Piece Assembly", "stack_three": "Stack Three",
        "hammer_cleanup": "Hammer Cleanup", "mug_cleanup": "Mug Cleanup"}

CSS = """
<style>
:root{--bg:#0d1015;--surface:#151a21;--surface2:#1b222b;--border:#28313d;--ink:#e7edf4;
--muted:#8a97a7;--faint:#5c6675;--accent:#38c5d6;--ok:#3fb84f;--ok-dim:rgba(63,184,79,.16);
--fail:#f2603f;--fail-dim:rgba(242,96,63,.16);--high:#2fd6c0;--low:#f0a63a;
--mono:ui-monospace,"SF Mono",Menlo,monospace;--sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
@media (prefers-color-scheme: light){:root{--bg:#eef1f5;--surface:#fff;--surface2:#f4f6f9;
--border:#dbe1e9;--ink:#18202a;--muted:#5a6573;--faint:#8b95a2;--accent:#0f9bad;--ok:#1f9d38;
--ok-dim:rgba(31,157,56,.12);--fail:#d64425;--fail-dim:rgba(214,68,37,.12);--high:#0e9d8c;--low:#c07d15;}}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.55}
.wrap{max-width:1180px;margin:0 auto;padding:clamp(24px,4vw,56px) clamp(16px,3vw,36px) 90px}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin-bottom:10px}
h1{font-size:clamp(24px,4vw,38px);line-height:1.12;letter-spacing:-.02em;margin:0 0 14px;text-wrap:balance;font-weight:680}
h2{font-size:22px;margin:56px 0 4px;font-weight:670;letter-spacing:-.01em}
h2 small{font-family:var(--mono);font-size:12.5px;color:var(--faint);font-weight:400;margin-left:8px}
.lede{color:var(--muted);max-width:72ch;margin:0 0 6px}.lede b{color:var(--ink)}
.toc{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 6px}
.toc a{font-family:var(--mono);font-size:12.5px;padding:5px 11px;border-radius:8px;background:var(--surface2);
border:1px solid var(--border);color:var(--ink);text-decoration:none}
.toc a:hover{border-color:var(--accent)}
table{border-collapse:collapse;font-size:13.5px;margin:20px 0;width:100%}
th,td{border-bottom:1px solid var(--border);padding:7px 10px;text-align:left;white-space:nowrap}
th{font-family:var(--mono);font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--faint)}
td.num{font-family:var(--mono);font-variant-numeric:tabular-nums}
.tblwrap{overflow-x:auto}
.bandbox{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:14px;margin:14px 0}
.bandhead{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:2px}
.bandhead h3{margin:0;font-size:15.5px;font-weight:640}
.dgr{font-family:var(--mono);font-size:12.5px;color:var(--faint)}
.rowlab{font-family:var(--mono);font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--faint);margin:12px 2px 6px}
.strip{display:flex;gap:10px;overflow-x:auto;padding:2px 2px 8px}
.cell{flex:0 0 auto;width:160px;margin:0}
.cell video{width:160px;height:160px;object-fit:cover;border-radius:9px;border:1px solid var(--border);background:var(--surface2);display:block}
.cap{font-family:var(--mono);font-size:10.5px;color:var(--faint);margin-top:3px;display:flex;gap:6px;align-items:center}
.pill{padding:1px 7px;border-radius:999px;font-size:10px;font-weight:600}
.pill.ok{background:var(--ok-dim);color:var(--ok)}.pill.fail{background:var(--fail-dim);color:var(--fail)}
a{color:var(--accent)}
footer{margin-top:56px;padding-top:20px;border-top:1px solid var(--border);font-family:var(--mono);font-size:12px;color:var(--faint);max-width:78ch}
@media (prefers-reduced-motion: reduce){.cell video{content-visibility:auto}}
</style>"""

JS = """
<script>
const io = new IntersectionObserver((es) => {
  for (const e of es) {
    const v = e.target;
    if (e.isIntersecting) {
      if (!v.src) { v.src = v.dataset.src; }
      if (!matchMedia('(prefers-reduced-motion: reduce)').matches) v.play().catch(() => {});
    } else {
      v.pause();
      if (e.intersectionRatio === 0 && v.src) { v.removeAttribute('src'); v.load(); }
    }
  }
}, { rootMargin: '600px 0px' });
document.querySelectorAll('video[data-src]').forEach(v => io.observe(v));
</script>"""


def vid(url, cap):
    return (f'<figure class="cell"><video data-src="{url}" muted loop playsinline '
            f'preload="none" controls></video><div class="cap">{cap}</div></figure>')


def cap(c):
    p = '<span class="pill ok">성공</span>' if c["success"] else '<span class="pill fail">실패</span>'
    d = f'<span>d={c["d_pos"]:.2f}</span>' if c.get("d_pos") is not None else ""
    return p + d


def strip(clips, task):
    if not clips:
        return '<div class="dgr">해당 사례 없음</div>'
    cells = "".join(vid(f"{BASE}/{task}/{c['file']}", cap(c)) for c in clips)
    return f'<div class="strip">{cells}</div>'


sections, rows, total = [], [], 0
for task in TASKS:
    mp = RD / task / "manifest.json"
    if not mp.exists():
        continue
    man = json.loads(mp.read_text())
    clips = man["clips"]; total += len(clips)
    n0, ring = man["bands"]["n0"], man["bands"]["ring"]
    hi, lo = man["hi_src"], man["lo_src"]
    n0_cell = f'{n0["dgr"]:.0%}†' if n0.get("mode") == "inner" else f'{n0["dgr"]:.0%}'
    rows.append(f'<tr><td><a href="#{task}">{NAME[task]}</a></td>'
                f'<td class="num">{n0_cell}</td><td class="num">{ring["dgr"]:.0%}</td>'
                f'<td class="num">{ring["n2_dgr"]:.0%}</td>'
                f'<td class="num">s{hi["id"]} · {hi["dgr"]:.0%}</td>'
                f'<td class="num">s{lo["id"]} · {lo["dgr"]:.0%}</td>'
                f'<td class="num">{len(clips)}</td></tr>')

    def reg(band, succ):
        return [c for c in clips if c["axis"] == "region" and c["band"] == band and c["success"] == succ]

    def src(role, succ, band):
        return [c for c in clips if c["axis"] == "source" and c["role"] == role
                and c["success"] == succ and c.get("band") == band]

    if n0.get("mode") == "inner":
        first_band = ("n0", n0, "N1 내부 — 구영역에서 생성",
                      "확장 전 영역(N1 상자) 안에 모든 물체가 놓인 시도. 이 태스크는 최협소 N0 상자에 "
                      "물체를 겹침 없이 놓을 수 없어(3물체) N0 풀이 없다")
    else:
        first_band = ("n0", n0, "N0 — 최협소 영역에서 생성", "원본 데모 배치와 거의 같은 자리")
    region_html = ""
    for band, meta, title, desc in [
            first_band,
            ("ring", ring, "N2∖N1 — 신규 확장 영역에서 생성", "N1 상자 밖, 확장으로 새로 열린 링")]:
        region_html += f'''<div class="bandbox"><div class="bandhead"><h3>{title}</h3>
<span class="dgr">생성 성공률 {meta["dgr"]:.0%} · 시도 {meta["n"]:,}</span><span class="dgr">{desc}</span></div>
<div class="rowlab">성공 사례</div>{strip(reg(band, True), task)}
<div class="rowlab">실패 사례</div>{strip(reg(band, False), task)}</div>'''

    source_html = ""
    for role, s, color in [("hi", hi, "var(--high)"), ("lo", lo, "var(--low)")]:
        nm = "많이 쓰이는 소스 (고DGR)" if role == "hi" else "드물게 남는 소스 (저DGR)"
        source_html += f'''<div class="bandbox"><div class="bandhead"><h3 style="color:{color}">{nm}</h3>
<span class="dgr">source s{s["id"]} · DGR {s["dgr"]:.0%}</span></div>
<div class="rowlab">성공 · 가까운 변형</div>{strip(src(role, True, "near"), task)}
<div class="rowlab">성공 · 먼 변형</div>{strip(src(role, True, "far"), task)}
<div class="rowlab">실패 · 먼 변형</div>{strip(src(role, False, "far"), task)}</div>'''

    sections.append(f'''<section id="{task}"><h2>{NAME[task]}
<small>N0 {n0["dgr"]:.0%} → 링 {ring["dgr"]:.0%} · 소스 s{hi["id"]} {hi["dgr"]:.0%} vs s{lo["id"]} {lo["dgr"]:.0%}</small></h2>
{region_html}{source_html}</section>''')

toc = "".join(f'<a href="#{t}">{NAME[t]}</a>' for t in TASKS if (RD / t / "manifest.json").exists())
page = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>합성 데모 케이스 스터디 v2 — 영역별·소스별</title>{CSS}</head><body>
<div class="wrap">
<div class="eyebrow">MimicGen · 합성 데이터 감사 · 케이스 스터디 v2</div>
<h1>어디서 생성했나, 어느 원본에서 왔나 — 합성 데모를 눈으로 보기</h1>
<p class="lede">MimicGen은 사람 데모 10개를 새 물체 배치로 <b>변형</b>해 합성 데모를 만들고, 시뮬레이터에서
성공한 것만 남긴다. 이 페이지는 그 생성 과정을 두 축으로 자른 실제 클립 모음이다.</p>
<p class="lede"><b>축 1 — 생성 영역.</b> 원본 배치와 거의 같은 자리(N0)에서 만든 시도와, 영역을 넓혀 새로
열린 링(N2∖N1)에서 만든 시도. 링으로 갈수록 생성 성공률이 떨어진다 — 아래 표의 두 번째·세 번째 열.</p>
<p class="lede"><b>축 2 — 소스 데모.</b> 같은 풀 안에서 생성이 잘 되는 원본(고DGR)과 거의 실패하는
원본(저DGR). 같은 거리에서도 소스에 따라 움직임의 질이 완전히 다르다.</p>
<p class="lede">각 줄 최대 10클립, 변형 거리(d)를 고르게 펼쳐 뽑았다. 스크롤하면 영상이 자동 로드된다.</p>
<div class="toc">{toc}</div>
<div class="tblwrap"><table><thead><tr><th>task</th><th>N0 생성 성공률</th><th>링 N2∖N1</th><th>N2 전체</th>
<th>고DGR 소스</th><th>저DGR 소스</th><th>클립</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
{"".join(sections)}
<footer>클립 총 {total:,}개 · 실제 생성 풀의 시도를 states 재생(agentview 160px)으로 렌더 · d = 정규화 변형 거리(0=원본 자리) ·
링 판정 = N1 상자(풀 실측 또는 레지스트리) 밖에 물체가 놓인 N2 시도 · † 표시 태스크는 최협소 N0 상자에 3개 물체를
겹침 없이 놓을 수 없어 N0 풀 대신 "N1 내부 생성" 시도를 보여줌 · 정량 결과는 저장소 mimic_the_mimicgen의
Motivation_All_Experiments_Summary.md 참고</footer></div>{JS}</body></html>'''

OUT.write_text(page)
print(f"wrote {OUT} ({OUT.stat().st_size/1e3:.0f}KB, {total} clips referenced)")
