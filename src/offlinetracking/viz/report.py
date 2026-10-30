"""人工审查报告：自包含静态 HTML（图片 base64 内嵌），可直接发给复核人员。

内容：总体指标 → 每段序列的俯视图总览 → 复核队列（按歧义度从高到低）→ 自动连接抽查表。
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

import cv2
import numpy as np

from .bev import edge_figure, fig_to_png, overview_figure, reliability_figure

CSS = """
:root{--bg:#fafafa;--card:#fff;--ink:#1f2328;--muted:#656d76;--line:#d0d7de;--auto:#1a9850;--review:#f28e2b;--i:#2166ac;--j:#b2182b}
@media (prefers-color-scheme: dark){:root{--bg:#0d1117;--card:#161b22;--ink:#e6edf3;--muted:#8d96a0;--line:#30363d}
 img.plot{background:#fff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
 font:14px/1.55 -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 12px;padding-top:8px;border-top:1px solid var(--line)}
h3{font-size:15px;margin:0 0 8px}.muted{color:var(--muted)}
.kpis{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin:16px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.kpi b{display:block;font-size:20px}.kpi span{color:var(--muted);font-size:12px}
img.plot{max-width:100%;border:1px solid var(--line);border-radius:6px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px;margin:12px 0}
.card .row{display:flex;flex-wrap:wrap;gap:14px}.card .row>div{flex:1 1 320px;min-width:0}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left}
th{color:var(--muted);font-weight:600}.tag{display:inline-block;border-radius:10px;padding:0 8px;font-size:12px;margin-right:4px;
 border:1px solid var(--line)}.tag.auto{color:var(--auto);border-color:var(--auto)}.tag.review{color:var(--review);border-color:var(--review)}
.tag.flag{color:var(--muted)}.crops{display:flex;gap:4px;flex-wrap:wrap;align-items:flex-end}.crops img{height:72px;border-radius:3px}
.crops .lab{font-size:12px;width:100%}.ci{border:2px solid var(--i)}.cj{border:2px solid var(--j)}
.scroll{overflow-x:auto}.legend span{margin-right:14px}
"""


def _img(png: bytes, cls: str = "plot") -> str:
    return f'<img class="{cls}" src="data:image/png;base64,{base64.b64encode(png).decode()}">'


def _crop(frames_dir: str | None, frame: int, box: np.ndarray, cls: str) -> str:
    if not frames_dir:
        return ""
    img = cv2.imread(str(Path(frames_dir) / f"{frame:06d}.jpg"))
    if img is None:
        return ""
    cx, cy, w, h = box
    m = 0.15
    x1, y1 = int(max(cx - w * (0.5 + m), 0)), int(max(cy - h * (0.5 + m), 0))
    x2, y2 = int(min(cx + w * (0.5 + m), img.shape[1])), int(min(cy + h * (0.5 + m), img.shape[0]))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return ""
    c = img[y1:y2, x1:x2]
    scale = 72 / c.shape[0]
    c = cv2.resize(c, (max(int(c.shape[1] * scale), 1), 72))
    ok, buf = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return f'<img class="{cls}" title="帧 {frame}" src="data:image/jpeg;base64,{base64.b64encode(buf).decode()}">'


def _f(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def _pct(v):
    return "–" if v is None or v != v else f"{100 * v:.1f}%"


def _evidence_table(e) -> str:
    rows = [
        ("链接代价 C / 概率 P", f"{_f(e.C)} / {_f(e.P)}"),
        ("缺口", f"{e.gap} 帧 · {e.tau:.2f} s"),
        ("运动 ẽ_f / ẽ_b", f"{_f(e.e_f)} / {_f(e.e_b)}（目标尺度）"),
        ("运动 C_m / Q_m · ρ", f"{_f(e.C_m)} / {_f(e.Q_m, 2)} · {_f(e.rho, 2)}"),
        ("外观 d_set / d_proto", f"{_f(e.d_set)} / {_f(e.d_proto)}"),
        ("外观 C_a / Q_a", f"{_f(e.C_a)} / {_f(e.Q_a, 2)}"),
        ("尺度 C_s / Q_s", f"{_f(e.C_s)} / {_f(e.Q_s, 0)}"),
        ("证据充分度 E", _f(e.E, 2)),
        ("互选 MNN · 歧义度 A", f"{'是' if e.mnn else '否'} · {_f(e.A)}"),
    ]
    return "<table>" + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows) + "</table>"


def _review_card(res, e, cfg, k: int) -> str:
    seq = res.seq
    tri, trj = seq.tracklets[e.i], seq.tracklets[e.j]
    fig = edge_figure(seq, res.edges, res.gid, e, cfg)
    crops_i = "".join(_crop(seq.frames_dir, f, tri.boxes[np.searchsorted(tri.frames, f)], "ci") for f in e.reps_i[:5])
    crops_j = "".join(_crop(seq.frames_dir, f, trj.boxes[np.searchsorted(trj.frames, f)], "cj") for f in e.reps_j[:5])
    if not crops_i:
        crops_i = _crop(seq.frames_dir, tri.t_e, tri.boxes[-1], "ci")
    if not crops_j:
        crops_j = _crop(seq.frames_dir, trj.t_s, trj.boxes[0], "cj")
    comp = "".join(f"<tr><td>{seq.tracklets[a].tid} → {seq.tracklets[b].tid}</td><td>{_f(c)}</td></tr>"
                   for a, b, c in e.competitors[:6]) or "<tr><td colspan=2 class=muted>无</td></tr>"
    flags = "".join(f'<span class="tag flag">{html.escape(x)}</span>' for x in sorted(e.flags))
    truth = ""
    if e.label in (0, 1):
        truth = f'<span class="tag">{"真值：同一目标" if e.label == 1 else "真值：不同目标"}</span>'
    why = "求解器选中但未达自动条件" if e.x else "未被选中，但是一端的最优候选且代价低于 θ_review"
    return f"""<div class="card"><h3>#{k} · 轨迹 {tri.tid} → {trj.tid}
<span class="tag review">待复核</span>{truth}{flags}</h3>
<div class="muted">{why}。蓝=前段 {tri.tid}（{tri.cls}，帧 {tri.t_s}–{tri.t_e}），红=后段 {trj.tid}（帧 {trj.t_s}–{trj.t_e}），灰=竞争者。</div>
<div class="row"><div>{_img(fig_to_png(fig, 100))}
<div class="crops"><div class="lab">前段代表帧</div>{crops_i}</div>
<div class="crops"><div class="lab">后段代表帧</div>{crops_j}</div></div>
<div>{_evidence_table(e)}<h3 style="margin-top:10px">竞争候选（代价相近）</h3>
<table><tr><th>边</th><th>代价</th></tr>{comp}</table></div></div></div>"""


def build_report(results: list, metrics: dict | None, cfg: dict, out_path: str | Path, max_cards: int = 25,
                 title: str = "STWO 轨迹重连审查报告") -> Path:
    parts = [f"<!doctype html><html lang=zh><head><meta charset=utf-8>"
             f"<meta name=viewport content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>"
             f"<style>{CSS}</style></head><body><main><h1>{html.escape(title)}</h1>"
             f"<div class=muted>实验 {html.escape(str(cfg.get('exp_name')))} · 融合 {cfg['fusion']['mode']} · "
             f"求解 {cfg['association']['solver']} · 运动 {cfg['motion']['space']}/{cfg['motion']['cost']} · "
             f"θ_auto={_f(cfg['decision']['theta_auto'])} θ_link={_f(cfg['association']['theta_link'])}</div>"]
    if metrics:
        mo = metrics.get("mot_auto", {})
        ml = metrics.get("mot_link", {})
        mr = metrics.get("mot_raw", {})
        kp = [("自动档精度", _pct(metrics.get("auto_precision"))), ("自动档覆盖率", _pct(metrics.get("auto_coverage"))),
              ("边精度", _pct(metrics.get("link_precision"))), ("边召回", _pct(metrics.get("link_recall"))),
              ("门控召回", _pct(metrics.get("gate_recall"))), ("复核 / 分钟", _f(metrics.get("review_per_min"), 1)),
              ("IDF1 上游 → 自动档", f"{_pct(mr.get('IDF1'))} → {_pct(mo.get('IDF1'))}"),
              ("IDF1 全部链接", _pct(ml.get("IDF1"))), ("ID 切换 上游 → 自动档", f"{mr.get('IDsw', '–')} → {mo.get('IDsw', '–')}")]
        parts.append("<div class=kpis>" + "".join(f"<div class=kpi><b>{v}</b><span>{k}</span></div>" for k, v in kp)
                     + "</div>")
        bk = metrics.get("buckets", {})
        if bk:
            parts.append("<h3>按缺口分桶</h3><div class=scroll><table><tr><th>缺口</th><th>精度</th><th>召回</th>"
                         "<th>真链接数</th></tr>" + "".join(
                f"<tr><td>{k}</td><td>{_pct(v['precision'])}</td><td>{_pct(v['recall'])}</td><td>{v['n_true']}</td></tr>"
                for k, v in bk.items()) + "</table></div>")
        if metrics.get("reliability"):
            parts.append(_img(fig_to_png(reliability_figure(metrics["reliability"], metrics["ece"]))))
    parts.append("<p class='legend muted'><span>● 起点</span><span>■ 终点</span><span style='color:#1a9850'>— 自动连接</span>"
                 "<span style='color:#f28e2b'>-- 待复核</span><span>··· 插值</span><span>黑线 = 自车</span></p>")
    for res in results:
        seq = res.seq
        rev = sorted([e for e in res.edges if e.decision == "review"], key=lambda e: e.A)
        auto = [e for e in res.edges if e.decision == "auto"]
        parts.append(f"<h2>{html.escape(seq.name)}</h2><div class=muted>{len(seq.tracklets)} 条上游轨迹 · "
                     f"{len(res.edges)} 条候选边 · 自动 {len(auto)} · 待复核 {len(rev)} · 合并后 {int(res.gid.max()) if len(res.gid) else 0} 个 ID</div>")
        parts.append(_img(fig_to_png(overview_figure(seq, res.edges, res.gid, cfg), 100)))
        if rev:
            parts.append(f"<h3 style='margin-top:16px'>复核队列（按歧义度从高到低，显示前 {min(len(rev), max_cards)} 条）</h3>")
            parts += [_review_card(res, e, cfg, k + 1) for k, e in enumerate(rev[:max_cards])]
        if auto:
            rows = "".join(
                f"<tr><td>{seq.tracklets[e.i].tid} → {seq.tracklets[e.j].tid}</td><td>{e.gap}</td><td>{_f(e.C)}</td>"
                f"<td>{_f(e.A)}</td><td>{_f(e.E, 2)}</td>"
                f"<td>{'' if e.label < 0 else ('✓' if e.label == 1 else '✗ 错误')}</td></tr>" for e in auto)
            parts.append("<details><summary>自动连接抽查表</summary><div class=scroll><table><tr><th>边</th><th>缺口(帧)</th>"
                         f"<th>C</th><th>A</th><th>E</th><th>真值</th></tr>{rows}</table></div></details>")
    parts.append("</main></body></html>")
    out_path = Path(out_path)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path
