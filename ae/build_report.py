#!/usr/bin/env python3
"""Generate a self-contained HTML results page from the rendered figures.

Single-column, full-width layout (no cramped side-by-side panels), with each
figure base64-embedded so `ae/results.html` is one portable file — open it
directly, serve it (`python -m http.server`), or drop it on Cloudflare Pages.

    make figures            # render ae/figures/*.png
    python ae/build_report.py   # → ae/results.html
"""
from __future__ import annotations

import base64
import html
import os

_AE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(_AE_DIR, "figures")
OUT = os.path.join(_AE_DIR, "results.html")

# (heading, html_prose, [(chart_id, caption), ...])
SECTIONS = [
    ("Headline (Fig 1)",
     "TensorDex stores a randomly-sampled Hugging Face corpus in <b>0.29×</b> "
     "the space (70.5% saved) — below every baseline — at high throughput. "
     "Reproduce the reduction with <code>make verify</code>; throughput is "
     "hardware-measured.",
     [("codec_storage_reduction", "Fig 1 (left) — normalized storage reduction vs baselines."),
      ("codec_throughput", "Fig 1 (right) — compression / decompression throughput.")]),

    ("End-to-end storage reduction (§6.2, Fig 11)",
     "Both variants beat all baselines and keep improving as the corpus grows: "
     "<b>FM++ 70.5%</b>, <b>TensorX 65.1%</b>, vs ZipLLM's 51.9%. "
     "<code>make verify</code> re-derives the per-tensor ratios bit-exact.",
     [("reduction_global", "Fig 11a — cumulative data reduction over the trace."),
      ("zipllm_real_cdf", "Fig 11b — per-tensor reduction CDF, all methods."),
      ("reduction_violin_by_family", "Fig 11c — reduction distribution by model family.")]),

    ("TensorSketch &amp; prediction (§6.3, Fig 12–13)",
     "TensorSketch + HNSW selects the <b>same delta base</b> as exact search — "
     "Recall@1 = 1.00 — from 8&nbsp;KB fingerprints. Reproduce with "
     "<code>make verify-recall</code>.",
     [("bcs_recall", "Fig 12a — Recall@1 vs baselines."),
      ("bcs_qps", "Fig 12b — end-to-end query throughput."),
      ("bcs_memory", "Fig 12c — per-query IO."),
      ("pred_vs_real_ratio", "Fig 13 — predicted vs real reduction ratio.")]),

    ("FlexSplit clustering (§6.4, Fig 14–16)",
     "FlexSplit stays within a few points of the ILP optimum at near-constant "
     "time, while ILP grows super-linearly.",
     [("algo_bench_q_proj", "Fig 14a — scalability vs ILP / Primal-Dual (q_proj)."),
      ("algo_bench_v_proj", "Fig 14b — scalability (v_proj)."),
      ("flexsplit_cluster_size", "Fig 15a — cluster-size distribution."),
      ("flexsplit_split_overview", "Fig 15b — split characteristics."),
      ("flexsplit_split_effect", "Fig 16a — reduction before vs after Phase-II split."),
      ("flexsplit_post_split_cr", "Fig 16b — post-split reduction gain.")]),

    ("Motivation (main body, Fig 2 &amp; 4)",
     "Model-hub storage is exploding and lineage metadata is mostly missing — "
     "the problem TensorDex targets.",
     [("modelhub_growth", "Fig 2 — Hugging Face storage &amp; model-count growth."),
      ("modelhub_metadata", "Fig 4 — metadata availability by download rank.")]),
]

CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin: 0; font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       color: #1a1a1a; background: #fafafa; }
.wrap { max-width: 860px; margin: 0 auto; padding: 40px 24px 80px; }
h1 { font-size: 34px; margin: 0 0 6px; }
.sub { color: #666; margin: 0 0 32px; }
h2 { font-size: 24px; margin: 48px 0 8px; padding-top: 16px; border-top: 1px solid #e2e2e2; }
p { margin: 8px 0 20px; }
code { background: #eef1f4; padding: 1px 6px; border-radius: 4px; font-size: 90%; }
figure { margin: 28px 0; }
figure img { width: 100%; height: auto; border: 1px solid #e2e2e2; border-radius: 8px;
             background: #fff; padding: 8px; }
figcaption { color: #555; font-size: 14px; margin-top: 8px; text-align: center; }
.missing { border: 1px dashed #bbb; border-radius: 8px; padding: 40px; text-align: center;
           color: #999; background: #fff; }
footer { color: #888; font-size: 13px; margin-top: 48px; border-top: 1px solid #e2e2e2; padding-top: 16px; }
"""


def img_tag(cid: str, caption: str) -> str:
    path = os.path.join(FIGDIR, f"{cid}.png")
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        media = f'<img alt="{html.escape(caption)}" src="data:image/png;base64,{b64}">'
    else:
        media = (f'<div class="missing">{html.escape(cid)}.png not generated yet — '
                 f'run <code>make figures</code></div>')
    return f'<figure>{media}<figcaption>{caption}</figcaption></figure>'


def main() -> int:
    n_fig = n_missing = 0
    parts = []
    for heading, prose, figs in SECTIONS:
        parts.append(f"<h2>{heading}</h2>\n<p>{prose}</p>")
        for cid, cap in figs:
            parts.append(img_tag(cid, cap))
            n_fig += 1
            if not os.path.exists(os.path.join(FIGDIR, f"{cid}.png")):
                n_missing += 1
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TensorDex — Results</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>TensorDex — Results</h1>
<p class="sub">Artifact-evaluation results, one figure per row. Regenerate with
<code>make figures &amp;&amp; python ae/build_report.py</code>.</p>
{''.join(parts)}
<footer>Generated from <code>ae/figures/</code> (published <code>results.db</code> cache).
See <code>ae/README.md</code> for the three reproducibility tiers and
<code>ae/FIGURE_MAP.md</code> for the figure↔chart mapping.</footer>
</div></body></html>"""
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(doc)
    sz = os.path.getsize(OUT) / 1e6
    print(f"wrote {OUT} ({sz:.1f} MB, {n_fig} figures"
          + (f", {n_missing} not yet rendered" if n_missing else "") + ")")
    if n_missing:
        print(f"WARNING: {n_missing} figures are missing from ae/figures/ — the "
              f"report is INCOMPLETE. Run `make figures` first, then `make report`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
