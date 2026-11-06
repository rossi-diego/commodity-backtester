"""Page CSS for the Streamlit app, injected once by ``app.main``."""

from __future__ import annotations

from typing import Final

__all__ = ["APP_CSS"]

# Amber means "the crush margin" and nothing else: controls and the other cards stay neutral.
# The rules target Streamlit test ids and key classes, so redo the screenshots when upgrading.
# The top padding clears the 56 px bar Streamlit Cloud draws over the app from its parent page,
# the bottom padding its floating badge, which covered the last card on a phone.
# The risk strip labels keep their case: uppercase would turn σ into Σ.
# In Streamlit 1.66 stButtonGroup is a block; the row of chips is its child div, so wrap that.
APP_CSS: Final[str] = """<style>
:root{--page:#0b1016;--card:#141b23;--raised:#1e2732;--line:#2a3440;--ink:#eef2f6;--mut:#a3afbd;--head:linear-gradient(115deg,#1a2430 0%,#16212c 55%,#3a2c10 100%);--crush:#c58512;--oil:#4d8fe8;--meal:#c76fae;--step:rgba(255,255,255,.07)}
[data-testid='stMainBlockContainer']{max-width:1480px;margin:0 auto;padding:3.75rem 2rem 5rem}
[data-testid='stHeaderActionElements']{display:none}
:focus-visible{outline:2px solid var(--ink)!important;outline-offset:2px}
.cm-hero{background:var(--head);border-radius:12px;padding:1.1rem 1.5rem;color:#fff;display:grid;grid-template-columns:minmax(0,13fr) minmax(0,11fr);gap:1.5rem;align-items:center}
.cm-hero h1{color:#fff;font-size:2.25rem;font-weight:600;letter-spacing:-.02em;line-height:1.1;margin:0 0 .4rem;padding:0}
.cm-q{font-size:1.0625rem;line-height:1.35;margin:0 0 .4rem;color:#fff}
.cm-meta{font-size:.875rem;line-height:1.45;margin:0;color:#dbe6f2}
.cm-meta a{color:#fff;font-weight:500}
.cm-meta strong{color:#fff;font-weight:600}
.cm-scope{margin-top:.4rem;font-size:.8125rem}
.cm-nowrap{white-space:nowrap}
.cm-steps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.75rem;margin:0;padding:0;list-style:none}
.cm-steps a{display:block;height:100%;box-sizing:border-box;background:var(--step);border:1px solid rgba(255,255,255,.18);border-radius:10px;padding:.6rem .75rem;font-size:.8125rem;line-height:1.3;color:#e8eff7;text-decoration:none}
.cm-steps a:hover{border-color:rgba(255,255,255,.45)}
.cm-steps b{color:#fff;font-size:.9375rem;font-weight:600}
.cm-steps span{display:block;margin-top:.3rem}
.cm-steps i,.cm-label i{font-style:normal;display:inline-block;width:1.35rem;height:1.35rem;line-height:1.35rem;text-align:center;border-radius:50%;background:#fff;color:#0b1016;font-weight:600;font-size:.75rem;margin-right:.35rem;flex:none}
.cm-kpis{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));grid-template-rows:repeat(5,auto);gap:0 12px}
.cm-kpi{grid-row:span 5;display:grid;grid-template-rows:subgrid;row-gap:8px;background:var(--card);border:1px solid var(--line);border-top:3px solid var(--acc);border-radius:10px;padding:14px 18px 16px;color:var(--ink);text-decoration:none}
.cm-kpi:hover{border-color:#3b4653;border-top-color:var(--acc)}
.cm-kpi-share{--acc:var(--oil)}
.cm-kpi-gap{--acc:var(--ink)}
.cm-kpi-crush{--acc:var(--crush)}
.cm-kpi-crush .cm-value{color:var(--crush)}
.cm-label{margin:0;display:flex;align-items:center;flex-wrap:wrap;gap:.2rem;font-size:.875rem;font-weight:600;color:var(--ink)}
.cm-tag{font-style:normal;font-weight:400;font-size:.6875rem;border:1px solid var(--line);border-radius:4px;padding:0 .3rem;margin-left:.3rem;color:var(--mut)}
.cm-value{margin:0;font-size:2rem;font-weight:600;line-height:1.1;letter-spacing:-.01em;white-space:nowrap;font-variant-numeric:tabular-nums}
.cm-value small{font-size:.8125rem;font-weight:400;letter-spacing:0;color:var(--mut);margin-left:.5rem;white-space:normal}
.cm-detail{font-size:.875rem;line-height:1.4;color:var(--ink);align-self:center}
.cm-rank{display:flex;flex-direction:column;gap:5px;font-size:.75rem;color:var(--mut)}
.cm-track{position:relative;height:6px;border-radius:3px;background:repeating-linear-gradient(90deg,var(--raised) 0 calc(20% - 2px),transparent calc(20% - 2px) 20%)}
.cm-mark{position:absolute;top:-4px;width:4px;height:14px;border-radius:2px;background:var(--acc);transform:translateX(-50%)}
.cm-caption{margin:0;font-size:.78rem;line-height:1.35;color:var(--mut)}
.cm-split{display:flex;gap:2px;height:10px;border-radius:5px;overflow:hidden}
.cm-split-oil{background:var(--oil)}
.cm-split-meal{flex:1;background:var(--meal)}
.cm-legend{margin:5px 0 0;font-size:.8125rem;font-variant-numeric:tabular-nums}
.cm-legend i{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:.3rem}
.cm-key-oil{background:var(--oil)}
.cm-key-meal{background:var(--meal);margin-left:.75rem}
[class*='st-key-card-sec']{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.9rem 1rem 1rem;gap:.6rem}
.cm-head{scroll-margin-top:72px}
.cm-head p{margin:0}
.cm-head .cm-label{font-size:1.0625rem}
.cm-head p+p{margin-top:.3rem;font-size:.9375rem;line-height:1.4;color:var(--ink)}
[class*='st-key-bar-']{min-height:40px;align-items:center;flex-wrap:wrap}
.st-key-rin-gap{margin-left:auto}
[class*='st-key-view-'] [data-testid='stButtonGroup']{justify-content:flex-end}
[class*='st-key-body-'] [data-testid='stHorizontalBlock']{align-items:stretch}
.cm-rail{height:400px;box-sizing:border-box;display:flex;flex-direction:column;gap:.75rem;border-left:1px solid var(--line);padding-left:1rem;font-size:.8125rem;line-height:1.45;color:var(--ink);overflow:hidden}
.cm-rail h3{font-size:.6875rem;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:var(--mut);margin:0 0 .2rem;padding:0}
.cm-rail p{margin:0}
.cm-today{margin-top:auto}
.cm-today dl{display:grid;grid-template-columns:1fr auto;gap:.25rem .75rem;margin:0;padding:0}
.cm-today dt{margin:0;padding:0;color:var(--mut)}
.cm-today dd{margin:0;padding:0;text-align:right;font-variant-numeric:tabular-nums}
.cm-risk{border-top:1px solid var(--line);padding-top:.75rem;color:var(--ink)}
.cm-risk dl{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.75rem 1.25rem;margin:0;padding:0}
.cm-risk-cell{min-width:0}
.cm-risk dt{margin:0 0 .2rem;padding:0;font-size:.75rem;font-weight:500;color:var(--mut)}
.cm-risk dd{margin:0;padding:0;font-size:.9375rem;font-weight:600;font-variant-numeric:tabular-nums}
.cm-risk-na{margin:0;font-size:.875rem;color:var(--mut)}
.cm-risk-note{margin:.6rem 0 0;font-size:.78rem;line-height:1.35;color:var(--mut)}
.cm-method{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.cm-method section{display:flex;flex-direction:column;background:var(--card);border:1px solid var(--line);border-top:3px solid var(--line);border-radius:10px;padding:.9rem 1rem}
.cm-method h2{font-size:.9375rem;font-weight:600;margin:0 0 .5rem;padding:0;color:var(--ink)}
.cm-method p{font-size:.875rem;line-height:1.45;color:var(--ink);margin:0 0 .5rem}
.cm-method a{color:var(--ink)}
.cm-method .cm-closing{margin:auto 0 0;padding-top:.25rem;font-size:.8125rem;color:var(--mut)}
[data-testid='stCaptionContainer']{opacity:1}
[data-testid='stCaptionContainer'] p{color:var(--mut)}
[data-testid='stCaptionContainer'] a{color:var(--ink)}
@media (max-width:1279px){
 [class*='st-key-body-'] [data-testid='stColumn']{min-width:100%}
 .cm-rail{height:auto;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem;border-left:0;border-top:1px solid var(--line);padding:.75rem 0 0}
 .cm-today{margin-top:0}
}
@media (max-width:1024px){
 .cm-hero{grid-template-columns:1fr;gap:.75rem}
 .cm-steps span{display:none}
 [class*='st-key-card-sec'] [data-testid='stColumn']{min-width:100%}
 .cm-value{font-size:1.625rem}
}
@media (max-width:1023px){
 .cm-risk dl{grid-template-columns:repeat(2,minmax(0,1fr))}
}
@media (max-width:760px){
 .cm-kpis{grid-template-columns:1fr;grid-template-rows:none;gap:12px}
 .cm-kpi{grid-row:auto;grid-template-rows:none}
 .cm-method{grid-template-columns:1fr}
 .cm-rail{grid-template-columns:1fr}
}
@media (max-width:640px){
 [data-testid='stMainBlockContainer']{padding:3.75rem .75rem 5rem}
 [class*='st-key-bar-'] [data-testid='stButtonGroup']>div{flex-wrap:wrap;row-gap:.4rem}
 .cm-hero{padding:1rem;border-radius:10px}
 .cm-hero h1{font-size:1.75rem}
 .cm-q{font-size:1rem}
 .cm-steps{display:none}
 [class*='st-key-card-sec']{padding:.7rem .8rem}
 .cm-risk dl{grid-template-columns:1fr}
 [data-testid='stPlotlyChart'] .nsewdrag{pointer-events:none}
 [data-testid='stPlotlyChart'] .modebar-container{display:none}
}
</style>"""
