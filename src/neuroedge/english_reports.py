"""Small dependency-free renderer for portable English research summaries."""
import base64
from html import escape
from pathlib import Path


def portable_page(directory,title,introduction,tables,images=()):
    directory=Path(directory)
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+escape(title)+'</title>'
    page+='<style>body{font:16px system-ui;line-height:1.7;max-width:1200px;margin:32px auto;padding:20px;background:#f4f7fb;color:#17243b}section{background:white;padding:24px;margin:20px 0;border:1px solid #dce4ef;border-radius:12px}h1{line-height:1.25}img{max-width:100%;height:auto}.table{overflow-x:auto}table{border-collapse:collapse;font-size:13px;white-space:nowrap}th,td{padding:9px;border-bottom:1px solid #dce4ef;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px system-ui}a{color:#1d4ed8}</style><h1>'+escape(title)+'</h1><p>'+escape(introduction)+'</p>'
    for name,table in tables.items():page+='<section><h2>'+escape(name)+'</h2><div class="table">'+table.to_html(index=False,border=0,float_format=lambda x:f'{x:.5f}')+'</div></section>'
    for name in images:
        path=directory/name
        if path.exists():page+='<section><img alt="'+escape(path.stem.replace('_',' '))+'" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'"></section>'
    report=directory/'REPORT.md'
    if report.exists():page+='<section><h2>Complete methods and interpretation</h2><pre>'+escape(report.read_text())+'</pre></section>'
    page+='<p><a href="REPORT.md">English report</a> · <a href="../session_adaptation/results.html">Expanded session-adaptation study</a></p></html>'
    (directory/'results.html').write_text(page,encoding='utf-8')
