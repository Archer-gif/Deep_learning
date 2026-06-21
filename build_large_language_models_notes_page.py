from __future__ import annotations

import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MD_PATH = ROOT / "large-language-models-notes.md"
OUT_PATH = ROOT / "large-language-models.html"


PRIORITY_CLASS = {
    "重点": "important",
    "了解": "know",
    "简单带过": "pass",
    "重点/了解": "important",
}


def inline_md(text: str) -> str:
    placeholders: list[str] = []

    def keep_math(match: re.Match[str]) -> str:
        placeholders.append(match.group(0))
        return f"@@MATH{len(placeholders) - 1}@@"

    text = re.sub(r"\$[^$\n]+\$", keep_math, text)
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    for i, value in enumerate(placeholders):
        text = text.replace(f"@@MATH{i}@@", value)
    return text


def split_priority(title: str) -> tuple[str, str | None]:
    match = re.search(r"【([^】]+)】\s*$", title)
    if not match:
        return title.strip(), None
    priority = match.group(1)
    return title[: match.start()].strip(), priority


def title_html(title: str) -> str:
    clean, priority = split_priority(title)
    rendered = inline_md(clean)
    if priority:
        cls = PRIORITY_CLASS.get(priority, "know")
        rendered += f' <span class="tag {cls}">{html.escape(priority)}</span>'
    return rendered


def nav_title(title: str) -> str:
    clean, _ = split_priority(title)
    clean = re.sub(r"^\d+(?:\.\d+)*\s*[.、]?\s*", "", clean)
    return clean.strip()


def slugify(title: str, index: int) -> str:
    clean = nav_title(title)
    mapping = {
        "大语言模型复习笔记": "overview",
        "复习优先级": "priority",
        "大语言模型主线：从“预测下一个词”到“会查资料再回答”": "llm-mainline",
        "基于统计的语言模型：N-gram": "n-gram",
        "语言模型采样：模型给概率，人类定策略": "sampling",
        "预训练、曝光偏差与扩展法则": "pretraining",
        "Transformer 的两类核心模块": "transformer",
        "三种大语言模型架构总览": "architectures",
        "Encoder-only：输入 token 之间充分互相理解": "encoder-only",
        "Decoder-only：只看过去，边生成边扩展上下文": "decoder-only",
        "Encoder-Decoder：输入理解和输出生成分工": "encoder-decoder",
        "三种注意力：完全、下三角、交叉": "attention-types",
        "后训练：让会接龙的模型更会说话": "post-training",
        "参数高效微调：Adapter 与 LoRA": "peft",
        "检索增强生成 RAG：为什么需要外部知识": "rag-why",
        "RAG 基本组成：知识检索 + 生成增强": "rag-components",
        "RAG 中的编码器：把问题和文档变成可比较的向量": "rag-encoder",
        "RAG 架构分类：黑盒增强与白盒增强": "rag-architecture",
        "何时增强、何处增强、何以增强": "rag-when-where-how",
        "知识检索流程：构库、构问、召回、重排": "retrieval-pipeline",
        "RAG 的降本增效": "rag-efficiency",
        "一页速记": "summary",
    }
    return mapping.get(clean, f"section-{index}")


def render_math(lines: list[str]) -> str:
    return f'<div class="formula">\n{html.escape(chr(10).join(lines))}\n</div>'


def render_table(lines: list[str]) -> str:
    rows = []
    for raw in lines:
        cells = [inline_md(cell.strip()) for cell in raw.strip().strip("|").split("|")]
        rows.append(cells)
    head = rows[0]
    body = rows[2:]
    out = ["<table>", "<thead><tr>"]
    out.extend(f"<th>{cell}</th>" for cell in head)
    out.append("</tr></thead><tbody>")
    for row in body:
        out.append("<tr>")
        out.extend(f"<td>{cell}</td>" for cell in row)
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def render_list(lines: list[str], ordered: bool) -> str:
    tag = "ol" if ordered else "ul"
    pattern = r"^\s*\d+\.\s+" if ordered else r"^\s*-\s+"
    out = [f"<{tag}>"]
    for raw in lines:
        out.append(f"<li>{inline_md(re.sub(pattern, '', raw).strip())}</li>")
    out.append(f"</{tag}>")
    return "\n".join(out)


def render_code(lines: list[str]) -> str:
    return f"<pre><code>{html.escape(chr(10).join(lines))}</code></pre>"


def convert_blocks(lines: list[str]) -> str:
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        if line.strip().startswith("```"):
            code_lines: list[str] = []
            i += 1
            while i < len(lines):
                if lines[i].strip().startswith("```"):
                    i += 1
                    break
                code_lines.append(lines[i])
                i += 1
            out.append(render_code(code_lines))
            continue

        if line.strip() == "$$":
            formula = [line]
            i += 1
            while i < len(lines):
                formula.append(lines[i])
                if lines[i].strip() == "$$":
                    i += 1
                    break
                i += 1
            out.append(render_math(formula))
            continue

        if line.strip().startswith("|") and i + 1 < len(lines):
            table_lines = [line, lines[i + 1]]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            out.append(render_table(table_lines))
            continue

        if re.match(r"^\s*-\s+", line):
            list_lines = []
            while i < len(lines) and re.match(r"^\s*-\s+", lines[i]):
                list_lines.append(lines[i])
                i += 1
            out.append(render_list(list_lines, ordered=False))
            continue

        if re.match(r"^\s*\d+\.\s+", line):
            list_lines = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                list_lines.append(lines[i])
                i += 1
            out.append(render_list(list_lines, ordered=True))
            continue

        hmatch = re.match(r"^(#{1,4})\s+(.+)$", line)
        if hmatch:
            level = min(len(hmatch.group(1)) + 1, 4)
            out.append(f"<h{level}>{title_html(hmatch.group(2).strip())}</h{level}>")
            i += 1
            continue

        if line.startswith("> "):
            quote_lines = []
            while i < len(lines) and lines[i].startswith("> "):
                quote_lines.append(lines[i][2:].strip())
                i += 1
            out.append(f"<blockquote>{' '.join(inline_md(q) for q in quote_lines)}</blockquote>")
            continue

        para = [line.strip()]
        i += 1
        while (
            i < len(lines)
            and lines[i].strip()
            and not re.match(r"^(#{1,4})\s+", lines[i])
            and not lines[i].startswith("> ")
            and lines[i].strip() != "$$"
            and not lines[i].strip().startswith("|")
            and not re.match(r"^\s*(-|\d+\.)\s+", lines[i])
        ):
            para.append(lines[i].strip())
            i += 1
        out.append(f"<p>{inline_md(' '.join(para))}</p>")
    return "\n".join(out)


def convert_markdown(md: str) -> tuple[str, list[tuple[int, str, str]]]:
    lines = md.splitlines()
    out: list[str] = []
    headings: list[tuple[int, str, str]] = []
    i = 0
    section_count = 0
    in_section = False

    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        if line.strip().startswith("```"):
            code_lines: list[str] = []
            i += 1
            while i < len(lines):
                if lines[i].strip().startswith("```"):
                    i += 1
                    break
                code_lines.append(lines[i])
                i += 1
            out.append(render_code(code_lines))
            continue

        if line.startswith("<details>"):
            summary = ""
            inner: list[str] = []
            i += 1
            while i < len(lines):
                if lines[i].startswith("<summary>"):
                    summary = lines[i]
                    i += 1
                    continue
                if lines[i].strip() == "</details>":
                    i += 1
                    break
                inner.append(lines[i])
                i += 1
            out.append("<details>\n" + summary + "\n" + convert_blocks(inner) + "\n</details>")
            continue

        if line.startswith("# "):
            title = line[2:].strip()
            out.append(f'<section class="hero" id="overview"><h1>{title_html(title)}</h1>')
            headings.append((1, title, "overview"))
            in_section = True
            i += 1
            continue

        hmatch = re.match(r"^(#{2,4})\s+(.+)$", line)
        if hmatch:
            level = len(hmatch.group(1))
            title = hmatch.group(2).strip()
            if level == 2:
                if in_section:
                    out.append("</section>")
                section_count += 1
                sid = slugify(title, section_count)
                headings.append((level, title, sid))
                out.append(f'<section class="section" id="{sid}"><h2>{title_html(title)}</h2>')
                in_section = True
            else:
                html_level = min(level, 4)
                out.append(f"<h{html_level}>{title_html(title)}</h{html_level}>")
            i += 1
            continue

        if line.startswith("> "):
            quote_lines = []
            while i < len(lines) and lines[i].startswith("> "):
                quote_lines.append(lines[i][2:].strip())
                i += 1
            out.append(f"<blockquote>{' '.join(inline_md(q) for q in quote_lines)}</blockquote>")
            continue

        if line.strip() == "$$":
            formula = [line]
            i += 1
            while i < len(lines):
                formula.append(lines[i])
                if lines[i].strip() == "$$":
                    i += 1
                    break
                i += 1
            out.append(render_math(formula))
            continue

        if line.strip().startswith("|") and i + 1 < len(lines):
            table_lines = [line, lines[i + 1]]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            out.append(render_table(table_lines))
            continue

        if re.match(r"^\s*-\s+", line):
            list_lines = []
            while i < len(lines) and re.match(r"^\s*-\s+", lines[i]):
                list_lines.append(lines[i])
                i += 1
            out.append(render_list(list_lines, ordered=False))
            continue

        if re.match(r"^\s*\d+\.\s+", line):
            list_lines = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                list_lines.append(lines[i])
                i += 1
            out.append(render_list(list_lines, ordered=True))
            continue

        para = [line.strip()]
        i += 1
        while (
            i < len(lines)
            and lines[i].strip()
            and not re.match(r"^(#{1,4})\s+", lines[i])
            and not lines[i].startswith("> ")
            and not lines[i].startswith("<details>")
            and lines[i].strip() != "$$"
            and not lines[i].strip().startswith("|")
            and not re.match(r"^\s*(-|\d+\.)\s+", lines[i])
        ):
            para.append(lines[i].strip())
            i += 1
        out.append(f"<p>{inline_md(' '.join(para))}</p>")
        continue

    if in_section:
        out.append("</section>")
    return "\n".join(out), headings


def build_nav(headings: list[tuple[int, str, str]]) -> str:
    links = []
    for level, title, sid in headings:
        if level == 1 or level == 2:
            links.append(f'<a href="#{sid}">{inline_md(nav_title(title))}</a>')
    return "\n        ".join(links)


STYLE = """
    :root {
      --bg: #f6f7fb;
      --paper: #ffffff;
      --ink: #20242c;
      --muted: #667085;
      --line: #dfe3ea;
      --nav: #14213d;
      --nav-soft: #22345f;
      --accent: #c62828;
      --accent-soft: #fff1f1;
      --blue: #246bfe;
      --green: #17805b;
      --shadow: 0 14px 34px rgba(20, 33, 61, 0.12);
    }

    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", Arial, sans-serif;
      line-height: 1.72;
    }

    .layout {
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
      min-height: 100vh;
    }

    aside {
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
      background: var(--nav);
      color: #fff;
      padding: 24px 18px;
      border-right: 1px solid rgba(255,255,255,0.08);
    }

    .brand {
      padding: 6px 8px 18px;
      border-bottom: 1px solid rgba(255,255,255,0.14);
      margin-bottom: 16px;
    }

    .brand h1 {
      margin: 0 0 8px;
      font-size: 24px;
      line-height: 1.18;
      letter-spacing: 0;
    }

    .brand p {
      margin: 0;
      color: rgba(255,255,255,0.74);
      font-size: 13px;
    }

    nav a {
      display: block;
      color: rgba(255,255,255,0.84);
      text-decoration: none;
      padding: 8px 10px;
      border-radius: 6px;
      font-size: 14px;
    }

    nav a:hover,
    nav a.active {
      background: var(--nav-soft);
      color: #fff;
    }

    main {
      min-width: 0;
      padding: 28px;
    }

    .hero,
    .section {
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(20, 33, 61, 0.06);
      padding: 24px;
      margin-bottom: 18px;
    }

    .hero {
      background: linear-gradient(120deg, #ffffff 0%, #f4f8ff 100%);
      box-shadow: var(--shadow);
      padding: 28px;
      margin-bottom: 22px;
    }

    h1 {
      margin: 0 0 10px;
      font-size: 34px;
      line-height: 1.16;
      letter-spacing: 0;
    }

    h2 {
      margin: 0 0 16px;
      font-size: 26px;
      line-height: 1.25;
      letter-spacing: 0;
    }

    h3 {
      margin: 24px 0 10px;
      font-size: 19px;
      line-height: 1.32;
      letter-spacing: 0;
    }

    h4 {
      margin: 18px 0 8px;
      font-size: 16px;
      line-height: 1.35;
      letter-spacing: 0;
    }

    p { margin: 10px 0; }
    ul, ol { padding-left: 22px; }
    li { margin: 4px 0; }

    blockquote {
      margin: 14px 0;
      border-left: 4px solid var(--accent);
      background: var(--accent-soft);
      padding: 12px 14px;
      border-radius: 0 8px 8px 0;
      color: var(--ink);
    }

    code {
      font-family: Consolas, "Cascadia Code", monospace;
      background: #eef2f7;
      border: 1px solid #d8dee8;
      border-radius: 5px;
      padding: 1px 5px;
      font-size: 0.92em;
    }

    pre {
      overflow-x: auto;
      background: #fbfcff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      margin: 12px 0;
    }

    pre code {
      display: block;
      background: transparent;
      border: 0;
      padding: 0;
      white-space: pre;
    }

    .tag {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      margin-left: 8px;
      vertical-align: middle;
    }

    .tag.important { color: var(--accent); background: var(--accent-soft); border: 1px solid #ffd3d3; }
    .tag.know { color: var(--green); background: #eefaf5; border: 1px solid #cdeedf; }
    .tag.pass { color: var(--blue); background: #eef4ff; border: 1px solid #d5e2ff; }

    table {
      width: 100%;
      border-collapse: collapse;
      margin: 14px 0;
      overflow: hidden;
      border-radius: 8px;
      border: 1px solid var(--line);
      display: table;
    }

    th, td {
      border: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }

    th {
      background: #f1f4f9;
      font-weight: 700;
    }

    .formula {
      overflow-x: auto;
      background: #fbfcff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      margin: 12px 0;
      text-align: center;
    }

    details {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      margin: 14px 0;
      background: #fcfdff;
    }

    summary {
      cursor: pointer;
      font-weight: 700;
      color: var(--nav);
    }

    .footer {
      color: var(--muted);
      font-size: 13px;
      padding: 18px 4px 4px;
      text-align: center;
    }

    @media (max-width: 900px) {
      .layout { grid-template-columns: 1fr; }
      aside {
        position: relative;
        height: auto;
        padding: 18px;
      }
      nav {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 6px;
      }
      main { padding: 16px; }
      h1 { font-size: 28px; }
      h2 { font-size: 22px; }
      table { display: block; overflow-x: auto; }
    }
"""


def main() -> None:
    md = MD_PATH.read_text(encoding="utf-8")
    body, headings = convert_markdown(md)
    nav = build_nav(headings)
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>大语言模型复习笔记</title>
  <script>
    window.MathJax = {{
      tex: {{ inlineMath: [['$', '$']], displayMath: [['$$', '$$']] }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script>
    window.addEventListener('DOMContentLoaded', () => {{
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js';
      script.async = true;
      document.head.appendChild(script);
    }});
  </script>
  <style>{STYLE}</style>
</head>
<body>
  <div class="layout">
    <aside>
      <div class="brand">
        <h1>大语言模型</h1>
        <p>第十至十二章复习笔记</p>
      </div>
      <nav id="toc">
        {nav}
      </nav>
    </aside>
    <main>
{body}
      <div class="footer">由 large-language-models-notes.md 生成，可直接上传到 GitHub Pages。</div>
    </main>
  </div>
  <script>
    const links = Array.from(document.querySelectorAll('nav a'));
    const sections = links.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);
    const activate = () => {{
      let current = sections[0];
      for (const section of sections) {{
        if (section.getBoundingClientRect().top <= 120) current = section;
      }}
      links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + current.id));
    }};
    document.addEventListener('scroll', activate, {{ passive: true }});
    activate();
  </script>
</body>
</html>
"""
    OUT_PATH.write_text(html_text, encoding="utf-8", newline="\n")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
