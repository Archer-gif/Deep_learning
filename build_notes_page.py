from __future__ import annotations

import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MD_PATH = ROOT / "machine-learning-foundations-notes.md"
OLD_HTML_PATH = ROOT / "index.html"
OUT_PATH = ROOT / "index-from-markdown.html"


PRIORITY_CLASS = {
    "重点": "important",
    "了解": "know",
    "带过": "pass",
}


def split_priority(title: str) -> tuple[str, str | None]:
    match = re.search(r"【(重点|了解|带过)】\s*$", title)
    if not match:
        return title.strip(), None
    return title[: match.start()].strip(), match.group(1)


def title_html(title: str) -> str:
    clean, priority = split_priority(title)
    rendered = inline_md(clean)
    if priority:
        rendered += f' <span class="tag {PRIORITY_CLASS[priority]}">{priority}</span>'
    return rendered


def nav_title(title: str) -> str:
    clean, _ = split_priority(title)
    return re.sub(r"^\d+(?:\.\d+)*\s*[.、]?\s*", "", clean).strip()


def slugify(title: str, index: int) -> str:
    cleaned = nav_title(re.sub(r"<[^>]+>", "", title))
    mapping = {
        "总览": "overview",
        "机器学习的定义": "definition",
        "线性回归": "linear",
        "可学习性、假设空间与 PAC": "pac",
        "复杂度：VC 维": "vc",
        "复杂度：Rademacher 复杂度": "rad",
        "泛化上界与经验风险最小化": "generalization",
        "过拟合、欠拟合与没有免费午餐": "fit",
        "正则化：岭回归与 Lasso": "regularization",
        "监督学习：二分类、逻辑回归、多分类": "logistic",
        "无监督学习：PCA": "pca",
        "无监督学习：k-means": "kmeans",
        "了解内容": "survey",
        "网页怎么做": "build",
    }
    return mapping.get(cleaned, f"section-{index}")


def inline_md(text: str) -> str:
    placeholders: list[str] = []

    def keep_math(match: re.Match[str]) -> str:
        placeholders.append(match.group(0))
        return f"@@MATH{len(placeholders)-1}@@"

    text = re.sub(r"\\\(.+?\\\)", keep_math, text)
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    for i, value in enumerate(placeholders):
        text = text.replace(f"@@MATH{i}@@", value)
    return text


def render_formula(lines: list[str]) -> str:
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
    out = [f"<{tag}>"]
    pattern = r"^\s*\d+\.\s+" if ordered else r"^\s*-\s+"
    for raw in lines:
        item = re.sub(pattern, "", raw).strip()
        out.append(f"<li>{inline_md(item)}</li>")
    out.append(f"</{tag}>")
    return "\n".join(out)


def convert_blocks(lines: list[str]) -> str:
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        if line.strip() == r"\[":
            formula = [line]
            i += 1
            while i < len(lines):
                formula.append(lines[i])
                if lines[i].strip() == r"\]":
                    i += 1
                    break
                i += 1
            out.append(render_formula(formula))
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
            and lines[i].strip() != r"\["
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

        hmatch = re.match(r"^(#{2,3})\s+(.+)$", line)
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
                out.append(f"<h3>{title_html(title)}</h3>")
            i += 1
            continue

        if line.startswith("> "):
            quote_lines = []
            while i < len(lines) and lines[i].startswith("> "):
                quote_lines.append(lines[i][2:].strip())
                i += 1
            out.append(f"<blockquote>{' '.join(inline_md(q) for q in quote_lines)}</blockquote>")
            continue

        if line.strip() == r"\[":
            formula = [line]
            i += 1
            while i < len(lines):
                formula.append(lines[i])
                if lines[i].strip() == r"\]":
                    i += 1
                    break
                i += 1
            out.append(render_formula(formula))
            continue

        if line.strip().startswith("|") and i + 1 < len(lines) and re.match(
            r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", lines[i + 1]
        ):
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
            and not re.match(r"^(#{1,3})\s+", lines[i])
            and not lines[i].startswith("> ")
            and not lines[i].startswith("<details>")
            and not re.match(r"^\s*(-|\d+\.)\s+", lines[i])
            and not lines[i].strip().startswith("|")
            and lines[i].strip() != r"\["
        ):
            para.append(lines[i].strip())
            i += 1
        out.append(f"<p>{inline_md(' '.join(para))}</p>")

    if in_section:
        out.append("</section>")
    return "\n".join(out), headings


def extract_style(old_html: str) -> str:
    match = re.search(r"<style>(.*?)</style>", old_html, re.S)
    if not match:
        raise RuntimeError("Could not find style block in index.html")
    style = match.group(1)
    style += """

    .markdown-body h1 {
      margin: 0 0 10px;
      font-size: 34px;
      line-height: 1.16;
      letter-spacing: 0;
    }

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

    .formula {
      text-align: center;
    }
"""
    return style


def build_nav(headings: list[tuple[int, str, str]]) -> str:
    links = []
    for level, title, sid in headings:
        if level == 2 or sid == "overview":
            links.append(f'<a href="#{sid}">{inline_md(nav_title(title))}</a>')
    return "\n        ".join(links)


def main() -> None:
    md = MD_PATH.read_text(encoding="utf-8")
    old_html = OLD_HTML_PATH.read_text(encoding="utf-8")
    body, headings = convert_markdown(md)
    style = extract_style(old_html)
    nav = build_nav(headings)
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>机器学习基础复习笔记 - Markdown 同步版</title>
  <script>
    window.MathJax = {{
      tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']] }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  <style>{style}</style>
</head>
<body>
  <div class="layout">
    <aside>
      <div class="brand">
        <h1>机器学习基础</h1>
        <p>Markdown 同步网页</p>
      </div>
      <nav id="toc">
        {nav}
      </nav>
    </aside>
    <main class="markdown-body">
{body}
      <div class="footer">此页面由 machine-learning-foundations-notes.md 自动生成，内容与 Markdown 保持一致。</div>
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
