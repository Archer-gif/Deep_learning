from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "index.html"


def extract_main(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"<main>\s*(.*?)\s*</main>", text, re.S)
    if not match:
        raise RuntimeError(f"Could not find <main> in {path.name}")
    body = match.group(1)
    body = re.sub(r'\s*<div class="footer">.*?</div>\s*', "\n", body, flags=re.S)
    return body.strip()


def prefix_fragment(html: str, prefix: str) -> str:
    html = re.sub(r'id="([^"]+)"', lambda m: f'id="{prefix}-{m.group(1)}"', html)
    html = re.sub(r'href="#([^"]+)"', lambda m: f'href="#{prefix}-{m.group(1)}"', html)
    return html


def page() -> str:
    machine = prefix_fragment(extract_main(ROOT / "machine-learning-foundations.html"), "ml")
    neural = prefix_fragment(extract_main(ROOT / "neural-network-foundations.html"), "nn")
    representation = prefix_fragment(extract_main(ROOT / "representation-learning.html"), "rep")
    deep_vision = prefix_fragment(extract_main(ROOT / "deep-vision-models.html"), "dv")
    generative = prefix_fragment(extract_main(ROOT / "generative-models.html"), "gen")
    llm = prefix_fragment(extract_main(ROOT / "large-language-models.html"), "llm")

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>深度学习导论课程笔记</title>
  <script>
    window.MathJax = {{
      tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']] }},
      svg: {{ fontCache: 'global' }}
    }};
    window.addEventListener('DOMContentLoaded', () => {{
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js';
      script.async = true;
      document.head.appendChild(script);
    }});
  </script>
  <style>
    :root {{
      --bg: #f7f8fb;
      --paper: #ffffff;
      --ink: #27242a;
      --muted: #746d78;
      --line: #e4e0e8;
      --nav: #2b9f9b;
      --nav-dark: #167774;
      --accent: #31b8ff;
      --accent-soft: #eef9ff;
      --red: #c62828;
      --green: #17805b;
      --blue: #246bfe;
      --shadow: 0 14px 34px rgba(29, 45, 68, 0.10);
    }}

    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", Arial, sans-serif;
      line-height: 1.72;
    }}

    .topbar {{
      position: sticky;
      top: 0;
      z-index: 10;
      height: 64px;
      display: grid;
      grid-template-columns: minmax(180px, 1fr) minmax(240px, 520px) minmax(160px, 1fr);
      align-items: center;
      gap: 18px;
      padding: 0 9vw;
      color: #fff;
      background: var(--nav);
      box-shadow: 0 2px 12px rgba(0,0,0,0.12);
    }}

    .brand {{
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }}

    .brand-mark {{
      width: 34px;
      height: 34px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      background: rgba(255,255,255,0.18);
      font-weight: 800;
    }}

    .brand h1 {{
      margin: 0;
      font-size: 24px;
      line-height: 1.1;
      letter-spacing: 0;
      white-space: nowrap;
    }}

    .search {{
      height: 42px;
      border-radius: 4px;
      border: 0;
      background: rgba(0,0,0,0.24);
      color: #fff;
      padding: 0 14px;
      font-size: 15px;
      outline: none;
    }}

    .search::placeholder {{ color: rgba(255,255,255,0.72); }}

    .repo-link {{
      justify-self: end;
      color: #fff;
      text-decoration: none;
      font-weight: 700;
      border: 1px solid rgba(255,255,255,0.35);
      border-radius: 6px;
      padding: 7px 10px;
    }}

    .layout {{
      display: grid;
      grid-template-columns: 280px minmax(0, 860px) 250px;
      gap: 36px;
      max-width: 1460px;
      margin: 0 auto;
      padding: 54px 28px 72px;
    }}

    aside,
    .toc {{
      position: sticky;
      top: 88px;
      align-self: start;
      max-height: calc(100vh - 110px);
      overflow: auto;
    }}

    aside {{
      border-right: 1px solid var(--line);
      padding-right: 18px;
    }}

    .side-title,
    .toc-title {{
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 15px;
      font-weight: 800;
    }}

    details.chapter {{
      border-radius: 6px;
      margin: 6px 0;
    }}

    details.chapter > summary {{
      cursor: pointer;
      list-style: none;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      min-height: 32px;
      padding: 4px 6px;
      border-radius: 5px;
      color: #4c4750;
      font-weight: 700;
    }}

    details.chapter > summary::-webkit-details-marker {{ display: none; }}
    details.chapter > summary:hover {{ background: #edf4f4; }}
    details.chapter > summary::after {{ content: ">"; color: #8a838d; }}
    details.chapter[open] > summary::after {{ content: "⌄"; }}

    .chapter-links {{
      display: grid;
      gap: 2px;
      padding: 2px 0 8px 14px;
    }}

    .chapter-links a,
    .toc a {{
      color: #5c5660;
      text-decoration: none;
      border-radius: 5px;
    }}

    .chapter-links a {{
      padding: 4px 8px;
      font-size: 14px;
    }}

    .chapter-links a:hover,
    .chapter-links a.active,
    .toc a:hover,
    .toc a.active {{
      color: var(--accent);
      background: var(--accent-soft);
    }}

    main {{
      min-width: 0;
    }}

    .course-divider {{
      margin: 0 0 24px;
      padding: 12px 16px;
      border-radius: 8px;
      color: #fff;
      background: var(--nav-dark);
      font-size: 20px;
      font-weight: 800;
      letter-spacing: 0;
    }}

    .hero,
    .section {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(20, 33, 61, 0.06);
      padding: 24px;
      margin-bottom: 18px;
    }}

    .hero {{
      background: linear-gradient(120deg, #ffffff 0%, #f4f8ff 100%);
      box-shadow: var(--shadow);
      padding: 28px;
      margin-bottom: 22px;
    }}

    h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      line-height: 1.16;
      letter-spacing: 0;
    }}

    h2 {{
      margin: 0 0 16px;
      font-size: 26px;
      line-height: 1.25;
      letter-spacing: 0;
    }}

    h3 {{
      margin: 24px 0 10px;
      font-size: 19px;
      line-height: 1.32;
      letter-spacing: 0;
    }}

    h4 {{
      margin: 18px 0 8px;
      font-size: 16px;
      line-height: 1.35;
      letter-spacing: 0;
    }}

    p {{ margin: 10px 0; }}
    ul, ol {{ padding-left: 22px; }}
    li {{ margin: 4px 0; }}

    blockquote,
    .note {{
      margin: 14px 0;
      border-left: 4px solid var(--red);
      background: #fff1f1;
      padding: 12px 14px;
      border-radius: 0 8px 8px 0;
      color: var(--ink);
    }}

    code {{
      font-family: Consolas, "Cascadia Code", monospace;
      background: #eef2f7;
      border: 1px solid #d8dee8;
      border-radius: 5px;
      padding: 1px 5px;
      font-size: 0.92em;
    }}

    pre {{
      overflow-x: auto;
      background: #fbfcff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      margin: 12px 0;
    }}

    pre code {{
      display: block;
      background: transparent;
      border: 0;
      padding: 0;
      white-space: pre;
    }}

    .tag {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      margin-left: 8px;
      vertical-align: middle;
      color: #5a6472;
      background: #f2f5f8;
      border: 1px solid #dde4ec;
    }}

    .tag.important {{ color: var(--red); background: #fff1f1; border-color: #ffd3d3; }}
    .tag.know {{ color: var(--green); background: #eefaf5; border-color: #cdeedf; }}
    .tag.pass {{ color: var(--blue); background: #eef4ff; border-color: #d5e2ff; }}

    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 14px 0;
      overflow: hidden;
      border-radius: 8px;
      border: 1px solid var(--line);
      display: table;
    }}

    th, td {{
      border: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }}

    th {{
      background: #f1f4f9;
      font-weight: 700;
    }}

    .formula {{
      overflow-x: auto;
      background: #fbfcff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      margin: 12px 0;
      text-align: center;
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}

    .mini {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fff;
    }}

    .mini h3 {{ margin-top: 0; }}

    details:not(.chapter) {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      margin: 14px 0;
      background: #fcfdff;
    }}

    details:not(.chapter) summary {{
      cursor: pointer;
      font-weight: 700;
      color: #14213d;
    }}

    .toc {{
      padding-left: 4px;
    }}

    .toc nav {{
      display: grid;
      gap: 8px;
    }}

    .toc a {{
      display: block;
      padding: 4px 8px;
      font-size: 14px;
    }}

    .footer {{
      margin-top: 30px;
      color: var(--muted);
      font-size: 13px;
      text-align: center;
    }}

    @media (max-width: 1050px) {{
      .topbar {{
        grid-template-columns: 1fr;
        height: auto;
        padding: 14px 18px;
      }}
      .repo-link {{ justify-self: start; }}
      .layout {{
        grid-template-columns: 1fr;
        padding: 28px 16px 54px;
      }}
      aside,
      .toc {{
        position: static;
        max-height: none;
        border: 0;
        padding: 0;
      }}
      h1 {{ font-size: 28px; }}
      h2 {{ font-size: 22px; }}
      table {{ display: block; overflow-x: auto; }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand">
      <div class="brand-mark">DL</div>
      <h1>深度学习导论笔记</h1>
    </div>
    <input class="search" id="search" type="search" placeholder="搜索章节、知识点">
    <a class="repo-link" href="https://github.com/Archer-gif/Deep_learning" target="_blank" rel="noreferrer">GitHub</a>
  </header>

  <div class="layout">
    <aside>
      <p class="side-title">课程笔记</p>
      <details class="chapter" open>
        <summary>机器学习基础</summary>
        <div class="chapter-links">
          <a href="#ml-overview">总览</a>
          <a href="#ml-definition">机器学习定义</a>
          <a href="#ml-linear">线性回归</a>
          <a href="#ml-pac">可学习性与 PAC</a>
          <a href="#ml-vc">VC 维</a>
          <a href="#ml-rad">Rademacher 复杂度</a>
          <a href="#ml-generalization">泛化与 ERM</a>
          <a href="#ml-fit">过拟合与正则化</a>
          <a href="#ml-logistic">逻辑回归与分类</a>
          <a href="#ml-pca">PCA</a>
          <a href="#ml-kmeans">k-means</a>
        </div>
      </details>
      <details class="chapter" open>
        <summary>神经网络基础</summary>
        <div class="chapter-links">
          <a href="#nn-overview">总览</a>
          <a href="#nn-mlp">MLP 基本结构</a>
          <a href="#nn-activation">激活函数</a>
          <a href="#nn-loss">损失函数</a>
          <a href="#nn-backprop">梯度与反向传播</a>
          <a href="#nn-data">数据准备</a>
          <a href="#nn-dropout">Dropout</a>
          <a href="#nn-normalization">归一化层</a>
          <a href="#nn-initialization">初始化</a>
          <a href="#nn-optimizers">优化器</a>
          <a href="#nn-weight-decay">权重衰减</a>
          <a href="#nn-lr">学习率技巧</a>
        </div>
      </details>
      <details class="chapter" open>
        <summary>表示学习</summary>
        <div class="chapter-links">
          <a href="#rep-representation-overview">总览</a>
          <a href="#rep-pretext">Pretext 任务</a>
          <a href="#rep-contrastive-learning">对比学习</a>
          <a href="#rep-simclr-architecture">SimCLR 结构</a>
          <a href="#rep-simclr-loss">NT-Xent 损失</a>
          <a href="#rep-skip-gram">Word2Vec</a>
          <a href="#rep-language-self-supervision">语言自监督</a>
        </div>
      </details>
      <details class="chapter" open>
        <summary>深度视觉模型</summary>
        <div class="chapter-links">
          <a href="#dv-overview">总览</a>
          <a href="#dv-generative-goal">生成目标</a>
          <a href="#dv-vae-motivation">VAE 动机</a>
          <a href="#dv-vae-math">VAE 数学原理</a>
          <a href="#dv-vae-loss">VAE 损失</a>
          <a href="#dv-diffusion-overview">Diffusion 总览</a>
          <a href="#dv-diffusion-training">训练流程</a>
        </div>
      </details>
      <details class="chapter" open>
        <summary>生成式模型</summary>
        <div class="chapter-links">
          <a href="#gen-overview">总览</a>
          <a href="#gen-vae-motivation">VAE 动机</a>
          <a href="#gen-vae-math">VAE 数学原理</a>
          <a href="#gen-reparameterization">重参数化</a>
          <a href="#gen-gan">GAN</a>
          <a href="#gen-diffusion-overview">Diffusion</a>
          <a href="#gen-latent-diffusion">Stable Diffusion</a>
        </div>
      </details>
      <details class="chapter" open>
        <summary>大语言模型</summary>
        <div class="chapter-links">
          <a href="#llm-overview">总览</a>
          <a href="#llm-architectures">三种架构</a>
          <a href="#llm-encoder-only">Encoder-only</a>
          <a href="#llm-decoder-only">Decoder-only</a>
          <a href="#llm-encoder-decoder">Encoder-Decoder</a>
          <a href="#llm-attention-types">三种注意力</a>
          <a href="#llm-rag-why">RAG</a>
          <a href="#llm-rag-when-where-how">何时何处增强</a>
        </div>
      </details>
    </aside>

    <main id="content">
      <div class="course-divider">机器学习基础</div>
{machine}

      <div class="course-divider" id="neural-network-course">神经网络基础</div>
{neural}

      <div class="course-divider" id="representation-course">表示学习</div>
{representation}

      <div class="course-divider" id="deep-vision-course">深度视觉模型</div>
{deep_vision}

      <div class="course-divider" id="generative-course">生成式模型</div>
{generative}

      <div class="course-divider" id="large-language-models-course">大语言模型</div>
{llm}

      <div class="footer">所有正文已合并到 index.html；GitHub Pages 只访问首页即可阅读完整笔记。</div>
    </main>

    <div class="toc">
      <p class="toc-title">目录</p>
      <nav>
        <a href="#ml-overview">机器学习基础</a>
        <a href="#ml-linear">线性回归</a>
        <a href="#ml-logistic">分类</a>
        <a href="#nn-overview">神经网络基础</a>
        <a href="#nn-loss">损失函数</a>
        <a href="#nn-backprop">反向传播</a>
        <a href="#nn-optimizers">优化器</a>
        <a href="#rep-representation-overview">表示学习</a>
        <a href="#dv-overview">深度视觉模型</a>
        <a href="#gen-overview">生成式模型</a>
        <a href="#llm-overview">大语言模型</a>
        <a href="#llm-attention-types">交叉注意力</a>
        <a href="#llm-rag-why">RAG</a>
      </nav>
    </div>
  </div>

  <script>
    const searchInput = document.getElementById('search');
    const searchableSections = Array.from(document.querySelectorAll('main section, main .course-divider'));
    searchInput.addEventListener('input', () => {{
      const keyword = searchInput.value.trim().toLowerCase();
      searchableSections.forEach(section => {{
        const hit = section.innerText.toLowerCase().includes(keyword);
        section.style.display = hit || !keyword ? '' : 'none';
      }});
    }});

    const links = Array.from(document.querySelectorAll('aside a, .toc a'));
    const targets = links.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);
    const activate = () => {{
      let current = targets[0];
      for (const target of targets) {{
        if (target.getBoundingClientRect().top <= 120) current = target;
      }}
      links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + current.id));
    }};
    document.addEventListener('scroll', activate, {{ passive: true }});
    activate();
  </script>
</body>
</html>
"""


def main() -> None:
    OUT_PATH.write_text(page(), encoding="utf-8", newline="\n")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
