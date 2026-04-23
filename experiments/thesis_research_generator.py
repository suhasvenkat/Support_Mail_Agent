#!/usr/bin/env python3
"""
================================================================================
  THESIS RESEARCH GENERATOR
  Intelligent Email Triage and Automated Response Generation
  Using RAG-Augmented LLM Agents: A Multi-Node LangGraph Architecture
  
  Student: Suhas Venkat | M.Sc. Data Science | University of Europe
  Supervisor: Prof. Dr. Raja Hashim Ali
  Target Journal: Expert Systems with Applications (Elsevier, IF ~8.5)

  USAGE (from project root):
      python thesis_research_generator.py

  OUTPUT:
      ./thesis_outputs/
          figures/   — 16 publication-quality PNG figures (300 DPI)
          tables/    — 12 CSV tables with dummy data
          latex/     — LaTeX snippets for each figure/table
          report/    — Full thesis proposal text sections as .txt files
================================================================================
"""

import os
import sys
import json
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# ── Output dirs ──────────────────────────────────────────────────────────────
BASE_OUT   = os.path.join(os.path.dirname(__file__), "thesis_outputs")
FIG_DIR    = os.path.join(BASE_OUT, "figures")
TAB_DIR    = os.path.join(BASE_OUT, "tables")
LATEX_DIR  = os.path.join(BASE_OUT, "latex")
REPORT_DIR = os.path.join(BASE_OUT, "report")
for d in [FIG_DIR, TAB_DIR, LATEX_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Global style ─────────────────────────────────────────────────────────────
JOURNAL_STYLE = {
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
}
plt.rcParams.update(JOURNAL_STYLE)

COLORS = {
    "proposed":  "#1a5276",
    "bert":      "#2e86c1",
    "gpt4":      "#5dade2",
    "keyword":   "#aed6f1",
    "accent":    "#c0392b",
    "neutral":   "#566573",
    "light":     "#d6eaf8",
    "green":     "#1e8449",
    "orange":    "#d68910",
    "purple":    "#6c3483",
}

def save_fig(fig, name, caption=""):
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    # Write LaTeX snippet
    latex = f"""\\begin{{figure}}[htbp]
  \\centering
  \\includegraphics[width=\\linewidth]{{figures/{name}.png}}
  \\caption{{{caption}}}
  \\label{{fig:{name}}}
\\end{{figure}}"""
    with open(os.path.join(LATEX_DIR, f"{name}.tex"), "w") as f:
        f.write(latex)
    print(f"  ✓ Figure saved: {name}.png")
    return path

def save_table(df, name, caption=""):
    path = os.path.join(TAB_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    # LaTeX table
    latex = df.to_latex(index=False, caption=caption, label=f"tab:{name}",
                        escape=True, column_format='l' + 'c' * (len(df.columns)-1))
    with open(os.path.join(LATEX_DIR, f"{name}.tex"), "w") as f:
        f.write(latex)
    print(f"  ✓ Table saved:  {name}.csv")
    return path

np.random.seed(42)

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — System Architecture Diagram (RQ1)
# ════════════════════════════════════════════════════════════════════════════
def fig_architecture():
    print("\n[Fig 1] System Architecture Diagram")
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')

    def box(ax, x, y, w, h, label, sublabel, color, textcolor='white'):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.62, label, ha='center', va='center',
                fontsize=11, fontweight='bold', color=textcolor,
                fontfamily='serif')
        ax.text(x + w/2, y + h*0.28, sublabel, ha='center', va='center',
                fontsize=8.5, color=textcolor, alpha=0.9, fontfamily='serif')

    def arrow(ax, x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=COLORS['neutral'],
                                   lw=1.8, mutation_scale=18))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my+0.15, label, ha='center', fontsize=8,
                    color=COLORS['neutral'], style='italic')

    # Input
    box(ax, 0.4, 3.2, 2.0, 1.6, "📧 EMAIL IN", "sender · subject · body",
        COLORS['neutral'], 'white')

    # Nodes
    nodes = [
        (3.0, 3.0, "① CLASSIFIER", "Intent · Category\nConfidence score",   COLORS['proposed']),
        (5.6, 3.0, "② KB RETRIEVER","FAISS / ChromaDB\nTop-k documents",     COLORS['bert']),
        (8.2, 3.0, "③ RESPONDER",   "LLM + RAG context\nDraft generation",   COLORS['green']),
        (10.8, 3.0, "④ ESCALATOR",  "Confidence check\nRouting decision",    COLORS['orange']),
    ]
    for x, y, label, sub, color in nodes:
        box(ax, x, y, 2.2, 2.0, label, sub, color)

    # Arrows between nodes
    arrow(ax, 2.4, 4.0, 3.0, 4.0, "email state")
    arrow(ax, 5.2, 4.0, 5.6, 4.0, "+ intent")
    arrow(ax, 7.8, 4.0, 8.2, 4.0, "+ docs")
    arrow(ax, 10.4, 4.0, 10.8, 4.0, "+ draft")

    # Outputs
    box(ax, 12.0, 5.0, 1.6, 1.2, "✅ AUTO\nREPLY", "email sent", COLORS['green'])
    box(ax, 12.0, 2.8, 1.6, 1.2, "🚨 ESCALATE\nTO HUMAN", "ticket raised", COLORS['accent'])
    arrow(ax, 13.0, 4.0, 13.0, 5.0)
    arrow(ax, 13.0, 4.0, 13.0, 3.4)

    # Knowledge Base bubble
    kb_rect = FancyBboxPatch((5.4, 0.5), 2.6, 1.5, boxstyle="round,pad=0.1",
                             facecolor='#fef9e7', edgecolor=COLORS['orange'], lw=1.5, linestyle='--')
    ax.add_patch(kb_rect)
    ax.text(6.7, 1.25, "📚 Knowledge Base", ha='center', fontsize=9.5,
            fontweight='bold', color=COLORS['orange'])
    ax.text(6.7, 0.85, "FAQ docs · product manuals · policies", ha='center',
            fontsize=8, color=COLORS['neutral'])
    ax.annotate("", xy=(6.7, 3.0), xytext=(6.7, 2.0),
                arrowprops=dict(arrowstyle="<|-", color=COLORS['orange'],
                                lw=1.5, linestyle='dashed', mutation_scale=14))

    # LangGraph StateGraph label
    sg_rect = FancyBboxPatch((2.8, 2.6), 8.4, 2.8, boxstyle="round,pad=0.15",
                             facecolor='none', edgecolor=COLORS['proposed'],
                             lw=1.5, linestyle=':', alpha=0.6)
    ax.add_patch(sg_rect)
    ax.text(7.0, 5.6, "LangGraph StateGraph Orchestration", ha='center',
            fontsize=9, color=COLORS['proposed'], style='italic')

    ax.set_title("Fig. 1 — Multi-Node LangGraph Agentic Architecture for Customer Support Email Automation",
                 fontsize=12, fontweight='bold', pad=14, color='#1a1a2e')

    caption = ("Fig. 1. Overview of the proposed multi-node LangGraph agentic architecture comprising "
               "four functional nodes: (1) Intent Classifier, (2) Knowledge Base Retriever, "
               "(3) Response Generator, and (4) Confidence-Aware Escalator. Dashed border denotes "
               "the LangGraph StateGraph boundary. The KB is queried by the Retriever node via "
               "FAISS semantic search.")
    save_fig(fig, "fig01_architecture", caption)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Baseline Comparison Bar Chart (RQ1)
# ════════════════════════════════════════════════════════════════════════════
def fig_baseline_comparison():
    print("\n[Fig 2] Baseline Comparison")
    systems  = ["Keyword\nBaseline", "Fine-tuned\nBERT", "Vanilla\nGPT-4", "Proposed\n(LangGraph-RAG)"]
    metrics  = ["Intent F1", "Response BLEU-4", "BERTScore F1", "Escalation F1"]
    data = np.array([
        [0.61, 0.18, 0.72, 0.54],
        [0.79, 0.27, 0.81, 0.68],
        [0.83, 0.34, 0.86, 0.71],
        [0.91, 0.43, 0.92, 0.84],
    ])
    colors_bar = [COLORS['keyword'], COLORS['bert'], COLORS['gpt4'], COLORS['proposed']]
    x   = np.arange(len(metrics))
    w   = 0.18
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (sys_label, row, c) in enumerate(zip(systems, data, colors_bar)):
        bars = ax.bar(x + i*w, row, w, label=sys_label, color=c,
                      edgecolor='white', linewidth=0.8)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha='center', va='bottom',
                    fontsize=7.5, color='#333')

    ax.set_xticks(x + 1.5*w)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xlabel("Evaluation Metric", fontsize=11)
    ax.set_title("Fig. 2 — Performance Comparison: Proposed System vs. Baselines", fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9, ncol=2)
    # Highlight proposed
    for pos in x + 3*w:
        ax.axvspan(pos - w*0.55, pos + w*0.55, alpha=0.06, color=COLORS['proposed'])

    caption = ("Fig. 2. Comparative evaluation of the proposed LangGraph-RAG system against three baselines "
               "— keyword-based classifier, fine-tuned BERT pipeline, and vanilla GPT-4 — across four "
               "metrics on the curated support email benchmark (n=1,200). The proposed system achieves "
               "consistent improvements across all dimensions.")
    save_fig(fig, "fig02_baseline_comparison", caption)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — RAG Config Heatmap (RQ2)
# ════════════════════════════════════════════════════════════════════════════
def fig_rag_heatmap():
    print("\n[Fig 3] RAG Configuration Heatmap")
    chunks  = [128, 256, 512]
    topks   = [1, 3, 5, 7]
    # FAISS faithfulness scores
    faiss_data = np.array([
        [0.62, 0.71, 0.75, 0.73],
        [0.74, 0.81, 0.84, 0.82],
        [0.69, 0.77, 0.79, 0.78],
    ])
    chroma_data = np.array([
        [0.65, 0.74, 0.78, 0.76],
        [0.77, 0.83, 0.87, 0.85],
        [0.71, 0.79, 0.82, 0.80],
    ])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = LinearSegmentedColormap.from_list("thesis", ["#d6eaf8", "#1a5276"], N=256)

    for ax, data, title in zip(axes, [faiss_data, chroma_data], ["FAISS", "ChromaDB"]):
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0.58, vmax=0.92)
        ax.set_xticks(range(len(topks)));  ax.set_xticklabels([f"k={k}" for k in topks])
        ax.set_yticks(range(len(chunks))); ax.set_yticklabels([f"{c} tok" for c in chunks])
        ax.set_xlabel("Top-k Retrieved Documents", fontsize=11)
        ax.set_ylabel("Chunk Size (tokens)", fontsize=11)
        ax.set_title(f"{title} — RAGAS Faithfulness", fontweight='bold')
        for i in range(len(chunks)):
            for j in range(len(topks)):
                v = data[i, j]
                color = 'white' if v > 0.78 else '#1a1a2e'
                ax.text(j, i, f"{v:.2f}", ha='center', va='center',
                        fontsize=11, color=color, fontweight='bold')

    fig.colorbar(im, ax=axes, label="RAGAS Faithfulness Score", shrink=0.8, pad=0.02)
    fig.suptitle("Fig. 3 — RAG Configuration Grid: Faithfulness vs. Chunk Size × Top-k",
                 fontweight='bold', fontsize=12)

    caption = ("Fig. 3. Heatmap of RAGAS faithfulness scores across 12 retrieval configurations "
               "(chunk size ∈ {128, 256, 512}; top-k ∈ {1, 3, 5, 7}) for FAISS and ChromaDB backends. "
               "Optimal configuration (chunk=256, k=5, ChromaDB) achieves faithfulness=0.87.")
    save_fig(fig, "fig03_rag_heatmap", caption)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — RAGAS Grouped Bar (RQ2)
# ════════════════════════════════════════════════════════════════════════════
def fig_ragas_comparison():
    print("\n[Fig 4] RAGAS Metric Comparison")
    dims   = ["Answer\nFaithfulness", "Answer\nRelevancy", "Context\nPrecision", "Context\nRecall"]
    faiss  = [0.84, 0.81, 0.79, 0.77]
    chroma = [0.87, 0.83, 0.82, 0.80]
    x = np.arange(len(dims)); w = 0.30
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w/2, faiss,  w, label="FAISS",    color=COLORS['bert'],     edgecolor='white')
    b2 = ax.bar(x + w/2, chroma, w, label="ChromaDB", color=COLORS['proposed'], edgecolor='white')
    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(dims)
    ax.set_ylim(0.70, 0.96); ax.set_ylabel("RAGAS Score")
    ax.set_xlabel("Evaluation Dimension")
    ax.set_title("Fig. 4 — RAGAS Evaluation: FAISS vs. ChromaDB (chunk=256, k=5)", fontweight='bold')
    ax.legend(framealpha=0.9)
    ax.axhline(y=0.80, color=COLORS['accent'], linestyle=':', lw=1.2, alpha=0.6,
               label="Acceptance threshold")

    caption = ("Fig. 4. Comparison of four RAGAS evaluation dimensions for FAISS vs. ChromaDB "
               "under optimal configuration (chunk=256, top-k=5). ChromaDB consistently outperforms "
               "FAISS across all dimensions, with faithfulness gain of +0.03.")
    save_fig(fig, "fig04_ragas_comparison", caption)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Faithfulness vs Response Length Scatter (RQ2)
# ════════════════════════════════════════════════════════════════════════════
def fig_faithfulness_scatter():
    print("\n[Fig 5] Faithfulness vs Response Length")
    intents = ['billing', 'technical', 'refund', 'shipping', 'account', 'general']
    colors_intent = [COLORS['proposed'], COLORS['accent'], COLORS['green'],
                     COLORS['orange'], COLORS['purple'], COLORS['neutral']]
    fig, ax = plt.subplots(figsize=(10, 6))
    for intent, c in zip(intents, colors_intent):
        n   = 60
        length = np.random.randint(60, 220, n)
        base_faith = {'billing':0.84,'technical':0.82,'refund':0.86,
                      'shipping':0.79,'account':0.83,'general':0.76}[intent]
        faith = np.clip(base_faith - 0.0015 * length + np.random.normal(0, 0.04, n), 0.5, 0.98)
        ax.scatter(length, faith, alpha=0.65, s=35, color=c, label=intent.capitalize())

    # Regression line over all
    all_len = np.random.randint(60, 220, 360)
    all_faith = np.clip(0.88 - 0.0014 * all_len + np.random.normal(0, 0.04, 360), 0.5, 0.98)
    m, b = np.polyfit(all_len, all_faith, 1)
    xl = np.linspace(55, 225, 100)
    ax.plot(xl, m*xl + b, color='black', lw=2, linestyle='--', label=f"Regression (r={-0.61:.2f})")

    ax.set_xlabel("Generated Response Length (tokens)")
    ax.set_ylabel("RAGAS Answer Faithfulness")
    ax.set_title("Fig. 5 — Faithfulness vs. Response Length by Intent Category", fontweight='bold')
    ax.legend(ncol=2, fontsize=9, framealpha=0.9)
    ax.text(185, 0.92, f"r = −0.61\np < 0.001", fontsize=9, color='black',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    caption = ("Fig. 5. Scatter plot of RAGAS Answer Faithfulness scores against generated response "
               "token length, segmented by email intent category (n=360). A negative correlation "
               "(r=−0.61, p<0.001) motivates response length constraints in the Responder node.")
    save_fig(fig, "fig05_faithfulness_scatter", caption)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — ROC Curves (RQ3)
# ════════════════════════════════════════════════════════════════════════════
def fig_roc_curves():
    print("\n[Fig 6] Escalation ROC Curves")
    from sklearn.metrics import roc_curve, auc

    def make_scores(pos_mean, neg_mean, n=200):
        pos = np.clip(np.random.normal(pos_mean, 0.12, n), 0, 1)
        neg = np.clip(np.random.normal(neg_mean, 0.12, n), 0, 1)
        y_true  = np.array([1]*n + [0]*n)
        y_score = np.concatenate([pos, neg])
        return y_true, y_score

    configs = [
        ("Static τ=0.50",         0.72, 0.42, COLORS['keyword'],  ':'),
        ("Optimal Static τ=0.65", 0.76, 0.38, COLORS['bert'],     '--'),
        ("Learned (Proposed)",    0.84, 0.28, COLORS['proposed'], '-'),
    ]
    fig, ax = plt.subplots(figsize=(8, 7))
    for label, pos_m, neg_m, c, ls in configs:
        yt, ys = make_scores(pos_m, neg_m)
        fpr, tpr, _ = roc_curve(yt, ys)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=c, lw=2.2, linestyle=ls,
                label=f"{label}  (AUC = {roc_auc:.3f})")

    ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5, label="Random classifier")
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    ax.set_xlabel("False Positive Rate (Human-Escalated Unnecessarily)")
    ax.set_ylabel("True Positive Rate (Correctly Escalated)")
    ax.set_title("Fig. 6 — ROC Curves: Escalation Decision Strategies", fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.fill_between([0,1],[0,1], alpha=0.04, color='gray')

    caption = ("Fig. 6. ROC curves comparing three escalation strategies — static threshold (τ=0.50), "
               "grid-searched optimal static threshold (τ=0.65), and the proposed learned confidence-aware "
               "escalation model — on the held-out test set. The learned model achieves the highest AUC.")
    save_fig(fig, "fig06_roc_curves", caption)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Escalation Rate vs Threshold (RQ3)
# ════════════════════════════════════════════════════════════════════════════
def fig_escalation_tradeoff():
    print("\n[Fig 7] Escalation Rate vs Threshold Trade-off")
    tau = np.linspace(0.3, 0.95, 200)
    static_rate    = np.clip(1 - tau + np.random.normal(0, 0.01, 200), 0, 1)
    learned_rate   = np.clip(0.9 - 1.2*(tau - 0.3)**1.5 + np.random.normal(0, 0.008, 200), 0, 1)
    static_acc     = np.clip(0.60 + 0.35*tau - 0.12*tau**2 + np.random.normal(0, 0.008, 200), 0, 1)
    learned_acc    = np.clip(0.68 + 0.38*tau - 0.10*tau**2 + np.random.normal(0, 0.007, 200), 0, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    ax1.plot(tau, static_rate * 100,  color=COLORS['bert'],     lw=2, linestyle='--', label='Static')
    ax1.plot(tau, learned_rate * 100, color=COLORS['proposed'], lw=2.2,               label='Learned (Proposed)')
    opt_zone = (tau > 0.58) & (tau < 0.72)
    ax1.fill_between(tau, 0, 100, where=opt_zone, alpha=0.10, color=COLORS['green'],
                     label='Optimal operating zone')
    ax1.set_xlabel("Confidence Threshold τ"); ax1.set_ylabel("Escalation Rate (%)")
    ax1.set_title("Escalation Rate vs. τ", fontweight='bold')
    ax1.legend(); ax1.set_xlim(0.3, 0.95); ax1.set_ylim(0, 80)

    ax2.plot(tau, static_acc,  color=COLORS['bert'],     lw=2, linestyle='--', label='Static')
    ax2.plot(tau, learned_acc, color=COLORS['proposed'], lw=2.2,               label='Learned (Proposed)')
    ax2.fill_between(tau, 0, 1, where=opt_zone, alpha=0.10, color=COLORS['green'])
    ax2.set_xlabel("Confidence Threshold τ"); ax2.set_ylabel("Automated Response Accuracy")
    ax2.set_title("Response Accuracy vs. τ", fontweight='bold')
    ax2.legend(); ax2.set_xlim(0.3, 0.95); ax2.set_ylim(0.55, 1.02)

    fig.suptitle("Fig. 7 — Escalation Rate and Response Accuracy Trade-off vs. Confidence Threshold",
                 fontweight='bold', fontsize=12)

    caption = ("Fig. 7. Trade-off between human escalation rate and automated response accuracy as a "
               "function of confidence threshold τ for static and learned escalation models. "
               "The shaded region indicates the optimal operating zone (τ ∈ [0.58, 0.72]) "
               "balancing automation coverage and response quality.")
    save_fig(fig, "fig07_escalation_tradeoff", caption)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Confusion Matrix (RQ3)
# ════════════════════════════════════════════════════════════════════════════
def fig_confusion_matrix():
    print("\n[Fig 8] Escalation Confusion Matrix")
    cm = np.array([[298, 34], [14, 54]])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')
    labels = ['Auto-Respond', 'Escalate-to-Human']
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Fig. 8 — Escalation Decision Confusion Matrix\n(Confidence-Aware Module, n=400)",
                 fontweight='bold')
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i,j] > 150 else '#1a1a2e'
            ax.text(j, i, f"{cm[i,j]}\n({cm[i,j]/total*100:.1f}%)",
                    ha='center', va='center', fontsize=14, color=color, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8)

    caption = ("Fig. 8. Confusion matrix of escalation decisions (Auto-Respond vs. Escalate-to-Human) "
               "produced by the confidence-aware module on 400 test emails. "
               "The model achieves precision=0.88, recall=0.79, F1=0.83.")
    save_fig(fig, "fig08_confusion_matrix", caption)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Per-Class Intent F1 (RQ4)
# ════════════════════════════════════════════════════════════════════════════
def fig_per_class_f1():
    print("\n[Fig 9] Per-Class Intent F1")
    classes  = ['Billing', 'Refund', 'Technical', 'Shipping', 'Account', 'General']
    systems  = ['Keyword', 'BERT', 'GPT-4', 'Proposed']
    data = np.array([
        [0.55, 0.62, 0.58, 0.60, 0.65, 0.67],
        [0.76, 0.80, 0.78, 0.73, 0.81, 0.79],
        [0.82, 0.85, 0.83, 0.80, 0.86, 0.82],
        [0.92, 0.94, 0.91, 0.89, 0.93, 0.88],
    ])
    colors_s = [COLORS['keyword'], COLORS['bert'], COLORS['gpt4'], COLORS['proposed']]
    x = np.arange(len(classes)); w = 0.18
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (sys, row, c) in enumerate(zip(systems, data, colors_s)):
        ax.bar(x + (i-1.5)*w, row, w, label=sys, color=c, edgecolor='white', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylim(0.45, 1.02); ax.set_ylabel("F1 Score"); ax.set_xlabel("Intent Category")
    ax.set_title("Fig. 9 — Per-Class Intent Classification F1 Across Systems", fontweight='bold')
    ax.legend(ncol=2, framealpha=0.9)

    caption = ("Fig. 9. Per-class intent classification F1 scores for six support email categories "
               "comparing keyword baseline, fine-tuned BERT, GPT-4 zero-shot, and the proposed "
               "LangGraph classifier node. The proposed system achieves consistent superiority, "
               "with 'Refund' yielding the highest F1 (0.94).")
    save_fig(fig, "fig09_per_class_f1", caption)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — Radar Chart Robustness (RQ4)
# ════════════════════════════════════════════════════════════════════════════
def fig_radar_robustness():
    print("\n[Fig 10] Radar Chart")
    dims = ['Clean\nAccuracy', 'Noise\nTolerance', 'Multilingual\nGeneralization',
            'Ambiguity\nHandling', 'OOD\nDetection']
    N = len(dims)
    systems_data = {
        'Keyword':   [0.61, 0.45, 0.32, 0.38, 0.42],
        'BERT':      [0.79, 0.68, 0.55, 0.61, 0.64],
        'GPT-4':     [0.83, 0.78, 0.74, 0.72, 0.70],
        'Proposed':  [0.91, 0.87, 0.81, 0.83, 0.79],
    }
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    colors_r = [COLORS['keyword'], COLORS['bert'], COLORS['gpt4'], COLORS['proposed']]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for (sname, vals), c in zip(systems_data.items(), colors_r):
        vals_plot = vals + vals[:1]
        ax.plot(angles, vals_plot, color=c, lw=2.2, label=sname)
        ax.fill(angles, vals_plot, color=c, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([d.replace('\n', '\n') for d in dims], fontsize=10)
    ax.set_ylim(0, 1); ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2','0.4','0.6','0.8','1.0'], fontsize=8)
    ax.set_title("Fig. 10 — Multi-Dimensional Robustness Profile", fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), framealpha=0.9)

    caption = ("Fig. 10. Radar chart comparing four classifiers across five robustness dimensions: "
               "clean accuracy, noisy-input tolerance, multilingual generalization, ambiguous-intent "
               "handling, and out-of-domain detection. Each axis is normalized to [0,1]. "
               "The proposed system achieves the largest area.")
    save_fig(fig, "fig10_radar_robustness", caption)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 11 — Correlation Heatmap Metrics vs Human (RQ5)
# ════════════════════════════════════════════════════════════════════════════
def fig_correlation_heatmap():
    print("\n[Fig 11] Metric–Human Correlation Heatmap")
    auto_metrics  = ['BLEU-4', 'ROUGE-L', 'BERTScore F1', 'RAGAS\nCorrectness']
    human_ratings = ['Fluency', 'Relevance', 'Completeness', 'Tone', 'Overall']
    corr = np.array([
        [0.54, 0.48, 0.51, 0.46, 0.52],
        [0.61, 0.57, 0.63, 0.50, 0.59],
        [0.76, 0.81, 0.74, 0.69, 0.79],
        [0.68, 0.73, 0.71, 0.64, 0.72],
    ])
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = LinearSegmentedColormap.from_list("corr", ["#fdfefe","#d6eaf8","#1a5276"], N=256)
    im = ax.imshow(corr, cmap=cmap, aspect='auto', vmin=0.40, vmax=0.90)
    ax.set_xticks(range(len(human_ratings))); ax.set_xticklabels(human_ratings, fontsize=11)
    ax.set_yticks(range(len(auto_metrics)));  ax.set_yticklabels(auto_metrics, fontsize=11)
    ax.set_xlabel("Human Rating Dimensions", fontsize=11)
    ax.set_ylabel("Automated Metrics", fontsize=11)
    ax.set_title("Fig. 11 — Pearson Correlation: Automated Metrics vs. Human Ratings (n=300)",
                 fontweight='bold')
    for i in range(len(auto_metrics)):
        for j in range(len(human_ratings)):
            v = corr[i, j]
            color = 'white' if v > 0.72 else '#1a1a2e'
            ax.text(j, i, f"{v:.2f}", ha='center', va='center',
                    fontsize=12, color=color, fontweight='bold')
    fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)

    caption = ("Fig. 11. Pearson correlation matrix between automated evaluation metrics (BLEU-4, "
               "ROUGE-L, BERTScore F1, RAGAS Answer Correctness) and human-rated response quality "
               "dimensions across 300 annotated responses. BERTScore F1 shows the highest correlation "
               "with human-rated Relevance (r=0.81).")
    save_fig(fig, "fig11_correlation_heatmap", caption)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 12 — BERTScore vs Human Relevance Scatter (RQ5)
# ════════════════════════════════════════════════════════════════════════════
def fig_bertscore_scatter():
    print("\n[Fig 12] BERTScore vs Human Relevance")
    np.random.seed(7)
    bert_scores = np.random.uniform(0.62, 0.97, 300)
    human       = np.clip(0.35 + 4.2 * (bert_scores - 0.62) + np.random.normal(0, 0.25, 300), 1, 5)
    systems_label = np.random.choice(['Keyword','BERT','GPT-4','Proposed'],
                                     size=300, p=[0.15, 0.25, 0.25, 0.35])
    colors_s = {'Keyword': COLORS['keyword'], 'BERT': COLORS['bert'],
                'GPT-4': COLORS['gpt4'], 'Proposed': COLORS['proposed']}

    fig, ax = plt.subplots(figsize=(9, 7))
    for sys in ['Keyword','BERT','GPT-4','Proposed']:
        mask = systems_label == sys
        ax.scatter(bert_scores[mask], human[mask], alpha=0.55, s=28,
                   color=colors_s[sys], label=sys)
    m, b = np.polyfit(bert_scores, human, 1)
    xl = np.linspace(0.60, 0.99, 100)
    yl = m*xl + b
    ax.plot(xl, yl, 'k--', lw=2.2, label=f'Regression (r=0.81)')
    ci = 0.3
    ax.fill_between(xl, yl - ci, yl + ci, alpha=0.10, color='gray', label='95% CI')
    ax.set_xlabel("BERTScore F1", fontsize=11)
    ax.set_ylabel("Human-Rated Relevance (1–5 Likert)", fontsize=11)
    ax.set_title("Fig. 12 — BERTScore F1 vs. Human-Rated Relevance", fontweight='bold')
    ax.legend(ncol=2, framealpha=0.9, fontsize=9)
    ax.text(0.90, 1.6, "r = 0.81\np < 0.001", fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    caption = ("Fig. 12. Scatter plot of BERTScore F1 against human-rated relevance scores for "
               "300 generated responses, with linear regression fit and 95% confidence interval. "
               "BERTScore shows the strongest correlation (r=0.81) among all automated metrics, "
               "validating its use as the primary automated proxy for response quality.")
    save_fig(fig, "fig12_bertscore_scatter", caption)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 13 — Latency vs Load (RQ6)
# ════════════════════════════════════════════════════════════════════════════
def fig_latency_load():
    print("\n[Fig 13] Latency vs Concurrent Load")
    loads = [1, 5, 10, 20, 30, 50, 75, 100]
    gpt4     = [820, 860, 940, 1150, 1480, 2200, 3400, 5100]
    gpt35    = [320, 340, 380, 470,  620,  950, 1500, 2400]
    mistral  = [180, 195, 220, 290,  390,  610, 1020, 1780]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(loads, gpt4,    'o-', color=COLORS['accent'],   lw=2.2, label='GPT-4-turbo',     ms=7)
    ax.plot(loads, gpt35,   's-', color=COLORS['bert'],     lw=2.2, label='GPT-3.5-turbo',   ms=7)
    ax.plot(loads, mistral, '^-', color=COLORS['proposed'], lw=2.2, label='Mistral-7B-Inst.', ms=7)
    ax.axhline(2000, color='gray', linestyle=':', lw=1.2, alpha=0.7, label='SLA threshold (2s)')
    ax.fill_between(loads, 0, 2000, alpha=0.04, color=COLORS['green'])
    ax.set_xlabel("Concurrent Email Requests"); ax.set_ylabel("Mean End-to-End Latency (ms)")
    ax.set_title("Fig. 13 — Pipeline Latency vs. Concurrent Load by LLM Backend", fontweight='bold')
    ax.legend(framealpha=0.9); ax.set_xlim(0, 105)

    caption = ("Fig. 13. Mean end-to-end pipeline latency (ms) as a function of concurrent email "
               "request load for GPT-4-turbo, GPT-3.5-turbo, and Mistral-7B-Instruct backends. "
               "Mistral-7B meets the 2-second SLA up to 50 concurrent requests; "
               "GPT-3.5-turbo satisfies the SLA up to 30 requests.")
    save_fig(fig, "fig13_latency_load", caption)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 14 — Stacked Latency Breakdown (RQ6)
# ════════════════════════════════════════════════════════════════════════════
def fig_latency_breakdown():
    print("\n[Fig 14] Latency Breakdown by Node")
    backends = ['GPT-4\nturbo', 'GPT-3.5\nturbo', 'Mistral-7B\nInstruct']
    classifier = [45, 42, 38]
    retriever  = [180, 175, 168]
    responder  = [780, 240, 128]
    escalator  = [35, 33, 30]

    x = np.arange(len(backends)); w = 0.45
    fig, ax = plt.subplots(figsize=(9, 6))
    b1 = ax.bar(x, classifier, w, label='Classifier',  color=COLORS['keyword'])
    b2 = ax.bar(x, retriever,  w, bottom=classifier,   label='KB Retriever',   color=COLORS['bert'])
    b3 = ax.bar(x, responder,  w,
                bottom=np.array(classifier)+np.array(retriever),
                label='Responder (LLM)', color=COLORS['proposed'])
    b4 = ax.bar(x, escalator,  w,
                bottom=np.array(classifier)+np.array(retriever)+np.array(responder),
                label='Escalator', color=COLORS['green'])
    ax.set_xticks(x); ax.set_xticklabels(backends)
    ax.set_ylabel("Latency Contribution (ms)"); ax.set_xlabel("LLM Backend")
    ax.set_title("Fig. 14 — Pipeline Latency Breakdown by Node per LLM Backend", fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    totals = [sum(x) for x in zip(classifier, retriever, responder, escalator)]
    for i, t in enumerate(totals):
        ax.text(i, t + 15, f"{t} ms", ha='center', fontsize=11, fontweight='bold')

    caption = ("Fig. 14. Stacked bar chart decomposing total pipeline latency into per-node "
               "contributions for each LLM backend at single-request load. "
               "The Responder node dominates latency across all backends; "
               "Classifier and Escalator contribute negligibly (<10%).")
    save_fig(fig, "fig14_latency_breakdown", caption)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 15 — Ablation Bar Chart (RQ7)
# ════════════════════════════════════════════════════════════════════════════
def fig_ablation():
    print("\n[Fig 15] Ablation Study")
    configs = ['−Confidence\nCalibration', '−RAG\nRetrieval', '−MultiNode\nRouting',
               '−LangGraph\nOrchestration', 'Full Proposed\nSystem']
    intent_f1   = [0.85, 0.88, 0.82, 0.79, 0.91]
    bert_score  = [0.85, 0.78, 0.82, 0.80, 0.92]
    escalation  = [0.71, 0.84, 0.75, 0.68, 0.87]
    x = np.arange(len(configs)); w = 0.24
    fig, ax = plt.subplots(figsize=(12, 6.5))
    b1 = ax.bar(x - w, intent_f1,  w, label='Intent F1',      color=COLORS['bert'],     edgecolor='white')
    b2 = ax.bar(x,     bert_score, w, label='BERTScore F1',   color=COLORS['proposed'], edgecolor='white')
    b3 = ax.bar(x + w, escalation, w, label='Escalation F1',  color=COLORS['green'],    edgecolor='white')
    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=8)
    ax.axvline(x=3.5, color=COLORS['accent'], linestyle='--', lw=1.5, alpha=0.7)
    ax.text(3.65, 0.97, "Full System →", fontsize=9, color=COLORS['accent'])
    ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=10)
    ax.set_ylim(0.60, 1.02); ax.set_ylabel("Score"); ax.set_xlabel("Ablation Configuration")
    ax.set_title("Fig. 15 — Ablation Study: Component Contribution to System Performance",
                 fontweight='bold')
    ax.legend(framealpha=0.9)

    caption = ("Fig. 15. Ablation study results showing end-to-end system performance when each "
               "pipeline component is individually removed. Removing multi-node routing causes "
               "the largest drop in Intent F1 (−0.09), while removing RAG retrieval causes "
               "the largest drop in BERTScore F1 (−0.14).")
    save_fig(fig, "fig15_ablation", caption)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 16 — Waterfall Chart (RQ7)
# ════════════════════════════════════════════════════════════════════════════
def fig_waterfall():
    print("\n[Fig 16] Waterfall Chart — Cumulative Gains")
    stages   = ['Base LLM\n(GPT-3.5)', '+ Intent\nClassifier', '+ RAG\nRetrieval',
                '+ Multi-Node\nRouting', '+ Confidence\nEscalation', 'Full\nSystem']
    values   = [0.71, 0.05, 0.08, 0.04, 0.04, 0.0]
    running  = [0.71, 0.76, 0.84, 0.88, 0.92, 0.92]
    gains    = [0.0,  0.05, 0.08, 0.04, 0.04, 0.0]

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (s, r, g) in enumerate(zip(stages, running, gains)):
        if i == 0:
            ax.bar(i, r, color=COLORS['bert'], edgecolor='white', lw=1.2, label='Base')
            ax.text(i, r + 0.005, f"{r:.2f}", ha='center', va='bottom', fontweight='bold')
        elif i == len(stages)-1:
            ax.bar(i, r, color=COLORS['proposed'], edgecolor='white', lw=1.2, label='Full System')
            ax.text(i, r + 0.005, f"{r:.2f}", ha='center', va='bottom', fontweight='bold',
                    color=COLORS['proposed'])
        else:
            ax.bar(i, g, bottom=running[i-1], color=COLORS['green'],
                   edgecolor='white', lw=1.2, label='Component gain' if i==1 else "")
            ax.bar(i, running[i-1], color='white', edgecolor='none')
            ax.bar(i, running[i-1], color=COLORS['bert'], alpha=0.25, edgecolor='none')
            ax.text(i, r + 0.005, f"+{g:.2f}", ha='center', va='bottom',
                    color=COLORS['green'], fontweight='bold')
            ax.text(i, r + 0.023, f"({r:.2f})", ha='center', va='bottom', fontsize=9, color='gray')

    ax.set_xticks(range(len(stages))); ax.set_xticklabels(stages, fontsize=10)
    ax.set_ylim(0.60, 1.02); ax.set_ylabel("Overall System F1")
    ax.set_title("Fig. 16 — Waterfall: Cumulative Performance Gain per Architectural Component",
                 fontweight='bold')
    handles = [mpatches.Patch(color=COLORS['bert'],     label='Base performance'),
               mpatches.Patch(color=COLORS['green'],    label='Component gain'),
               mpatches.Patch(color=COLORS['proposed'], label='Full system')]
    ax.legend(handles=handles, framealpha=0.9)

    caption = ("Fig. 16. Waterfall chart illustrating cumulative overall system F1 gain as each "
               "architectural component is progressively added. RAG retrieval contributes the "
               "largest single gain (+0.08), followed by intent classification (+0.05). "
               "The full proposed system achieves F1=0.92.")
    save_fig(fig, "fig16_waterfall", caption)


# ════════════════════════════════════════════════════════════════════════════
# TABLES
# ════════════════════════════════════════════════════════════════════════════
def generate_tables():
    print("\n[Tables] Generating all 12 CSV tables...")

    # Table 1 — Baseline comparison
    df = pd.DataFrame({
        'System':               ['Keyword Baseline','Fine-tuned BERT','Vanilla GPT-4','Proposed (LangGraph-RAG)'],
        'Intent F1':            [0.61, 0.79, 0.83, 0.91],
        'BLEU-4':               [0.18, 0.27, 0.34, 0.43],
        'BERTScore F1':         [0.72, 0.81, 0.86, 0.92],
        'Escalation Precision': [0.54, 0.68, 0.71, 0.84],
        'Escalation Recall':    [0.50, 0.65, 0.68, 0.86],
    })
    save_table(df, "tab01_baseline_comparison",
               "Performance comparison of the proposed multi-node agentic system versus baseline approaches (n=1,200 test emails).")

    # Table 2 — Node specification
    df = pd.DataFrame({
        'Node':             ['Classifier','KB Retriever','Responder','Escalator'],
        'Input State Vars': ['email_body, subject','email_body, intent','email_body, intent, kb_docs','response_draft, confidence'],
        'Processing':       ['LLM zero-shot + keyword fallback','FAISS/ChromaDB semantic search','LLM generation with RAG context','Learned threshold model'],
        'Output State Vars':['intent, confidence','kb_docs, retrieval_score','response_draft','escalated (bool), routing'],
        'Transition':       ['Always → Retriever','Always → Responder','Always → Escalator','Conditional: Auto-Reply or Escalate'],
    })
    save_table(df, "tab02_node_specification",
               "Formal specification of each LangGraph node including state variables, processing function, output, and transition logic.")

    # Table 3 — RAG config ablation
    rows = []
    for vs in ['FAISS','ChromaDB']:
        for cs in [128, 256, 512]:
            for k in [1, 3, 5, 7]:
                base = 0.84 if vs == 'ChromaDB' else 0.80
                faith = round(base + 0.03*(cs==256) + 0.02*(k==5) - 0.02*(cs==512) - 0.04*(k==1) + np.random.uniform(-0.01, 0.01), 2)
                rel   = round(faith - np.random.uniform(0.01, 0.04), 2)
                cp    = round(faith - np.random.uniform(0.02, 0.06), 2)
                hall  = round(max(0.04, 0.22 - faith * 0.18 + np.random.uniform(-0.01, 0.01)), 2)
                rows.append([vs, cs, k, faith, rel, cp, f"{int(hall*100)}%"])
    df = pd.DataFrame(rows, columns=['Vector Store','Chunk Size','Top-k','Faithfulness','Relevancy','Context Precision','Hallucination Rate'])
    save_table(df, "tab03_rag_ablation",
               "RAGAS evaluation across 24 retrieval configurations (vector store × chunk size × top-k).")

    # Table 4 — Escalation strategies
    df = pd.DataFrame({
        'Strategy':               ['Static (τ=0.50)','Optimal Static (τ=0.65)','Learned (Proposed)'],
        'Precision':              [0.71, 0.78, 0.88],
        'Recall':                 [0.68, 0.75, 0.86],
        'F1':                     [0.69, 0.76, 0.87],
        'Escalation Rate (%)':    [42, 35, 22],
        'Workload Reduction (%)': [31, 44, 63],
    })
    save_table(df, "tab04_escalation_strategies",
               "Escalation routing evaluation metrics for three threshold strategies across 400 test emails.")

    # Table 5 — Robustness under noise
    df = pd.DataFrame({
        'System':         ['Keyword','Fine-tuned BERT','GPT-4','Proposed'],
        'Clean F1':       [0.61, 0.79, 0.83, 0.91],
        'Noisy (5% typo) F1':    [0.44, 0.71, 0.78, 0.87],
        'Ambiguous F1':   [0.39, 0.62, 0.71, 0.80],
        'Multilingual F1':[0.22, 0.51, 0.69, 0.77],
        'Avg Drop':       [0.19, 0.11, 0.07, 0.05],
    })
    save_table(df, "tab05_robustness",
               "Intent classification F1 under three adversarial conditions. Drop = mean F1 decrease from clean baseline.")

    # Table 6 — Misclassification patterns
    df = pd.DataFrame({
        'True Intent':       ['Billing','Billing','Refund','Technical','Account'],
        'Predicted Intent':  ['Refund','General','Billing','General','General'],
        'Keyword (count)':   [41, 28, 35, 22, 19],
        'BERT (count)':      [18, 11, 15, 9, 12],
        'GPT-4 (count)':     [12, 7, 10, 6, 8],
        'Proposed (count)':  [5, 3, 4, 4, 5],
    })
    save_table(df, "tab06_misclassification",
               "Top-5 misclassification pairs by system. Proposed system reduces confusion substantially in semantically adjacent categories.")

    # Table 7 — Human evaluation
    df = pd.DataFrame({
        'System':         ['Keyword','Fine-tuned BERT','GPT-4','Proposed'],
        'Fluency (μ±σ)':  ['2.8±0.6','3.4±0.5','4.1±0.4','4.4±0.3'],
        'Relevance (μ±σ)':['2.3±0.7','3.1±0.6','3.9±0.5','4.3±0.4'],
        'Completeness':   ['2.1±0.8','3.0±0.6','3.7±0.5','4.2±0.4'],
        'Tone (μ±σ)':     ['3.0±0.5','3.5±0.4','4.0±0.4','4.3±0.3'],
        "Cohen's κ":      ['0.71','0.74','0.76','0.79'],
    })
    save_table(df, "tab07_human_evaluation",
               "Human evaluation on 1–5 Likert scale (n=300 responses, 3 annotators). Cohen's κ for inter-rater agreement.")

    # Table 8 — Automated metrics
    df = pd.DataFrame({
        'System':        ['Keyword','Fine-tuned BERT','GPT-4','Proposed'],
        'BLEU-4':        [0.18, 0.27, 0.34, 0.43],
        'ROUGE-1':       [0.31, 0.44, 0.53, 0.62],
        'ROUGE-L':       [0.27, 0.39, 0.48, 0.58],
        'BERTScore P':   [0.70, 0.79, 0.84, 0.91],
        'BERTScore R':   [0.73, 0.82, 0.87, 0.93],
        'BERTScore F1':  [0.72, 0.81, 0.86, 0.92],
        'RAGAS Correct': [0.51, 0.64, 0.74, 0.83],
    })
    save_table(df, "tab08_automated_metrics",
               "Automated metric scores for the proposed system and all baselines on the full test set.")

    # Table 9 — LLM cost-performance
    df = pd.DataFrame({
        'LLM Backend':         ['GPT-4-turbo','GPT-3.5-turbo','Mistral-7B-Instruct'],
        'Throughput (emails/min)': [12, 35, 68],
        'Mean Latency (ms)':   [940, 380, 220],
        'P95 Latency (ms)':    [1540, 680, 410],
        'Cost per 1k emails ($)': [3.20, 0.42, 0.09],
        'BERTScore F1':        [0.92, 0.88, 0.84],
    })
    save_table(df, "tab09_llm_cost_performance",
               "Throughput, latency, cost and quality comparison across LLM backends at single-request load.")

    # Table 10 — Scalability
    df = pd.DataFrame({
        'Load Tier':           ['Low (10/min)','Medium (100/min)','High (500/min)'],
        'Mean Latency (ms)':   [380, 820, 2640],
        'P95 Latency (ms)':    [620, 1480, 4900],
        'Error Rate (%)':      [0.0, 0.8, 5.2],
        'Throughput (actual)': [10, 98, 441],
    })
    save_table(df, "tab10_scalability",
               "System performance at three load tiers (GPT-3.5-turbo backend, FastAPI deployment).")

    # Table 11 — Full ablation matrix
    df = pd.DataFrame({
        'Configuration':        ['−Confidence Calibration','−RAG Retrieval','−Multi-Node Routing',
                                 '−LangGraph Orchestration','Full Proposed System'],
        'Intent F1':            [0.85, 0.88, 0.82, 0.79, 0.91],
        'BLEU-4':               [0.40, 0.32, 0.38, 0.35, 0.43],
        'ROUGE-L':              [0.54, 0.46, 0.51, 0.48, 0.58],
        'BERTScore F1':         [0.89, 0.78, 0.85, 0.83, 0.92],
        'RAGAS Faithfulness':   [0.84, 0.61, 0.80, 0.77, 0.87],
        'Escalation F1':        [0.71, 0.84, 0.75, 0.68, 0.87],
        'Overall Composite F1': [0.84, 0.81, 0.82, 0.78, 0.92],
    })
    save_table(df, "tab11_ablation_matrix",
               "Complete ablation results. Each row removes one component; last row is full proposed system.")

    # Table 12 — Sensitivity analysis
    df = pd.DataFrame({
        'Module Replaced':     ['Random Classifier','BM25 Retrieval','Greedy Generation','Random Escalation'],
        'Replaces':            ['Intent Classifier','KB Retriever','LLM Responder','Escalator'],
        'Intent F1 (μ±σ)':    ['0.18±0.02','0.91±0.01','0.91±0.01','0.91±0.01'],
        'BERTScore F1 (μ±σ)': ['0.71±0.03','0.74±0.04','0.79±0.05','0.92±0.01'],
        'Escalation F1 (μ±σ)':['0.87±0.01','0.87±0.01','0.87±0.01','0.52±0.06'],
        'Δ vs Full System':    ['−0.73','−0.18','−0.13','−0.35'],
    })
    save_table(df, "tab12_sensitivity",
               "Sensitivity analysis: performance drop when each module is replaced with a weak alternative (5 runs each).")

    print("  ✓ All 12 tables saved.")


# ════════════════════════════════════════════════════════════════════════════
# REPORT SECTIONS
# ════════════════════════════════════════════════════════════════════════════
def generate_report():
    print("\n[Report] Writing thesis proposal text sections...")

    abstract = """ABSTRACT
========

Customer support email automation remains a critical challenge for modern enterprises, where
high volumes, linguistic diversity, and the need for contextually accurate responses exceed the
capabilities of rule-based systems. This thesis proposes, implements, and rigorously evaluates
an intelligent multi-node agentic pipeline for end-to-end email triage and automated response
generation, built on LangGraph's StateGraph abstraction and augmented with Retrieval-Augmented
Generation (RAG). The architecture comprises four functional nodes — Intent Classifier,
Knowledge Base Retriever, Response Generator, and Confidence-Aware Escalator — coordinated
through a formally specified state schema.

Extending an existing open-source prototype (github.com/suhasvenkat/Support_Mail_Agent),
this work introduces a systematic evaluation framework addressing seven research questions
spanning architecture effectiveness, RAG faithfulness, escalation robustness, intent
classification under noise, response quality metric validation, system scalability, and
ablation analysis. Evaluation is conducted on curated datasets drawn from the ASAP corpus,
Enron Email Dataset subsets, and synthetically generated support emails, using RAGAS,
BERTScore, BLEU-4, ROUGE-L, and human annotation studies.

Results demonstrate that the proposed system achieves superior intent classification F1,
response faithfulness, and escalation precision compared to keyword-based, fine-tuned BERT,
and vanilla LLM baselines. The confidence-aware escalation module reduces human agent
workload by over 60% while maintaining response quality. This work contributes a formally
evaluated, reproducible, and openly available framework for RAG-augmented LLM agent systems
in enterprise NLP applications, with findings directly relevant to the scope of
Expert Systems with Applications (Elsevier, IF ~8.5).

Keywords: Large language models, Retrieval-augmented generation, Email automation,
LangGraph, Intent classification, Confidence-aware escalation, Customer support NLP
"""

    rqs = """RESEARCH QUESTIONS (RQ1–RQ7)
=============================

RQ1 — Architecture Effectiveness:
  How does a formally specified multi-node LangGraph agentic architecture (Classifier →
  KB Retriever → Responder → Escalator) compare to monolithic LLM-based and rule-based
  baselines in end-to-end customer support email automation?

RQ2 — RAG Faithfulness & Hallucination:
  To what extent does RAG reduce hallucination risk in support response generation, and
  which retrieval configuration (FAISS vs. ChromaDB, chunk size, top-k) maximizes
  faithfulness and relevance as measured by RAGAS?

RQ3 — Confidence-Aware Escalation:
  Can a learned confidence-aware escalation mechanism trained on historical email patterns
  outperform static threshold-based routing in precision, recall, and workload reduction?

RQ4 — Intent Classification Robustness:
  How robust is LLM-based intent classification within the agentic pipeline when exposed
  to noisy, ambiguous, or multilingual support emails vs. fine-tuned BERT classifiers?

RQ5 — Response Quality Evaluation:
  Which combination of automated metrics (BLEU, ROUGE-L, BERTScore, RAGAS) best
  correlates with human-rated response quality in customer support email generation tasks?

RQ6 — Scalability and Latency Under Load:
  What are the throughput, latency, and cost trade-offs of the multi-node LangGraph
  pipeline under varying email volume loads across GPT-4, GPT-3.5-turbo, and Mistral-7B?

RQ7 — Ablation of Pipeline Components:
  Which individual components (RAG retrieval, multi-node routing, confidence calibration,
  LLM backbone) contribute most to overall system performance via ablation studies?
"""

    methodology = """METHODOLOGY
===========

PHASE 1 — Systematic Literature Review (4 weeks)
  • LLM-based email automation and intent classification
  • RAG architectures: FAISS, ChromaDB, RAGAS evaluation framework
  • Agentic LLM workflows: LangGraph, LangChain, multi-agent orchestration
  • Customer support NLP benchmarks and evaluation protocols

PHASE 2 — Architecture Design & Formalization (3 weeks)
  • Define LangGraph StateGraph with 4 nodes: Classifier, KB Retriever, Responder, Escalator
  • Specify state schema: email_body, intent, confidence, kb_docs, response_draft, escalated
  • Define transition logic and inter-node information flow
  • Extend existing prototype: github.com/suhasvenkat/Support_Mail_Agent

PHASE 3 — Dataset Construction & Knowledge Base Setup (3 weeks)
  • ASAP Dataset: adapt support-relevant subset, annotate intent labels
  • Enron Email Dataset: filter business/support threads, annotate escalation labels
  • Synthetic data: 500–1,000 emails across 6 intents via GPT-4 + human review
  • Knowledge Base: 50–100 FAQ documents, chunked at 256 tokens, 10% overlap
  • Embeddings: text-embedding-ada-002 or sentence-transformers/all-mpnet-base-v2

PHASE 4 — Pipeline Implementation & RAG Evaluation (5 weeks)
  • Implement 24 RAG configurations: {FAISS, ChromaDB} × {128,256,512 tokens} × {k=1,3,5,7}
  • Evaluate with RAGAS: Answer Faithfulness, Relevancy, Context Precision, Context Recall
  • Select optimal config based on faithfulness × relevancy composite score

PHASE 5 — Confidence-Aware Escalation Module (4 weeks)
  • Implement 3 strategies: static τ=0.50, grid-searched static, learned model
  • Train escalation model (logistic regression + MLP) on 70% annotated split
  • Evaluate on 30% holdout: precision, recall, F1, escalation rate, workload reduction
  • Compare ROC/AUC across strategies

PHASE 6 — Comprehensive Evaluation & Ablation (4 weeks)
  • Benchmark 4 systems: keyword, fine-tuned BERT, vanilla GPT-4, proposed
  • Robustness testing: noisy inputs, ambiguous intents, multilingual (DE/ES)
  • Human annotation: 3 annotators, 300 responses, 5 Likert-scale dimensions
  • Compute Pearson/Spearman correlations: automated metrics vs. human ratings
  • Load testing via Locust: 10/50/100/200/500 concurrent requests, 3 LLM backends
  • Full ablation: 5 configurations (remove each component individually)

PHASE 7 — Thesis Writing & Submission (5 weeks)
  • Draft all sections following Elsevier article template
  • Prepare publication-ready figures and tables (this script output)
  • Submit for supervisor review and incorporate feedback
"""

    contributions = """RESEARCH CONTRIBUTIONS
======================

1. FORMALLY SPECIFIED MULTI-NODE LANGGRAPH ARCHITECTURE
   First end-to-end formalization of a LangGraph StateGraph for customer support automation,
   covering intent detection → knowledge retrieval → response generation → confidence escalation
   as an integrated, evaluable pipeline.

2. REPRODUCIBLE RAG BENCHMARKING FRAMEWORK
   24-configuration RAGAS evaluation grid (FAISS/ChromaDB × chunk size × top-k) providing
   reusable benchmarking infrastructure for RAG-based NLP support systems.

3. LEARNED CONFIDENCE-AWARE ESCALATION MODEL
   A trained escalation mechanism reducing human agent workload by ~63% vs. 31% for static
   baseline, without sacrificing response quality — addressing a critical operational gap.

4. AUTOMATED METRIC VALIDATION STUDY
   Empirical Pearson correlation study (n=300) establishing BERTScore F1 as the most
   reliable automated proxy for human-rated relevance (r=0.81) in the support domain.

5. OPEN-SOURCE RESEARCH EXTENSION
   Publicly available evaluation harness extending github.com/suhasvenkat/Support_Mail_Agent
   with RAGAS integration, ablation runner, load tester, and human annotation interface.
"""

    for fname, content in [("abstract.txt", abstract), ("research_questions.txt", rqs),
                            ("methodology.txt", methodology), ("contributions.txt", contributions)]:
        path = os.path.join(REPORT_DIR, fname)
        with open(path, "w") as f:
            f.write(content)
        print(f"  ✓ Report section: {fname}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("  THESIS RESEARCH GENERATOR — Suhas Venkat")
    print("  Intelligent Email Triage with RAG-Augmented LLM Agents")
    print("=" * 72)

    print("\n── Generating Figures (16 total) ──────────────────────────────────")
    fig_architecture()
    fig_baseline_comparison()
    fig_rag_heatmap()
    fig_ragas_comparison()
    fig_faithfulness_scatter()
    fig_roc_curves()
    fig_escalation_tradeoff()
    fig_confusion_matrix()
    fig_per_class_f1()
    fig_radar_robustness()
    fig_correlation_heatmap()
    fig_bertscore_scatter()
    fig_latency_load()
    fig_latency_breakdown()
    fig_ablation()
    fig_waterfall()

    print("\n── Generating Tables (12 total) ───────────────────────────────────")
    generate_tables()

    print("\n── Generating Report Sections ─────────────────────────────────────")
    generate_report()

    print("\n" + "=" * 72)
    print("  ✅  ALL DONE! Outputs written to:")
    print(f"      {BASE_OUT}/")
    print(f"        figures/   — 16 PNG figures @ 300 DPI")
    print(f"        tables/    — 12 CSV tables with dummy data")
    print(f"        latex/     — LaTeX snippets for all figures & tables")
    print(f"        report/    — Abstract, RQs, Methodology, Contributions (.txt)")
    print("=" * 72)
    print()
    print("  NEXT STEPS:")
    print("  1. Copy figures/ and tables/ into your LaTeX/Word thesis document")
    print("  2. Use latex/ snippets directly in your .tex source files")
    print("  3. Replace dummy data in tables/ with real experimental results")
    print("  4. Run again after experiments to regenerate updated figures")
    print("=" * 72)


if __name__ == "__main__":
    # Install missing deps if needed
    try:
        from sklearn.metrics import roc_curve, auc
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "scikit-learn", "--quiet"])
        from sklearn.metrics import roc_curve, auc
    main()
