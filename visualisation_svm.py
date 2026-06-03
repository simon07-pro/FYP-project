# visualisation_svm.py
# Purpose: Visualise SVM behavioural classification results across all platforms.
# Input:   comments_gambling_{platform}.csv  (from tfidf_svm.py)
# Output:  3 rows x 3 charts:
#            Col 1 - Gambling vs non-gambling predicted proportion (bar chart)
#            Col 2 - Sentiment breakdown WITHIN gambling-like comments (bar chart)
#            Col 3 - Word cloud of gambling-like comments only
#
# This addresses Research Question 3:
# "What linguistic patterns indicate gambling-like or addictive behaviour?"
# And Research Objective 2:
# "To develop a classification framework using NLP and ML to interpret player discourse."
# Justified by Section 3.7 and Section 3.9 of the FYP report.

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from wordcloud import WordCloud
import re

# ----------------------------
# Configuration
# ----------------------------
platform_folders = {
    "hoyolab": r"C:\Users\HP\Documents\fyp chap 4\hoyolab\result",
    "reddit":   r"C:\Users\HP\Documents\fyp chap 4\reddit\result",
    "youtube":  r"C:\Users\HP\Documents\fyp chap 4\youtube\result"
}

# Words to exclude from word cloud
STOPWORDS = {
    'the', 'and', 'is', 'are', 'was', 'it', 'to', 'of', 'in', 'that',
    'have', 'for', 'not', 'with', 'you', 'this', 'but', 'from', 'they',
    'your', 'just', 'can', 'get', 'my', 'me', 'so', 'do', 'did',
    'got', 'her', 'his', 'she', 'he', 'we', 'be', 'been', 'will',
    'would', 'could', 'about', 'all', 'when', 'there', 'at', 'by',
    'an', 'if', 'up', 'now', 'out', 'no', 'yes', 'i', 'a', 'its',
    'also', 'more', 'than', 'then', 'like', 'still', 'even',
    'don', 'doesn', 'didn', 'won', 'isn', 'aren', 'wasn', 't', 's',
    'already', 'really', 'know', 'think', 'want', 'one', 'go',
    'on', 'or', 'as', 'at', 're', 'll', 've',
}

SENTIMENT_COLORS = {
    'positive': '#97C459',
    'neutral':  '#85B7EB',
    'negative': '#F09595',
}

plt.rcParams['font.family'] = 'DejaVu Sans'


# ----------------------------
# Helper function
# ----------------------------
def make_wordcloud(text_series, title, ax, colormap='Reds', max_words=60):
    """Generate word cloud from a text series."""
    text = ' '.join(text_series.dropna().astype(str).tolist())
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'\d+', '', text)

    if not text.strip():
        ax.text(0.5, 0.5, 'No gambling-like\ncomments detected',
                ha='center', va='center', fontsize=10,
                color='gray', transform=ax.transAxes)
        ax.axis('off')
        ax.set_title(title, fontsize=11, pad=8)
        return

    wc = WordCloud(
        width=700, height=350,
        background_color='white',
        stopwords=STOPWORDS,
        max_words=max_words,
        collocations=False,
        colormap=colormap
    ).generate(text)

    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=11, pad=8)


# ----------------------------
# Build figure
# ----------------------------
n_rows = len(platform_folders)

fig = plt.figure(figsize=(16, 5 * n_rows))
fig.suptitle('Gambling-Like Behaviour Classification Results\nSVM Predictions Across Platforms',
             fontsize=12, fontweight='bold', y=0.96)

gs = gridspec.GridSpec(
    n_rows, 3,
    figure=fig,
    hspace=0.5,
    wspace=0.35,
    left=0.06, right=0.97,
    top=0.85, bottom=0.04
)

# ----------------------------
# Main loop — one row per platform
# ----------------------------
for row_idx, (platform, folder) in enumerate(platform_folders.items()):
    print(f"\n=== Visualising SVM results: {platform} ===")

    # Read the gambling classification file produced by tfidf_svm.py
    gambling_file = os.path.join(folder, f"comments_gambling_{platform}.csv")
    if not os.path.exists(gambling_file):
        print(f"  File not found: {gambling_file}")
        print(f"  Please run tfidf_svm.py first.")
        continue

    df = pd.read_csv(gambling_file, encoding='utf-8', on_bad_lines='skip')

    # Verify required column exists
    if 'gambling_pred_svm' not in df.columns:
        print(f"  Missing 'gambling_pred_svm' column. Available: {df.columns.tolist()}")
        print(f"  Please re-run tfidf_svm.py.")
        continue

    # Use cleaned_text for word cloud
    text_col = 'cleaned_text' if 'cleaned_text' in df.columns else 'content'
    total = len(df)

    # Split into gambling-like and normal
    df_gambling = df[df['gambling_pred_svm'] == 1]
    df_normal   = df[df['gambling_pred_svm'] == 0]
    n_gambling  = len(df_gambling)
    n_normal    = len(df_normal)

    print(f"  Total:          {total}")
    print(f"  Gambling-like:  {n_gambling} ({n_gambling/total*100:.1f}%)")
    print(f"  Normal:         {n_normal} ({n_normal/total*100:.1f}%)")

    # ---- Chart 1: Gambling vs Normal predicted proportion ----
    ax1 = fig.add_subplot(gs[row_idx, 0])

    categories = ['Normal\nBehaviour', 'Gambling-Like\nBehaviour']
    values     = [n_normal, n_gambling]
    colors     = ['#97C459', '#F09595']

    bars = ax1.bar(categories, values, color=colors, edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars, values):
        pct = val / total * 100
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + total * 0.01,
                 f'{val}\n({pct:.1f}%)',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_title(f'{platform.upper()} — SVM Predicted Behaviour', fontsize=11, pad=8)
    ax1.set_ylabel('Number of Comments', fontsize=9)
    ax1.set_ylim(0, max(values) * 1.22)
    ax1.tick_params(axis='both', labelsize=8)
    ax1.spines[['top', 'right']].set_visible(False)

    # ---- Chart 2: Sentiment breakdown within gambling-like comments ----
    # This shows whether gambling-like comments are mostly negative, positive, or neutral
    # A research insight: gambling-like comments being predominantly negative supports
    # the hypothesis that gacha mechanics cause emotional distress
    ax2 = fig.add_subplot(gs[row_idx, 1])

    if 'sentiment_label' in df.columns and n_gambling > 0:
        sent_counts = df_gambling['sentiment_label'].value_counts()
        s_pos = sent_counts.get('positive', 0)
        s_neu = sent_counts.get('neutral',  0)
        s_neg = sent_counts.get('negative', 0)

        sent_labels = ['Positive', 'Neutral', 'Negative']
        sent_values = [s_pos, s_neu, s_neg]
        sent_colors = [SENTIMENT_COLORS['positive'],
                       SENTIMENT_COLORS['neutral'],
                       SENTIMENT_COLORS['negative']]

        bars2 = ax2.bar(sent_labels, sent_values, color=sent_colors,
                        edgecolor='white', linewidth=0.5)

        for bar, val in zip(bars2, sent_values):
            if val > 0:
                pct = val / n_gambling * 100
                ax2.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + n_gambling * 0.02,
                         f'{val}\n({pct:.0f}%)',
                         ha='center', va='bottom', fontsize=8)

        ax2.set_title(f'{platform.upper()} — Sentiment of\nGambling-Like Comments (n={n_gambling})',
                      fontsize=11, pad=8)
        ax2.set_ylabel('Count', fontsize=9)
        ax2.set_ylim(0, max(sent_values) * 1.25 if max(sent_values) > 0 else 10)
        ax2.tick_params(axis='both', labelsize=8)
        ax2.spines[['top', 'right']].set_visible(False)

    else:
        # Fallback: show overall sentiment breakdown if no sentiment column
        ax2.text(0.5, 0.5,
                 'Run vader_analysis.py first\nto see sentiment breakdown',
                 ha='center', va='center', fontsize=9, color='gray',
                 transform=ax2.transAxes)
        ax2.axis('off')
        ax2.set_title(f'{platform.upper()} — Sentiment of Gambling-Like Comments',
                      fontsize=11, pad=8)

    # ---- Chart 3: Word cloud of gambling-like comments only ----
    # Shows the most common words in comments classified as gambling-like
    # This directly answers Research Question 3:
    # "What linguistic patterns indicate gambling-like behaviour?"
    ax3 = fig.add_subplot(gs[row_idx, 2])
    make_wordcloud(
        df_gambling[text_col] if n_gambling > 0 else pd.Series(dtype=str),
        f'{platform.upper()} — Gambling-Like\nComment Word Cloud',
        ax3,
        colormap='OrRd'
    )

plt.savefig('visualisation_svm.png', dpi=150, bbox_inches='tight')
print("\nSaved: visualisation_svm.png")
plt.show()