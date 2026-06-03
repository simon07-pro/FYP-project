# visualisation.py
# Purpose: Visualise VADER sentiment analysis results across all platforms.
# Input:   comments_sentiment_{platform}.csv  (from vader_analysis.py)
# Output:  3 rows x 3 charts:
#            Col 1 - Sentiment distribution bar chart (positive/neutral/negative counts)
#            Col 2 - Sentiment proportion pie chart
#            Col 3 - Word cloud of all comments
#
# This addresses Research Question 1:
# "What types of sentiments do Genshin Impact players express regarding gacha systems?"
# Justified by Section 3.9 of the FYP report.

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re

# ----------------------------
# Configuration
# ----------------------------
platform_folders = {
    "hoyolab": r"C:\Users\HP\Documents\fyp chap 4\hoyolab\result",
    "reddit":   r"C:\Users\HP\Documents\fyp chap 4\reddit\result",
    "youtube":  r"C:\Users\HP\Documents\fyp chap 4\youtube\result"
}

# Words to exclude from word cloud (too common, not meaningful)
STOPWORDS = {
    'the', 'and', 'is', 'are', 'was', 'it', 'to', 'of', 'in', 'that',
    'have', 'for', 'not', 'with', 'you', 'this', 'but', 'from', 'they',
    'your', 'just', 'can', 'get', 'my', 'me', 'so', 'do', 'did',
    'got', 'her', 'his', 'she', 'he', 'we', 'be', 'been', 'will',
    'would', 'could', 'about', 'all', 'when', 'there', 'at', 'by',
    'an', 'if', 'up', 'now', 'out', 'no', 'yes', 'i', 'a', 'its',
    'also', 'more', 'than', 'then', 'like', 'still', 'even',
    'don', 'doesn', 'didn', 'won', 'isn', 'aren', 'wasn',
    'already', 'really', 'know', 'think', 'want', 'one', 'go',
    'on', 'or', 'as', 'his', 'at', 're', 'll', 've', 't', 's',
}

# Colour scheme consistent across all charts
SENTIMENT_COLORS = {
    'positive': '#97C459',   # green
    'neutral':  '#85B7EB',   # blue
    'negative': '#F09595',   # red
}

plt.rcParams['font.family'] = 'DejaVu Sans'


# ----------------------------
# Helper functions
# ----------------------------
def make_wordcloud(text_series, title, ax, max_words=80):
    """Generate a word cloud from a series of comments."""
    text = ' '.join(text_series.dropna().astype(str).tolist())
    # Remove single characters and numbers
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'\d+', '', text)

    if not text.strip():
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=11,
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
        colormap='RdYlGn'
    ).generate(text)

    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=11, pad=8)


# ----------------------------
# Build figure
# ----------------------------
n_rows = len(platform_folders)

# Taller figure so charts are not squashed — 5 inches per platform row
fig = plt.figure(figsize=(16, 5 * n_rows))
fig.suptitle('Sentiment Analysis of Genshin Impact Gacha Community\nVADER Results Across Platforms',
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
    print(f"\n=== Visualising: {platform} ===")

    # Read the VADER results file produced by vader_analysis.py
    sentiment_file = os.path.join(folder, f"comments_sentiment_{platform}.csv")
    if not os.path.exists(sentiment_file):
        print(f"  File not found: {sentiment_file}")
        print(f"  Please run vader_analysis.py first.")
        continue

    df = pd.read_csv(sentiment_file, encoding='utf-8', on_bad_lines='skip')

    # Verify required columns exist
    if 'sentiment_label' not in df.columns or 'vader_compound' not in df.columns:
        print(f"  Missing required columns. Available: {df.columns.tolist()}")
        print(f"  Please re-run vader_analysis.py.")
        continue

    # Use cleaned_text for word cloud (better quality than raw content)
    text_col = 'cleaned_text' if 'cleaned_text' in df.columns else 'content'
    total = len(df)
    print(f"  Total comments: {total}")

    # Count sentiments
    counts = df['sentiment_label'].value_counts()
    pos = counts.get('positive', 0)
    neu = counts.get('neutral',  0)
    neg = counts.get('negative', 0)
    print(f"  Positive: {pos} ({pos/total*100:.1f}%)")
    print(f"  Neutral:  {neu} ({neu/total*100:.1f}%)")
    print(f"  Negative: {neg} ({neg/total*100:.1f}%)")

    # ---- Chart 1: Sentiment count bar chart ----
    ax1 = fig.add_subplot(gs[row_idx, 0])
    labels    = ['Positive', 'Neutral', 'Negative']
    values    = [pos, neu, neg]
    bar_colors = [SENTIMENT_COLORS['positive'],
                  SENTIMENT_COLORS['neutral'],
                  SENTIMENT_COLORS['negative']]

    bars = ax1.bar(labels, values, color=bar_colors, edgecolor='white', linewidth=0.5)

    # Add count labels on top of each bar
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                 str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_title(f'{platform.upper()} — Sentiment Distribution', fontsize=11, pad=8)
    ax1.set_ylabel('Number of Comments', fontsize=9)
    ax1.set_ylim(0, max(values) * 1.18)
    ax1.tick_params(axis='both', labelsize=8)
    ax1.spines[['top', 'right']].set_visible(False)

    # ---- Chart 2: Sentiment proportion pie chart ----
    ax2 = fig.add_subplot(gs[row_idx, 1])
    pie_labels = [f'Positive\n{pos/total*100:.1f}%',
                  f'Neutral\n{neu/total*100:.1f}%',
                  f'Negative\n{neg/total*100:.1f}%']
    pie_colors = [SENTIMENT_COLORS['positive'],
                  SENTIMENT_COLORS['neutral'],
                  SENTIMENT_COLORS['negative']]
    wedges, texts = ax2.pie(
        [pos, neu, neg],
        labels=pie_labels,
        colors=pie_colors,
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    for text in texts:
        text.set_fontsize(8)
    ax2.set_title(f'{platform.upper()} — Sentiment Proportions\n(n={total})', fontsize=11, pad=8)

    # ---- Chart 3: Word cloud of all comments ----
    ax3 = fig.add_subplot(gs[row_idx, 2])
    make_wordcloud(df[text_col], f'{platform.upper()} — Word Cloud', ax3)

plt.savefig('visualisation_vader.png', dpi=150, bbox_inches='tight')
print("\nSaved: visualisation_vader.png")
plt.show()