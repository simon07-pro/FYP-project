# vader_analysis.py
# Purpose: Run VADER sentiment analysis on the cleaned comments from preprocess.py
# Input:   cleaned_comments_{platform}.csv  (produced by preprocess.py)
# Output:  comments_sentiment_{platform}.csv (saved to result folder)
#
# What this script does:
#   1. Reads the cleaned_comments file for each platform
#   2. Runs VADER on the 'cleaned_text' column (NOT the raw content column)
#   3. Saves all 4 VADER scores: pos, neu, neg, compound
#   4. Classifies each comment as positive / neutral / negative
#   5. Prints a summary so you can verify results make sense

import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ----------------------------
# Platform folders
# Update these paths to match your machine
# ----------------------------
platform_folders = {
    "hoyolab": r"C:\Users\HP\Documents\fyp chap 4\hoyolab\data(csv)",
    "reddit":   r"C:\Users\HP\Documents\fyp chap 4\reddit\data(csv)",
    "youtube":  r"C:\Users\HP\Documents\fyp chap 4\youtube\data(csv)"
}

# Initialise VADER
# VADER is chosen over TextBlob because it is specifically designed for social media text.
# It handles slang, ALL-CAPS, punctuation emphasis, and emojis natively.
# Justified by Section 2.5.2 and Section 3.6.1 of the FYP report.
analyzer = SentimentIntensityAnalyzer()


# ----------------------------
# Sentiment classification thresholds
# These are VADER's recommended thresholds from Hutto & Gilbert (2014)
# compound >= 0.05  → positive
# compound <= -0.05 → negative
# between -0.05 and 0.05 → neutral
# ----------------------------
def classify_sentiment(compound_score):
    """
    Classify a VADER compound score into positive, neutral, or negative.
    Thresholds are from the original VADER paper (Hutto & Gilbert, 2014)
    and are the standard values used in academic NLP research.
    """
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'


# ----------------------------
# Main processing loop
# ----------------------------
for platform, folder in platform_folders.items():

    if not os.path.exists(folder):
        print(f"\nWarning: {platform} folder not found, skipping.")
        print(f"  Expected path: {folder}")
        continue

    # Look for the cleaned_comments file produced by preprocess.py
    # This is the key change from the original - we now use cleaned_text, not raw content
    cleaned_file = os.path.join(folder, f"cleaned_comments_{platform}.csv")

    if not os.path.exists(cleaned_file):
        print(f"\nWarning: {platform} cleaned file not found.")
        print(f"  Expected: {cleaned_file}")
        print(f"  Please run preprocess.py first.")
        continue

    print(f"\n=== Processing platform: {platform} ===")
    print(f"  Reading: cleaned_comments_{platform}.csv")

    try:
        df = pd.read_csv(cleaned_file, encoding='utf-8', on_bad_lines='skip')
    except Exception as e:
        print(f"  Could not read file: {e}")
        continue

    # Verify the cleaned_text column exists
    # This column is created by preprocess.py and contains properly cleaned text
    if 'cleaned_text' not in df.columns:
        print(f"  Error: 'cleaned_text' column not found.")
        print(f"  Columns available: {df.columns.tolist()}")
        print(f"  Please re-run preprocess.py to regenerate the cleaned file.")
        continue

    # Remove any rows where cleaned_text is empty
    df = df.dropna(subset=['cleaned_text'])
    df = df[df['cleaned_text'].str.len() > 1]
    print(f"  Rows to analyse: {len(df)}")

    # ----------------------------
    # Run VADER on cleaned_text
    # We run VADER on cleaned_text (not the original content column) because:
    # - Gaming slang has been replaced with standard English (e.g. copium → denial coping)
    # - Emojis have been translated to text (e.g. 😭 → loudly_crying_face)
    # - Repeated characters have been normalised (e.g. noooo → noo)
    # - Non-English comments have been removed
    # Running VADER on raw text would miss all of this context.
    # ----------------------------
    def get_vader_scores(text):
        scores = analyzer.polarity_scores(str(text))
        return scores

    print(f"  Running VADER analysis...")
    vader_results = df['cleaned_text'].apply(get_vader_scores)

    # Save all 4 VADER score components
    # pos/neu/neg = proportion of text that is positive/neutral/negative (sum to 1.0)
    # compound = overall sentiment score normalised between -1 (most negative) and +1 (most positive)
    df['vader_pos']      = vader_results.apply(lambda x: x['pos'])
    df['vader_neu']      = vader_results.apply(lambda x: x['neu'])
    df['vader_neg']      = vader_results.apply(lambda x: x['neg'])
    df['vader_compound'] = vader_results.apply(lambda x: x['compound'])

    # Classify sentiment based on compound score
    df['sentiment_label'] = df['vader_compound'].apply(classify_sentiment)

    # ----------------------------
    # Print summary to verify results look reasonable
    # ----------------------------
    label_counts = df['sentiment_label'].value_counts()
    total = len(df)

    print(f"\n  --- Sentiment Summary for {platform} ---")
    for label in ['positive', 'neutral', 'negative']:
        count = label_counts.get(label, 0)
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {label:>10}: {count:>5} comments ({pct:.1f}%)")
    print(f"  {'TOTAL':>10}: {total:>5} comments")

    avg_compound = df['vader_compound'].mean()
    print(f"\n  Average compound score: {avg_compound:.4f}")
    print(f"  (range: -1.0 = most negative, 0 = neutral, +1.0 = most positive)")

    # Show a few example comments from each sentiment category so you can sanity check
    print(f"\n  Sample positive comment:")
    pos_sample = df[df['sentiment_label'] == 'positive']['cleaned_text'].head(1).values
    if len(pos_sample) > 0:
        print(f"    \"{pos_sample[0][:100]}\"")

    print(f"  Sample negative comment:")
    neg_sample = df[df['sentiment_label'] == 'negative']['cleaned_text'].head(1).values
    if len(neg_sample) > 0:
        print(f"    \"{neg_sample[0][:100]}\"")

    # ----------------------------
    # Save results to the result folder
    # ----------------------------
    parent_folder = os.path.dirname(folder)
    result_folder = os.path.join(parent_folder, "result")
    os.makedirs(result_folder, exist_ok=True)

    output_path = os.path.join(result_folder, f"comments_sentiment_{platform}.csv")
    df.to_csv(output_path, index=False)
    print(f"\n  Saved to: {output_path}")

print("\n=== VADER analysis complete for all platforms ===")
print("Columns added to output: vader_pos, vader_neu, vader_neg, vader_compound, sentiment_label")
print("Next step: run tfidf_svm.py for behavioural classification.")