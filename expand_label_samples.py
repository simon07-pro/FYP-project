# expand_label_samples.py
# Purpose: From cleaned_comments across all platforms, filter out comments
#          that may contain gambling behaviour, for quick manual labelling.
#
# Keyword selection rationale: FYP Report Table 3.1 (Psychological Gambling Indicators -> Linguistic Features)
#   - Chasing Losses
#   - Near-Miss Effect
#   - Sunk Cost Fallacy
#   - Emotional Volatility
#   - High Spending Slang
#
# Note: Keywords are only used to speed up sampling. Whether a comment is
#       ultimately labelled as gambling-related still requires full manual review.

import os
import pandas as pd
import re

# ============================================================
# 1. Gambling behaviour keywords by category (based on Table 3.1 + Genshin community slang)
# ============================================================

# 1.1 Chasing Losses
# Definition: Continuing to spend money or resources to recover previous losses
CHASING_KEYWORDS = [
    "just one more", "one more pull", "one more ten",
    "swipe", "credit card", "top up", "top-up",
    "wallet", "rent money", "savings", "broke",
    "can't stop", "couldn't stop", "i need to stop"
]

# 1.2 Near-Miss Effect
# Definition: Feeling frustrated that the outcome was "almost" a success, and immediately wanting to try again
NEAR_MISS_KEYWORDS = [
    "so close", "soft pity", "hard pity", "ruined pity",
    "wrong 5 star", "lost 50/50", "lost the 50/50",
    "qiqi", "qiqi'd", "qiqied",
    "shaking", "so unlucky", "89 pity", "near guarantee"
]

# 1.3 Sunk Cost Fallacy
# Definition: Unable to stop because too much has already been invested
SUNK_COST_KEYWORDS = [
    "already spent", "already in", "already pulled",
    "wasted", "can't stop now", "pulls deep",
    "invested", "guarantee", "guaranteed",
    "might as well", "have to go to pity", "committed"
]

# 1.4 Emotional Volatility
# Definition: Extreme emotional expressions (positive or negative) about pulling/gacha
EMOTIONAL_KEYWORDS = [
    "hate this game", "i hate hoyo", "i love hoyo",
    "addicted", "crying", "cry", "depressed", "rage",
    "quit", "finally got", "finally pulled",
    "shaking", "heartbroken", "devastated", "ecstatic"
]

# 1.5 High Spending (community slang reflecting excessive spending)
SPENDING_KEYWORDS = [
    "whale", "whaling", "dolphin", "p2w",
    "c6", "r5", "maxed out", "maxed", "c6r5",
    "spent", "spent heavily", "spent money",
    "swipey", "top-up", "genesis crystals"
]

# Merge all keywords (for fast matching)
ALL_KEYWORDS = (CHASING_KEYWORDS + NEAR_MISS_KEYWORDS +
                SUNK_COST_KEYWORDS + EMOTIONAL_KEYWORDS +
                SPENDING_KEYWORDS)

# Build a lowercase set for easier matching
KEYWORD_SET = set(kw.lower() for kw in ALL_KEYWORDS)

# ============================================================
# 2. Helper functions: identify which psychological categories are matched
# ============================================================
def get_matched_categories(text):
    """
    Returns the psychological indicator categories matched by the text (comma-separated).
    Displayed in Excel to help with quick labelling decisions.
    """
    if not isinstance(text, str):
        return ""
    text_lower = text.lower()
    categories = []
    if any(kw in text_lower for kw in CHASING_KEYWORDS):
        categories.append("ChasingLosses")
    if any(kw in text_lower for kw in NEAR_MISS_KEYWORDS):
        categories.append("NearMiss")
    if any(kw in text_lower for kw in SUNK_COST_KEYWORDS):
        categories.append("SunkCost")
    if any(kw in text_lower for kw in EMOTIONAL_KEYWORDS):
        categories.append("Emotional")
    if any(kw in text_lower for kw in SPENDING_KEYWORDS):
        categories.append("HighSpending")
    return ",".join(categories) if categories else "None"

def is_gambling_related(text):
    """Quick check whether the text contains any gambling-related keywords (for filtering)."""
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in KEYWORD_SET)

# ============================================================
# 3. Configure paths
# ============================================================
platform_folders = {
    "hoyolab": r"C:\Users\HP\Documents\fyp chap 4\hoyolab\data(csv)",
    "reddit":   r"C:\Users\HP\Documents\fyp chap 4\reddit\data(csv)",
    "youtube":  r"C:\Users\HP\Documents\fyp chap 4\youtube\data(csv)"
}

OUTPUT_FILE = r"C:\Users\HP\Documents\fyp chap 4\potential_gambling_to_label.xlsx"
SAMPLE_SIZE = 500  # Cap at 500 samples to keep the labelling task manageable

# ============================================================
# 4. Main pipeline: read cleaned_comments, filter potential gambling comments
# ============================================================
all_candidates = []

for platform, folder in platform_folders.items():
    cleaned_file = os.path.join(folder, f"cleaned_comments_{platform}.csv")
    
    if not os.path.exists(cleaned_file):
        print(f"Warning: cleaned file for {platform} not found, skipping.")
        continue
    
    print(f"Reading {platform} ...")
    df = pd.read_csv(cleaned_file, encoding='utf-8', on_bad_lines='skip')
    
    if 'cleaned_text' not in df.columns:
        print(f"  Skipping: 'cleaned_text' column not found")
        continue
    
    # Filter comments containing keywords
    mask = df['cleaned_text'].apply(is_gambling_related)
    candidates = df[mask].copy()
    
    if len(candidates) == 0:
        print(f"  {platform}: No potential gambling comments found")
        continue
    
    # Add platform column and matched category column
    candidates['platform'] = platform
    candidates['matched_categories'] = candidates['cleaned_text'].apply(get_matched_categories)
    
    # Keep only the necessary columns
    candidates = candidates[['cleaned_text', 'platform', 'matched_categories']]
    
    # Deduplicate based on text content
    candidates = candidates.drop_duplicates(subset=['cleaned_text'])
    
    print(f"  {platform}: Found {len(candidates)} potential gambling comments")
    all_candidates.append(candidates)

if not all_candidates:
    print("No potential gambling comments found. Please verify that cleaned_comments files exist and contain data.")
    exit()

# Merge candidates from all platforms
df_all = pd.concat(all_candidates, ignore_index=True)
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

# Apply sample size cap
if len(df_all) > SAMPLE_SIZE:
    df_all = df_all.head(SAMPLE_SIZE)

# Add blank label column (to be filled in manually)
df_all['gambling_label'] = ''  # 1 = gambling behaviour, 0 = normal

# Reorder columns for easier reading
df_all = df_all[['cleaned_text', 'gambling_label', 'matched_categories', 'platform']]

# ============================================================
# 5. Save to Excel with an annotation guide sheet
# ============================================================
with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
    # Main data sheet
    df_all.to_excel(writer, index=False, sheet_name='Label Here')
    
    # Set column widths
    ws = writer.sheets['Label Here']
    ws.column_dimensions['A'].width = 80   # cleaned_text
    ws.column_dimensions['B'].width = 16   # gambling_label
    ws.column_dimensions['C'].width = 30   # matched_categories
    ws.column_dimensions['D'].width = 12   # platform
    
    # Labelling guide sheet (based on Table 3.1)
    guide_data = {
        'Label': [1, 1, 1, 1, 0, 0, 0],
        'Psychological Indicator': [
            'Chasing Losses',
            'Near-Miss Effect',
            'Sunk Cost Fallacy',
            'Emotional Volatility',
            'Normal Comment',
            'Normal Comment',
            'Normal Comment'
        ],
        'Example Comment': [
            'I lost the 50/50 so I had to swipe my card. There goes my rent money.',
            'I was at 75 pity and got Qiqi. I am shaking right now.',
            'I am already 70 pulls deep, might as well go to hard pity.',
            'I literally hate this game but I finally got her, see you tomorrow.',
            'Hu Tao design is really beautiful, I love her animations.',
            'When is the next banner coming out?',
            'Does anyone know a good team comp for Hu Tao?'
        ],
        'Keywords to Look For': [
            'swipe, credit card, top up, broke, rent money, just one more',
            'so close, soft pity, Qiqi, lost 50/50, shaking',
            'already spent, wasted, guarantee, might as well, invested',
            'hate, love, addicted, crying, quit, finally',
            'No distress/financial loss keywords',
            'No distress/financial loss keywords',
            'No distress/financial loss keywords'
        ]
    }
    guide_df = pd.DataFrame(guide_data)
    guide_df.to_excel(writer, index=False, sheet_name='Labelling Guide (Table 3.1)')
    
    ws2 = writer.sheets['Labelling Guide (Table 3.1)']
    ws2.column_dimensions['A'].width = 10
    ws2.column_dimensions['B'].width = 35
    ws2.column_dimensions['C'].width = 60
    ws2.column_dimensions['D'].width = 45

print(f"\nDone! Saved to: {OUTPUT_FILE}")
print(f"Total {len(df_all)} comments to label (all filtered by gambling-related keywords)")
print("\nNext steps:")
print("1. Open the Excel file above")
print("2. Read the content in the 'cleaned_text' column")
print("3. Use the four psychological indicators in 'Labelling Guide (Table 3.1)' to decide if the comment reflects gambling behaviour")
print("4. Enter in the 'gambling_label' column: 1 = gambling behaviour, 0 = normal")
print("5. Save the file")
print("6. Run the subsequent label-merging script to combine the new labels with existing ones")