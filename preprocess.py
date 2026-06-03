# preprocess.py
# Purpose: Clean and normalise raw comments from all platforms.
# This script does NOT filter by keywords - it keeps ALL English comments.
# The goal is to produce clean text that VADER and the SVM can work on accurately.
#
# What this script does:
#   1. Removes noise: URLs, HTML, HoYoLAB custom emojis, spam
#   2. Removes non-English comments (VADER is English-only - Section 3.3 of report)
#   3. Translates real unicode emojis to text (Section 3.4.2 of report)
#   4. Normalises gaming slang to standard English (Table 3.1 of report)
#   5. Normalises repeated characters ("noooo" → "noo")
#   6. Preserves apostrophes and contractions for correct VADER scoring

import os
import re
import pandas as pd
import emoji

# ----------------------------
# Gaming slang dictionary
# Based on Table 3.1 in the FYP report - maps domain-specific slang to
# standard English equivalents so VADER can score them correctly.
# ----------------------------
SLANG_DICT = {
    # Characters often received as bad outcomes (losing the 50/50)
    "qiqi'd":       "lost bad outcome",
    "qiqied":       "lost bad outcome",
    "qiqi":         "bad outcome loss",
    "jean'd":       "lost bad outcome",

    # Gambling-related slang
    "copium":       "denial rationalization coping",
    "hopium":       "false hope denial",
    "shafted":      "unlucky lost cheated",
    "rip":          "lost failed unfortunate",
    "whale":        "high spender spending money",
    "dolphin":      "medium spender spending money",
    "f2p":          "free to play no spending",
    "c6":           "maxed out spent heavily",
    "r5":           "maxed weapon spent heavily",

    # Pity system slang
    "soft pity":    "near guarantee approaching pity",
    "hard pity":    "guaranteed pity limit",
    "pity reset":   "lost pity reset wasted",
    "lost 50/50":   "lost coin flip bad luck wasted",
    "won 50/50":    "won coin flip lucky success",
    "50/50":        "coin flip chance luck",

    # Emotional expressions common in gacha communities
    "bricked":      "broken failed wasted",
    "scammed":      "cheated unfair angry",
    "L pull":       "bad pull loss failure",
    "W pull":       "good pull win success",
}

# ----------------------------
# Platform folders - update these paths to match your machine
# ----------------------------
platform_folders = {
    "hoyolab": r"C:\Users\HP\Documents\fyp chap 4\hoyolab\data(csv)",
    "reddit":   r"C:\Users\HP\Documents\fyp chap 4\reddit\data(csv)",
    "youtube":  r"C:\Users\HP\Documents\fyp chap 4\youtube\data(csv)"
}


# ============================================================
# CLEANING FUNCTIONS
# Each function does one specific job so it is easy to explain
# ============================================================

def remove_noise(text):
    """
    Remove URLs, HTML tags, and non-meaningful special characters.
    These add no sentiment value and confuse VADER.

    Special handling for HoYoLAB custom emojis:
    HoYoLAB uses its own emoji format: [emoji:https://...url...]
    These are NOT unicode emojis - they are image URLs inside brackets.
    We remove them entirely since we cannot translate them to sentiment text.

    Apostrophes are preserved (don't → don't, not don t) because:
    - VADER handles contractions correctly
    - Removing apostrophes changes word meaning (e.g. "can't" → "cant")
    """
    text = re.sub(r'\[emoji:[^\]]*\]', '', text)               # remove HoYoLAB custom emojis
    text = text.replace('\u2019', "'").replace('\u2018', "'")  # curly apostrophes → straight
    text = text.replace('\u2026', ' ')                         # ellipsis … → space
    text = re.sub(r'http\S+|www\S+', '', text)                 # remove URLs
    text = re.sub(r'<.*?>', '', text)                          # remove HTML tags
    # Keep: letters, digits, spaces, apostrophes, unicode emojis
    # Remove: all other special characters
    text = re.sub(r"[^\w\s'\U00010000-\U0010ffff]", ' ', text, flags=re.UNICODE)
    return text


def translate_emojis(text):
    """
    Convert emojis to their text description.
    Example: 😭 → "crying face"
    This is important because VADER can score emotion words
    but cannot directly read emoji unicode symbols.
    Justified by Section 3.4.2 of the FYP report.
    """
    return emoji.demojize(text, delimiters=(" ", " "))


def normalize_slang(text):
    """
    Replace gaming-specific slang with standard English equivalents.
    This allows VADER to correctly score emotional content that would
    otherwise be missed (e.g., "copium" has no score in standard dictionaries).
    Justified by Table 3.1 and Section 3.4.2 of the FYP report.
    """
    for slang, replacement in SLANG_DICT.items():
        # Use word boundary matching so "qiqi" doesn't match inside other words
        text = re.sub(r'\b' + re.escape(slang) + r'\b', replacement, text, flags=re.IGNORECASE)
    return text


def normalize_repeated_chars(text):
    """
    Reduce exaggerated characters to maximum 2 repetitions.
    Example: "noooooo" → "noo", "PLEASEEE" → "PLEASEE"
    Social media users repeat characters for emphasis.
    Keeping 2 preserves the emphasis signal for VADER.
    """
    return re.sub(r'(.)\1{2,}', r'\1\1', text)


def normalize_case(text):
    """
    Convert text to lowercase for consistency.
    Exception: We do NOT lowercase before VADER runs, because VADER
    uses ALL-CAPS as an intensity signal (e.g., "HATE" scores more
    negatively than "hate"). This function is only used for
    duplicate detection, not for the final cleaned text.
    """
    return text.lower()


def clean_comment(text):
    """
    Master cleaning function - applies all steps in the correct order.
    Order matters: translate emojis before removing symbols,
    normalize slang before case changes.
    """
    text = str(text)                      # ensure it is a string
    text = remove_noise(text)             # step 1: remove URLs, HTML
    text = translate_emojis(text)         # step 2: convert emojis to text
    text = normalize_slang(text)          # step 3: replace gaming slang
    text = normalize_repeated_chars(text) # step 4: reduce "noooo" to "noo"
    text = text.strip()                   # step 5: remove leading/trailing spaces
    return text


# ============================================================
# MAIN PROCESSING LOOP
# ============================================================

for platform, platform_folder in platform_folders.items():

    if not os.path.exists(platform_folder):
        print(f"\nWarning: {platform} folder not found, skipping.")
        print(f"  Expected path: {platform_folder}")
        continue

    dfs = []
    print(f"\n=== Processing platform: {platform} ===")

    for filename in os.listdir(platform_folder):

        # Skip any files that were previously generated by this script
        if not filename.endswith(".csv"):
            continue
        if filename.startswith("cleaned_comments"):
            continue

        file_path = os.path.join(platform_folder, filename)
        print(f"  Reading: {filename}")

        # Reddit CSVs exported from PRAW have 4 header rows of metadata
        try:
            if platform == "reddit":
                df = pd.read_csv(file_path, skiprows=4, on_bad_lines='skip')
            else:
                df = pd.read_csv(file_path, on_bad_lines='skip')
        except Exception as e:
            print(f"  Could not read {filename}: {e}")
            continue

        # Strip whitespace from column names (common issue with CSV exports)
        df.columns = df.columns.map(str).str.strip()
        print(f"  Columns found: {df.columns.tolist()}")

        # Detect the comment text column - support multiple naming conventions
        columns_lower = [col.lower() for col in df.columns]
        text_column = None

        for candidate in ['content', 'comment', 'text', 'body']:
            if candidate in columns_lower:
                text_column = df.columns[columns_lower.index(candidate)]
                break

        if text_column is None:
            print(f"  Warning: No text column found in {filename}, skipping.")
            continue

        # --- Basic cleaning ---
        df = df.dropna(subset=[text_column])                   # remove empty rows
        df = df.drop_duplicates(subset=[text_column])          # remove exact duplicates
        df = df[df[text_column].str.len() > 5]                 # remove very short comments (e.g. "ok", "lol")

        # --- Remove non-English comments ---
        # Your report (Section 3.3) specifies English-language data only.
        # VADER is an English tool - non-English text produces unreliable scores.
        # We keep a row only if the majority of its characters are ASCII (Latin alphabet).
        # This allows game terms like "Hu Tao" in otherwise English sentences to pass.
        def is_english(text):
            text = str(text)
            if len(text) == 0:
                return False

            # Check 1: At least 80% ASCII characters
            ascii_chars = sum(1 for c in text if ord(c) < 128)
            if (ascii_chars / len(text)) < 0.8:
                return False

            # Check 2: Detect garbled UTF-8 encoding artifacts
            # These appear when non-English UTF-8 text is read with wrong encoding.
            # Spanish accents become: Ã³ Ã­ Ã¡ Ã± Ã© Ã  Ãº Ã‰ Ã
            # Russian Cyrillic becomes: Ð Ñ patterns mixed with Latin
            # A real English comment will never contain these sequences.
            garbled_patterns = [
                r'Ã[³íáñéàúÃ‰Ã]',   # broken Spanish accented characters
                r'Ð[^\s]{1,3}Ñ',     # broken Cyrillic pattern
                r'Ã\xa0|Ã\x83',      # other common UTF-8 artifacts
                r'Ã³|Ã­|Ã¡|Ã±|Ã©|Ãº|Ã ',  # explicit broken Spanish chars
            ]
            for pattern in garbled_patterns:
                if re.search(pattern, text):
                    return False

            # Check 3: Detect non-English comments that use plain ASCII letters
            # (German, French, short Spanish/Italian without accents etc.)
            # For ALL comments under 15 words, verify enough common English words exist.
            # Longer comments are skipped to avoid false-positives on mixed content.
            common_english = {
                'the','and','is','are','was','it','to','of','in','that',
                'have','for','not','with','you','this','but','from','they',
                'your','just','can','want','get','my','me','so','what',
                'do','did','go','got','her','his','she','he','we','be',
                'been','will','would','could','should','about','how','all',
                'when','where','who','why','there','their','at','by','an',
                'if','up','already','really','know','think','like','still',
                'even','also','more','than','then','now','out','no','yes',
                'i','a','its','one','please','come','give','need','pull',
                "don't","i'm","it's","can't","i've","won't","didn't",
            }
            words = text.lower().split()

            # Apply ratio check to short comments (under 15 words) always,
            # and to longer comments only if they contain accented characters
            accented_latin = sum(1 for c in text if '\u00c0' <= c <= '\u017e')
            should_check = (len(words) <= 15) or (accented_latin > 0)

            if should_check:
                english_word_count = sum(1 for w in words if w in common_english)
                english_ratio = english_word_count / max(len(words), 1)
                if english_ratio < 0.25:
                    return False

            return True

        before_lang = len(df)
        df = df[df[text_column].apply(is_english)]
        print(f"  Removed {before_lang - len(df)} non-English rows")

        # --- Remove spam/promotional comments ---
        # HoYoLAB is heavily targeted by referral code bots that post in many languages.
        # These bots always share the same structure: a referral link + invitation code.
        # We detect them by their structural patterns rather than language.
        #
        # Patterns detected:
        #   English : "exclusive link", "invitation code", "lucky draw rewards"
        #   Spanish : "enlace exclusivo", "Código de invitación", "Protogemas"
        #   Turkish : "özel bağlantı", "Davet kodum", "çekiliş"
        #   Vietnamese: "liên kết độc quyền", "Nguyên Thạch"
        #   Universal: hoyo.link referral URLs, invitation code alphanumeric patterns
        spam_patterns = (
            r'exclusive link|click my link|invitation code|lucky draw rewards'
            r'|enlace exclusivo|[Cc]ódigo de invitaci|Protogemas.*sorteo'
            r'|özel ba.lant|Davet kodum|çekili'
            r'|liên k.t .c quy|Nguy.n Th'
            r'|hoyo\.link'                          # all bots share this referral domain
            r'|m_code=[A-Z0-9]{8,}'                 # referral code URL parameter
        )
        before_spam = len(df)
        df = df[~df[text_column].str.contains(spam_patterns, case=False, regex=True, na=False)]
        print(f"  Removed {before_spam - len(df)} spam rows")

        # --- Apply full text cleaning pipeline ---
        df['cleaned_text'] = df[text_column].apply(clean_comment)

        # Remove rows where cleaning produced an empty result
        df = df[df['cleaned_text'].str.len() > 5]

        # --- Second language check on cleaned_text ---
        # Some non-English text appears garbled in the raw CSV (e.g. Ð²Ð¾Ñ‚ Ð¸Ð¼ÐµÐ½Ð½Ð¾)
        # but after pandas reads and cleans it, the real Cyrillic characters appear
        # (e.g. вот именно я собираюсь...). The first language check runs on the
        # raw content column and catches the garbled version, but misses the decoded
        # version that ends up in cleaned_text. This second pass catches those cases.
        before_second = len(df)
        df = df[df['cleaned_text'].apply(is_english)]
        print(f"  Removed {before_second - len(df)} rows caught by second language check")

        # Add a column to track which platform this comment came from
        df['platform'] = platform

        print(f"  Rows after cleaning: {len(df)}")
        dfs.append(df)

    if not dfs:
        print(f"  No valid CSV files found for {platform}.")
        continue

    # Combine all files from this platform into one dataframe
    df_platform = pd.concat(dfs, ignore_index=True)

    # Final deduplication across all files (in case same comment appeared in multiple files)
    df_platform = df_platform.drop_duplicates(subset=['cleaned_text'])
    print(f"  Total rows after merging all files: {len(df_platform)}")

    # Save the cleaned output - this is what vader_analysis.py and tfidf_svm.py will use
    output_path = os.path.join(platform_folder, f"cleaned_comments_{platform}.csv")
    df_platform.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

print("\n=== Preprocessing complete for all platforms ===")
print("Next step: run vader_analysis.py on the cleaned_comments files.")