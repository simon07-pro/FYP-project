# Sentiment Analysis of Gambling-Like Behaviour in Genshin Impact Gacha Community

> Final Year Project (FYP) — Faculty of Computer Science and Information Technology, UNIMAS  
> Author: Hii Lu Ping | Bachelor of Computer Science with Honours (Network Computing) | 2026

---

## Project Overview

This project investigates **gambling-like behaviour** in the Genshin Impact gacha community by analysing real player comments from three online platforms: **Reddit**, **HoYoLAB**, and **YouTube**.

Instead of using self-reported surveys (which suffer from recall bias and social desirability bias), this study takes a **computational, data-driven approach** using Natural Language Processing (NLP) and supervised machine learning.

### Research Questions

1. What types of sentiments do Genshin Impact players express regarding gacha systems?
2. How can NLP and machine learning classify and interpret these sentiments effectively?
3. What linguistic patterns indicate gambling-like or addictive behaviour among players?

### Key Findings

- **44.5%** of all comments were classified as positive; **28.0%** as negative overall
- Reddit showed the highest negative proportion at **29.4%**, consistent with its venting culture
- The SVM identified **13.2%** of all 10,732 comments as showing gambling-like behaviour signals
- YouTube had the highest gambling-like rate (**16.7%**), followed by Reddit (**11.7%**) and HoYoLAB (**6.0%**)
- **39–41%** of gambling-like comments carried *positive* sentiment — demonstrating that sentiment analysis alone is insufficient to detect compulsive behaviour
- Top predictors of gambling-like behaviour: `50/50`, `lost`, `pity`, `depressed`, `spent`, `tighnari`

---

## Repository Structure

```
├── import_requests.py       # Step 0: Scrape comments from HoYoLAB via HTTP API
├── preprocess.py            # Step 1: Clean and normalise raw comments
├── vader_analysis.py        # Step 2: VADER sentiment analysis
├── tfidf_svm.py             # Step 3: TF-IDF + SVM / Logistic Regression classification
├── visualisation.py         # Step 4a: Visualise VADER sentiment results
├── visualisation_svm.py     # Step 4b: Visualise SVM gambling-like behaviour results
└── README.md
```

---

## Pipeline Overview

```
[Data Collection]         [Preprocessing]         [Sentiment Analysis]
import_requests.py   →   preprocess.py        →   vader_analysis.py
(HoYoLAB scraper)        (clean + normalise)       (VADER scores)
                                                          ↓
                                               [Behavioural Classification]
                                                   tfidf_svm.py
                                               (TF-IDF + SVM + LR)
                                                          ↓
                                                  [Visualisation]
                                         visualisation.py  |  visualisation_svm.py
```

---

## Requirements

### Python Version
Python 3.8 or higher is recommended.

### Install Dependencies

```bash
pip install pandas scikit-learn vaderSentiment emoji matplotlib seaborn wordcloud openpyxl requests
```

| Package | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `scikit-learn` | TF-IDF vectorisation, SVM, Logistic Regression |
| `vaderSentiment` | Sentiment analysis |
| `emoji` | Emoji-to-text translation |
| `matplotlib` / `seaborn` | Visualisation |
| `wordcloud` | Word cloud generation |
| `openpyxl` | Reading/writing Excel files for labelling |
| `requests` | HTTP requests for HoYoLAB scraper |

---

## Setup: Folder Structure

Before running any script, create the following folder structure on your machine and update the paths in each script accordingly.

```
fyp chap 4/
├── hoyolab/
│   ├── data(csv)/          ← Place raw HoYoLAB CSV files here
│   └── result/             ← Auto-created by scripts
├── reddit/
│   ├── data(csv)/          ← Place raw Reddit CSV files here
│   └── result/
└── youtube/
    ├── data(csv)/          ← Place raw YouTube CSV files here
    └── result/
```

### Update Paths in Each Script

Every script contains a `platform_folders` dictionary near the top. Update the paths to match your machine:

```python
platform_folders = {
    "hoyolab": r"C:\Your\Path\hoyolab\data(csv)",
    "reddit":   r"C:\Your\Path\reddit\data(csv)",
    "youtube":  r"C:\Your\Path\youtube\data(csv)"
}
```

---

## Step-by-Step User Guide

### Step 0 — Data Collection (HoYoLAB only)

> Reddit and YouTube comments were collected using browser extensions (Reddit Comment Scraper and YouTube Comment Exporter). Only HoYoLAB requires the Python script.

**Script:** `import_requests.py`

1. Open `import_requests.py` and edit the configuration at the top:
   ```python
   post_ids = ["43967898", "44296219", ...]   # List of HoYoLAB post IDs to scrape
   output_folder = r"C:\Your\Path\hoyolab\data(csv)"
   ```
2. Run the script:
   ```bash
   python import_requests.py
   ```
3. One CSV file per post will be saved to your output folder, named `hoyolab_post_{post_id}.csv`.

**Output columns:** `reply_id`, `parent_id`, `user_id`, `nickname`, `content`, `like_num`, `created_at`

---

### Step 1 — Preprocessing

**Script:** `preprocess.py`

This script cleans all raw CSV files from all three platforms and produces a single merged `cleaned_comments_{platform}.csv` for each platform.

**What it does:**
- Removes URLs, HTML tags, and HoYoLAB custom emoji image links
- Translates Unicode emojis to text descriptions (e.g., 😭 → `loudly_crying_face`)
- Normalises gaming slang to standard English (e.g., `copium` → `denial rationalization coping`)
- Removes non-English comments (VADER is English-only)
- Removes spam/referral bot comments
- Reduces exaggerated characters (e.g., `noooooo` → `noo`)

**Run:**
```bash
python preprocess.py
```

**Output:** `cleaned_comments_{platform}.csv` saved in each platform's `data(csv)` folder.

**Expected column:** `cleaned_text` — this is the column used by all subsequent scripts.

---

### Step 2 — Sentiment Analysis (VADER)

**Script:** `vader_analysis.py`

Runs VADER sentiment analysis on the `cleaned_text` column from Step 1.

**VADER thresholds** (Hutto & Gilbert, 2014):
- `compound >= 0.05` → Positive
- `compound <= -0.05` → Negative
- Between → Neutral

**Run:**
```bash
python vader_analysis.py
```

**Output:** `comments_sentiment_{platform}.csv` saved in each platform's `result/` folder.

**New columns added:**

| Column | Description |
|---|---|
| `vader_pos` | Proportion of positive text (0.0–1.0) |
| `vader_neu` | Proportion of neutral text (0.0–1.0) |
| `vader_neg` | Proportion of negative text (0.0–1.0) |
| `vader_compound` | Overall sentiment score (−1.0 to +1.0) |
| `sentiment_label` | `positive` / `neutral` / `negative` |

---

### Step 3 — Behavioural Classification (SVM + Logistic Regression)

**Script:** `tfidf_svm.py`

Trains SVM and Logistic Regression classifiers to detect gambling-like behaviour. Requires a **manually labelled Excel file**.

#### 3a — Prepare the Labelled File

Before running, you need a labelled dataset. The labelled file should be an Excel workbook with a sheet named `Label Here` containing at minimum:

| Column | Description |
|---|---|
| `cleaned_text` | The comment text |
| `gambling_label` | `1` = gambling-like behaviour, `0` = normal |

**Labelling guide** (based on Table 3.1 of the report):

| Label = 1 (Gambling-Like) | Example |
|---|---|
| Chasing losses | *"I lost the 50/50 so I swiped my card again"* |
| Near-Miss effect | *"I was at 75 pity and got Tighnari, I'm shaking"* |
| Sunk cost fallacy | *"I'm already 60 pulls in, might as well go to hard pity"* |
| Emotional volatility | *"I literally hate this game but I finally got her"* |

#### 3b — Update the Labelled File Path

In `tfidf_svm.py`, update:
```python
LABELLED_FILE = r"C:\Your\Path\your_labelled_file.xlsx"
```

#### 3c — Run the Script

```bash
python tfidf_svm.py
```

**Output:** `comments_gambling_{platform}.csv` saved in each platform's `result/` folder.

**New columns added:**

| Column | Description |
|---|---|
| `gambling_pred_svm` | SVM prediction: `1` = gambling-like, `0` = normal |
| `gambling_pred_lr` | Logistic Regression prediction |
| `gambling_risk_svm` | SVM probability score (0.0–1.0); higher = more gambling-like |
| `gambling_risk_lr` | Logistic Regression probability score |

**Console output includes:**
- Accuracy, Precision, Recall, F1-score for both models
- 5-fold cross-validation F1
- Top 15 features most associated with gambling-like behaviour

---

### Step 4a — Visualise VADER Results

**Script:** `visualisation.py`

Generates a 3×3 chart grid (one row per platform):
- **Column 1:** Sentiment distribution bar chart (positive / neutral / negative counts)
- **Column 2:** Sentiment proportion pie chart
- **Column 3:** Word cloud of all comments

**Run:**
```bash
python visualisation.py
```

**Output:** `visualisation_vader.png` saved in the current working directory.

---

### Step 4b — Visualise SVM Results

**Script:** `visualisation_svm.py`

Generates a 3×3 chart grid (one row per platform):
- **Column 1:** Gambling vs. normal predicted proportions (bar chart)
- **Column 2:** Sentiment breakdown *within* gambling-like comments
- **Column 3:** Word cloud of gambling-like comments only

**Run:**
```bash
python visualisation_svm.py
```

**Output:** `visualisation_svm.png` saved in the current working directory.

---

## Data Flow Summary

```
Raw CSV files (per platform)
        ↓
preprocess.py
        ↓
cleaned_comments_{platform}.csv
        ↓
vader_analysis.py
        ↓
comments_sentiment_{platform}.csv      ← also feeds visualisation.py
        ↓
tfidf_svm.py  (requires labelled Excel)
        ↓
comments_gambling_{platform}.csv       ← also feeds visualisation_svm.py
```

---

## Methodology Notes

### Why VADER (not TextBlob)?
VADER was specifically designed for social media text. It handles:
- ALL-CAPS emphasis (`HATE` scores more negatively than `hate`)
- Exclamation marks as intensity boosters
- Emojis (after translation to text in our pipeline)
- Contrastive conjunctions (e.g., "but")

TextBlob uses a standard English dictionary and misses gaming slang and social media conventions.

### Why TF-IDF + SVM?
- TF-IDF captures word importance within the corpus, including bigrams like `"just one"` or `"one more"` that are key gambling behaviour phrases
- SVM is well-suited to high-dimensional text classification
- Logistic Regression provides interpretable probability scores (risk scores) for each comment

### Why manual labelling?
Automatic keyword-based labelling is circular — the model would only learn to re-find the same keywords. Human labelling guided by psychological indicators (Table 3.1) allows the model to learn genuine linguistic patterns.

---

## Limitations

- The manually labelled dataset contains 299 comments (67 gambling-like), which is small for supervised learning. F1 scores of ~0.42–0.43 are expected and honest for this dataset size.
- VADER may misclassify sarcastic comments (e.g., *"Thanks Hoyo, I love getting another Tighnari"* reads as positive to VADER).
- Only English-language comments are analysed.
- Data represents a snapshot in time and does not capture temporal sentiment spikes around specific banner release events.

---

## Academic Context

This project is submitted in partial fulfilment of the requirements for the degree of Bachelor of Computer Science with Honours (Network Computing) at Universiti Malaysia Sarawak (UNIMAS), 2026.

**Report title:** *Sentiment Analysis of Gambling-Like Behaviour in Genshin Impact Gacha Community*

Key references:
- Hutto & Gilbert (2014) — VADER sentiment analysis
- Zendle & Cairns (2018) — Loot boxes and problem gambling
- Tang et al. (2022) — Psychological correlates in gacha gamers

---

## Ethical Statement

All data collected consists of publicly accessible comments only. No authentication or login-gated content was accessed. No personally identifiable information (usernames, profile links) is retained in the final dataset. Data is used strictly for academic research purposes.
