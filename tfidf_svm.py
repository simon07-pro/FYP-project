# tfidf_svm.py
# Purpose: Train SVM and Logistic Regression models to classify gambling-like
#          behaviour in gacha community comments.
#
# Input:  comments_to_label.xlsx  (manually labelled by you using label_helper.py)
#         comments_sentiment_{platform}.csv  (VADER results from vader_analysis.py)
# Output: comments_gambling_{platform}.csv  (with SVM and LR predictions added)
#
# Why this is real machine learning (unlike the old keyword-based version):
#   - Labels come from YOUR human judgement, guided by Table 3.1 of the FYP report
#   - The model learns linguistic PATTERNS from your labels, not just keyword matching
#   - It is evaluated on a held-out test set it has never seen
#   - This directly addresses Research Objective 2 and Research Question 3

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Configuration
# ----------------------------
LABELLED_FILE = r"C:\Users\HP\Documents\fyp chap 4\potential_gambling_to_label (version 1).xlsx"

platform_folders = {
    "hoyolab": r"C:\Users\HP\Documents\fyp chap 4\hoyolab\data(csv)",
    "reddit":   r"C:\Users\HP\Documents\fyp chap 4\reddit\data(csv)",
    "youtube":  r"C:\Users\HP\Documents\fyp chap 4\youtube\data(csv)"
}


# ----------------------------
# Step 1: Load and validate manually labelled data
# ----------------------------
print("=== Step 1: Loading manually labelled data ===")

if not os.path.exists(LABELLED_FILE):
    print(f"Error: Labelled file not found at {LABELLED_FILE}")
    print("Please run label_helper.py first, label the Excel file, then run this script.")
    exit()

df_labelled = pd.read_excel(LABELLED_FILE, sheet_name='Label Here')

# Check the label column exists and has values
if 'gambling_label' not in df_labelled.columns:
    print("Error: 'gambling_label' column not found in Excel file.")
    exit()

# Remove any rows where label was left blank (not yet labelled)
df_labelled = df_labelled.dropna(subset=['gambling_label'])
df_labelled = df_labelled[df_labelled['gambling_label'].astype(str).str.strip() != '']

# Convert labels to integer (in case Excel stored as float)
df_labelled['gambling_label'] = df_labelled['gambling_label'].astype(int)

# Validate labels are only 0 or 1
valid_labels = df_labelled['gambling_label'].isin([0, 1])
if not valid_labels.all():
    bad_rows = df_labelled[~valid_labels]
    print(f"Error: Found {len(bad_rows)} rows with invalid labels (not 0 or 1).")
    print(bad_rows[['row_number', 'gambling_label', 'cleaned_text']].head())
    exit()

print(f"  Total labelled comments loaded: {len(df_labelled)}")
label_counts = df_labelled['gambling_label'].value_counts()
print(f"  Label 0 (normal behaviour):   {label_counts.get(0, 0)} comments")
print(f"  Label 1 (gambling-like):       {label_counts.get(1, 0)} comments")

# Warn if labels are very imbalanced
ratio = label_counts.get(1, 0) / max(len(df_labelled), 1)
if ratio < 0.2 or ratio > 0.8:
    print(f"\n  Warning: Labels are imbalanced ({ratio:.0%} are class 1).")
    print(f"  Consider labelling more of the minority class for better model performance.")


# ----------------------------
# Step 2: TF-IDF Feature Extraction
# ----------------------------
# TF-IDF converts each comment into a numerical vector.
# Each unique word becomes a dimension. The value represents how important
# that word is in this comment relative to all other comments.
# n-gram range (1,2) means we capture single words AND two-word phrases
# e.g. "just one" and "one more" as separate features, not just "just", "one", "more"
# This is important because "just one more" is a key gambling behaviour phrase.
# Justified by Section 3.5.3 of the FYP report.
# ----------------------------
print("\n=== Step 2: TF-IDF Feature Extraction ===")

vectorizer = TfidfVectorizer(
    max_features=1000,      # keep top 1000 most informative words/phrases
    ngram_range=(1, 2),     # capture single words AND two-word phrases
    min_df=2,               # ignore words that appear in fewer than 2 comments
    sublinear_tf=True       # apply log scaling to term frequencies
)

X = vectorizer.fit_transform(df_labelled['cleaned_text'].astype(str))
y = df_labelled['gambling_label']

print(f"  Feature matrix shape: {X.shape}")
print(f"  ({X.shape[0]} comments × {X.shape[1]} TF-IDF features)")


# ----------------------------
# Step 3: Train/Test Split
# ----------------------------
# 80% of data used to train the model
# 20% held out to evaluate performance on unseen data
# stratify=y ensures both splits have the same ratio of 0s and 1s
# ----------------------------
print("\n=== Step 3: Train/Test Split ===")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # keep same class ratio in both train and test
)

print(f"  Training set: {X_train.shape[0]} comments")
print(f"  Test set:     {X_test.shape[0]} comments")


# ----------------------------
# Step 4: Handle class imbalance
# ----------------------------
# If you labelled more 0s than 1s (or vice versa), the model can be biased.
# class_weight='balanced' tells the model to pay more attention to the
# minority class during training, compensating for the imbalance.
# ----------------------------
classes = np.array([0, 1])
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"\n  Class weights applied: {class_weight_dict}")


# ----------------------------
# Step 5: Train SVM
# ----------------------------
# SVM finds the optimal boundary (hyperplane) that separates gambling-like
# comments from normal comments in the high-dimensional TF-IDF space.
# kernel='linear' works well for text classification.
# Justified by Section 2.5.3 and Section 3.7.1 of the FYP report.
# ----------------------------
print("\n=== Step 4: SVM Training and Evaluation ===")

clf_svm = SVC(
    kernel='linear',
    probability=True,
    class_weight='balanced',  # handle class imbalance
    random_state=42
)
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)

print(f"\n  --- SVM Results ---")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print()
print(classification_report(y_test, y_pred_svm,
      target_names=['Normal (0)', 'Gambling-like (1)']))

# Cross-validation gives a more reliable accuracy estimate
# It tests the model 5 times on different splits of the data
svm_cv_scores = cross_val_score(clf_svm, X, y, cv=5, scoring='f1')
print(f"  5-fold Cross-validation F1: {svm_cv_scores.mean():.4f} (+/- {svm_cv_scores.std():.4f})")


# ----------------------------
# Step 6: Train Logistic Regression
# ----------------------------
# Logistic Regression is used as a comparison model.
# It produces probability scores (risk scores) which can indicate
# HOW LIKELY a comment is to show gambling behaviour.
# Justified by Section 2.5.3 of the FYP report.
# ----------------------------
print("\n=== Step 5: Logistic Regression Training and Evaluation ===")

clf_lr = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
clf_lr.fit(X_train, y_train)
y_pred_lr = clf_lr.predict(X_test)

print(f"\n  --- Logistic Regression Results ---")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print()
print(classification_report(y_test, y_pred_lr,
      target_names=['Normal (0)', 'Gambling-like (1)']))

lr_cv_scores = cross_val_score(clf_lr, X, y, cv=5, scoring='f1')
print(f"  5-fold Cross-validation F1: {lr_cv_scores.mean():.4f} (+/- {lr_cv_scores.std():.4f})")


# ----------------------------
# Step 7: Show most informative words
# ----------------------------
# This shows WHICH words the model learned are most associated with
# gambling-like behaviour. This directly answers Research Question 3:
# "What linguistic patterns indicate gambling-like behaviour?"
# ----------------------------
print("\n=== Step 6: Most Informative Features (Logistic Regression) ===")

feature_names = vectorizer.get_feature_names_out()
lr_coefs = clf_lr.coef_[0]

# Top words pushing towards gambling-like (label 1)
top_gambling = sorted(zip(lr_coefs, feature_names), reverse=True)[:15]
print("\n  Top words/phrases indicating GAMBLING-LIKE behaviour (label 1):")
for coef, word in top_gambling:
    print(f"    {word:<30} score: {coef:.4f}")

# Top words pushing towards normal (label 0)
top_normal = sorted(zip(lr_coefs, feature_names))[:15]
print("\n  Top words/phrases indicating NORMAL behaviour (label 0):")
for coef, word in top_normal:
    print(f"    {word:<30} score: {coef:.4f}")


# ----------------------------
# Step 8: Apply model to ALL platform data and save
# ----------------------------
print("\n=== Step 7: Applying model to all platform data ===")

for platform, folder in platform_folders.items():
    # Read the VADER sentiment file (output of vader_analysis.py)
    # We add the SVM predictions on top of the VADER results
    result_folder = os.path.join(os.path.dirname(folder), "result")
    sentiment_file = os.path.join(result_folder, f"comments_sentiment_{platform}.csv")

    if not os.path.exists(sentiment_file):
        print(f"\n  Warning: {platform} sentiment file not found.")
        print(f"  Please run vader_analysis.py first.")
        continue

    print(f"\n  Processing {platform}...")
    df_platform = pd.read_csv(sentiment_file, encoding='utf-8', on_bad_lines='skip')

    if 'cleaned_text' not in df_platform.columns:
        print(f"  Warning: 'cleaned_text' column not found in {platform} file, skipping.")
        continue

    df_platform = df_platform.dropna(subset=['cleaned_text'])

    # Transform platform comments using the SAME vectorizer fitted on labelled data
    X_platform = vectorizer.transform(df_platform['cleaned_text'].astype(str))

    # Predict gambling-like behaviour
    df_platform['gambling_pred_svm'] = clf_svm.predict(X_platform)
    df_platform['gambling_pred_lr']  = clf_lr.predict(X_platform)

    # Also save the probability score (0.0 to 1.0) as a "risk score"
    # Higher score = higher likelihood of gambling-like behaviour
    df_platform['gambling_risk_svm'] = clf_svm.predict_proba(X_platform)[:, 1]
    df_platform['gambling_risk_lr']  = clf_lr.predict_proba(X_platform)[:, 1]

    # Save results
    os.makedirs(result_folder, exist_ok=True)
    output_path = os.path.join(result_folder, f"comments_gambling_{platform}.csv")
    df_platform.to_csv(output_path, index=False)

    # Print summary
    svm_gambling = df_platform['gambling_pred_svm'].sum()
    total = len(df_platform)
    print(f"  Total comments: {total}")
    print(f"  SVM predicted gambling-like: {svm_gambling} ({svm_gambling/total*100:.1f}%)")
    print(f"  Saved to: {output_path}")

print("\n=== SVM and Logistic Regression complete ===")
print("Columns added: gambling_pred_svm, gambling_pred_lr, gambling_risk_svm, gambling_risk_lr")
print("\nNew columns explained:")
print("  gambling_pred_svm/lr : 1 = gambling-like behaviour, 0 = normal")
print("  gambling_risk_svm/lr : probability score 0.0-1.0 (higher = more gambling-like)")