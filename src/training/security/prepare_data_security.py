import re
from pathlib import Path

import pandas as pd
from datasets import concatenate_datasets, load_dataset
from sklearn.model_selection import train_test_split

# Configuration & Paths
RANDOM_SEED = 42
TEST_SIZE = 0.15  # 85% Train / 15% Test stratified across source + label
BASE_DIR = Path(__file__).resolve().parent
EXCEL_PATH = BASE_DIR / "PromptInjectionPrompts.xlsx" #downloaed sillaannregi/hindi-and-hinglish-prompts
TRAIN_OUTPUT_PATH = BASE_DIR / "train.parquet"
TEST_OUTPUT_PATH = BASE_DIR / "test.parquet"


def normalize_label(val) -> int:
    """Standardizes string, boolean, or numeric labels into binary 0 (Benign) or 1 (Injection)."""
    if pd.isna(val):
        return -1
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return 1 if int(val) >= 1 else 0
    if isinstance(val, bool):
        return 1 if val else 0

    s = str(val).strip().lower()
    if s in {"1", "true", "yes", "malicious", "injection", "prompt_injection", "jailbreak", "attack", "unsafe", "harmful"}:
        return 1
    if s in {"0", "false", "no", "benign", "safe", "normal", "clean", "legitimate", "harmless"}:
        return 0
    return -1


def extract_text_and_label(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Auto-detects text and label columns across varying dataset schemas."""
    col_map = {c.lower().strip(): c for c in df.columns}

    # Candidate column names for prompt text and binary label
    text_candidates = ["text", "prompt", "prompts", "input", "query", "content", "instruction", "sentence", "user_prompt"]
    label_candidates = ["label", "labels", "is_attack", "is_injection", "type", "category", "class", "classification", "target", "malicious"]

    text_col = next((col_map[c] for c in text_candidates if c in col_map), None)
    label_col = next((col_map[c] for c in label_candidates if c in col_map), None)

    # Fallback to first two columns if custom headers are used (e.g., in Excel sheets)
    if text_col is None or label_col is None:
        if len(df.columns) >= 2:
            text_col = text_col or df.columns[0]
            label_col = label_col or df.columns[1]
        else:
            raise ValueError(f"[{source_name}] Could not identify text and label columns in: {list(df.columns)}")

    out = pd.DataFrame()
    out["text"] = df[text_col].astype(str)
    out["label"] = df[label_col].apply(normalize_label)
    out["source"] = source_name

    # Filter out invalid/empty rows or unmapped labels
    out = out[out["label"].isin([0, 1])].copy()
    out["text"] = out["text"].str.strip()
    out = out[out["text"].str.len() >= 3].copy()
    return out[["text", "label", "source"]]


def load_hf_dataset_all_splits(repo_id: str, source_name: str, config_name: str = None) -> pd.DataFrame:
    """Loads all splits (train/validation/test) from a Hugging Face dataset so we can globally deduplicate and stratify."""
    print(f"[*] Loading Hugging Face dataset: {repo_id} ...")
    try:
        ds_dict = load_dataset(repo_id, config_name) if config_name else load_dataset(repo_id)
    except Exception:
        # Fallback to default config if named subset is not required
        ds_dict = load_dataset(repo_id)

    if isinstance(ds_dict, dict) or hasattr(ds_dict, "keys"):
        combined_ds = concatenate_datasets([ds_dict[split] for split in ds_dict.keys()])
        df = combined_ds.to_pandas()
    else:
        df = ds_dict.to_pandas()

    cleaned_df = extract_text_and_label(df, source_name)
    print(f"    -> Loaded {len(cleaned_df):,} valid rows from {repo_id}")
    return cleaned_df


def load_hindi_hinglish_excel(excel_path: Path | str = EXCEL_PATH) -> pd.DataFrame:
    """Loads Hindi and Hinglish prompts across one or multiple sheets in PromptInjectionPrompts.xlsx."""
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Could not find Excel file at: {excel_path}")

    print(f"[*] Loading Indic/Hinglish dataset from Excel: {excel_path} ...")
    sheets_dict = pd.read_excel(excel_path, sheet_name=None)  # Reads all sheets if multiple exist (e.g., Hindi & Hinglish)
    frames = []

    for sheet_name, sheet_df in sheets_dict.items():
        if sheet_df.empty:
            continue
        try:
            parsed = extract_text_and_label(sheet_df, source_name="sillaannregi_hindi_hinglish")
            frames.append(parsed)
        except ValueError:
            # Handle case where sheet has columns side-by-side (e.g. Safe_Prompts | Malicious_Prompts)
            for col in sheet_df.columns:
                col_lower = str(col).lower()
                if "malicious" in col_lower or "injection" in col_lower or "attack" in col_lower or "unsafe" in col_lower:
                    sub = pd.DataFrame({"text": sheet_df[col].dropna().astype(str), "label": 1, "source": "sillaannregi_hindi_hinglish"})
                    frames.append(sub)
                elif "safe" in col_lower or "benign" in col_lower or "normal" in col_lower:
                    sub = pd.DataFrame({"text": sheet_df[col].dropna().astype(str), "label": 0, "source": "sillaannregi_hindi_hinglish"})
                    frames.append(sub)

    if not frames:
        raise ValueError(f"No valid rows could be parsed from {excel_path}. Check column headers.")

    hindi_df = pd.concat(frames, ignore_index=True)
    hindi_df["text"] = hindi_df["text"].str.strip()
    hindi_df = hindi_df[hindi_df["text"].str.len() >= 3].copy()
    print(f"    -> Loaded {len(hindi_df):,} valid rows from {excel_path.name}")
    return hindi_df


def clean_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes whitespace, removes contradictory labels across datasets, and drops exact duplicates."""
    print(f"\n[*] Running global deduplication and label contradiction purge on {len(df):,} total rows...")

    # Normalize unicode whitespace and create canonical fingerprint for deduplication
    df["text"] = df["text"].apply(lambda x: re.sub(r"\s+", " ", str(x)).strip())
    df["fingerprint"] = df["text"].str.lower()

    # 1. Purge Label Contradictions (if the exact same text is labeled 0 in one dataset and 1 in another)
    label_counts_per_text = df.groupby("fingerprint")["label"].nunique()
    contradictory_fingerprints = set(label_counts_per_text[label_counts_per_text > 1].index)
    if contradictory_fingerprints:
        print(f"    [!] Purging {len(contradictory_fingerprints):,} prompts with conflicting labels across datasets.")
        df = df[~df["fingerprint"].isin(contradictory_fingerprints)].copy()

    # 2. Drop Exact Duplicates (keep first occurrence)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["fingerprint"], keep="first").drop(columns=["fingerprint"]).reset_index(drop=True)
    print(f"    -> Removed {before_dedup - len(df):,} duplicate prompts. Clean dataset size: {len(df):,} rows.")
    return df


def main():
    # 1. Load all 4 datasets
    df_rikka = load_hf_dataset_all_splits(
        "rikka-snow/prompt-injection-multilingual",
        source_name="rikka_multilingual"
    )
    df_neuralchemy = load_hf_dataset_all_splits(
        "neuralchemy/Prompt-injection-dataset",
        source_name="neuralchemy_core",
        config_name="core"
    )
    df_prodnull = load_hf_dataset_all_splits(
        "prodnull/prompt-injection-repo-dataset",
        source_name="prodnull_repo_code"
    )
    df_hindi = load_hindi_hinglish_excel(EXCEL_PATH)

    # 2. Merge all datasets
    full_df = pd.concat([df_rikka, df_hindi, df_neuralchemy, df_prodnull], ignore_index=True)

    # 3. Clean, purge contradictions, and deduplicate globally
    full_df = clean_and_deduplicate(full_df)

    # 4. Joint Stratification key: (source + "_" + label)
    # This guarantees that EVERY dataset (including Hindi/Hinglish and Hard-Benign Code)
    # AND both classes (0 and 1) are proportionally represented in both train and test splits!
    full_df["stratify_key"] = full_df["source"] + "_" + full_df["label"].astype(str)

    train_df, test_df = train_test_split(
        full_df,
        test_size=TEST_SIZE,
        stratify=full_df["stratify_key"],
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    # 5. Drop helper column and randomize (shuffle) both splits thoroughly
    train_df = train_df.drop(columns=["stratify_key"]).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    test_df = test_df.drop(columns=["stratify_key"]).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    # 6. Save as train.parquet and test.parquet in the same directory
    train_df.to_parquet(TRAIN_OUTPUT_PATH, index=False, engine="pyarrow")
    test_df.to_parquet(TEST_OUTPUT_PATH, index=False, engine="pyarrow")

    # 7. Print Summary Audit
    print("\n" + "=" * 65)
    print("SECURITY DATASET PREPARATION COMPLETE")
    print("=" * 65)
    print(f"Saved Train Split -> {TRAIN_OUTPUT_PATH.resolve()} ({len(train_df):,} rows)")
    print(f"Saved Test Split  -> {TEST_OUTPUT_PATH.resolve()} ({len(test_df):,} rows)")

    print("\n--- Train Split Breakdown (Source x Label) ---")
    print(pd.crosstab(train_df["source"], train_df["label"], margins=True))

    print("\n--- Test Split Breakdown (Source x Label) ---")
    print(pd.crosstab(test_df["source"], test_df["label"], margins=True))


if __name__ == "__main__":
    main()
