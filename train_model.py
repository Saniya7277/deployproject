import os
import glob
import joblib
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis, entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuration
# Configuration
UPLOADS_DIR = "data"
FALLBACK_PATH = r"C:\Users\saniy\OneDrive\Desktop\MiniProject\data\synthetic_autism_eeg.csv"
MODEL_FILE = "asd_model.pkl"
CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
TIME_COL = 'Time'
LABEL_COL = 'Label'
WINDOW_SEC = 2.0  # Match app.py
OVERLAP = 0.3  # Match app.py


BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}


def infer_fs(time_series):
    """Infer sampling frequency."""
    try:
        t = np.array(time_series.dropna(), dtype=float)
        if len(t) < 2:
            return 256.0
        dt = np.median(np.diff(t))
        return float(round(1.0 / dt, 2)) if dt > 0 else 256.0
    except:
        return 256.0


def bandpower_welch(signal, fs, band, nperseg=None):
    """Compute band power."""
    if len(signal) < 4:
        return 0.0
    try:
        nperseg = nperseg or min(128, len(signal))
        f, Pxx = welch(signal, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        low, high = band
        idx = np.logical_and(f >= low, f <= high)
        return float(np.trapz(Pxx[idx], f[idx])) if np.any(idx) else 0.0
    except:
        return 0.0


def window_generator(n_samples, fs, window_sec=2.0, overlap=0.3):
    """Generate sliding windows."""
    step = max(1, int(window_sec * fs * (1.0 - overlap)))
    wlen = int(window_sec * fs)
    if wlen <= 0 or step <= 0:
        return
    start = 0
    while start + wlen <= n_samples:
        yield start, start + wlen
        start += step


def extract_features_window(win_df, channels, fs):
    """Extract features from window - optimized."""
    feats = {}
    for ch in channels:
        sig = pd.to_numeric(win_df.get(ch, 0), errors='coerce').fillna(0).values
        
        if len(sig) == 0:
            sig = np.zeros(1)
        
        feats[f'{ch}_mean'] = float(np.mean(sig))
        feats[f'{ch}_std'] = float(np.std(sig))
        
        if len(sig) > 2:
            feats[f'{ch}_skew'] = float(skew(sig))
            feats[f'{ch}_kurt'] = float(kurtosis(sig))
        else:
            feats[f'{ch}_skew'] = 0.0
            feats[f'{ch}_kurt'] = 0.0
        
        try:
            hist, _ = np.histogram(sig, bins=16, density=True)
            hist = hist + 1e-12
            feats[f'{ch}_entropy'] = float(entropy(hist))
        except:
            feats[f'{ch}_entropy'] = 0.0
        
        for band_name, band_range in BANDS.items():
            feats[f'{ch}_bp_{band_name}'] = bandpower_welch(sig, fs, band_range)
    
    return feats


def label_majority(win_df, label_col):
    """Get majority label for window."""
    if label_col in win_df.columns:
        vals = win_df[label_col].dropna().astype(str)
        if len(vals) > 0:
            mode = vals.mode()
            return mode.iloc[0] if len(mode) > 0 else vals.iloc[0]
    return None


def find_training_files():
    """Find training CSV files."""
    files = []
    if os.path.isdir(UPLOADS_DIR):
        files += glob.glob(os.path.join(UPLOADS_DIR, "*.csv"))
    if os.path.exists(FALLBACK_PATH) and FALLBACK_PATH not in files:
        files.append(FALLBACK_PATH)
    return files


def build_feature_table(file_list):
    """Build feature table from files."""
    feat_rows = []
    labels = []
    
    for fp in file_list:
        print(f"Processing: {fp}")
        df = pd.read_csv(fp)
        
        colmap = {c.lower(): c for c in df.columns}
        time_name = colmap.get(TIME_COL.lower())
        fs = infer_fs(df[time_name]) if time_name else 256.0
        
        print(f"  Sampling rate: {fs} Hz")
        
        n_windows = 0
        for s, e in window_generator(len(df), fs, WINDOW_SEC, OVERLAP):
            win = df.iloc[s:e]
            win2 = pd.DataFrame()
            
            for ch in CHANNELS:
                real = colmap.get(ch.lower(), ch if ch in df.columns else None)
                win2[ch] = win[real] if real else 0.0
            
            label_real = colmap.get(LABEL_COL.lower())
            if label_real:
                win2[LABEL_COL] = win[label_real]
            
            feats = extract_features_window(win2, CHANNELS, fs)
            lab = label_majority(win2, LABEL_COL)
            
            feat_rows.append(feats)
            labels.append(lab)
            n_windows += 1
        
        print(f"  Generated {n_windows} windows")
    
    X = pd.DataFrame(feat_rows)
    y = pd.Series(labels)
    
    # Remove rows without labels
    mask = y.notnull()
    X = X[mask.values].reset_index(drop=True)
    y = y[mask.values].reset_index(drop=True).astype(str)
    
    # Encode labels
    y = y.str.strip().str.lower().map(lambda x: 1 if 'autis' in x else 0)
    
    return X, y


def main():
    """Main training function."""
    print("=" * 60)
    print("ASD EEG Detection - Model Training")
    print("=" * 60)
    
    files = find_training_files()
    if not files:
        raise FileNotFoundError("No training CSVs found.")
    
    print(f"\nFound {len(files)} training file(s):")
    for f in files:
        print(f"  - {f}")
    
    print("\nBuilding feature table...")
    X, y = build_feature_table(files)
    
    print(f"\nDataset summary:")
    print(f"  Features: {X.shape[1]}")
    print(f"  Windows: {X.shape[0]}")
    print(f"  Class distribution:")
    for cls, count in y.value_counts().items():
        label = "ASD" if cls == 1 else "Control"
        print(f"    {label}: {count} ({count/len(y)*100:.1f}%)")
    
    if len(y.unique()) < 2:
        raise ValueError("Need at least two classes to train.")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} windows")
    print(f"Test set: {len(X_test)} windows")
    
    # Train model with optimized parameters
    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Control', 'ASD']))
    
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_proba)
        print(f"\nROC AUC Score: {auc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"           Control   ASD")
    print(f"Actual")
    print(f"Control    {cm[0,0]:6d}  {cm[0,1]:4d}")
    print(f"ASD        {cm[1,0]:6d}  {cm[1,1]:4d}")
    
    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
    print(f"CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Save model
    output = {
        'model': model,
        'channels': CHANNELS,
        'feature_columns': list(X.columns)
    }
    joblib.dump(output, MODEL_FILE)
    
    print(f"\n✓ Model saved as '{MODEL_FILE}'")
    print("=" * 60)


if __name__ == "__main__":
    main()