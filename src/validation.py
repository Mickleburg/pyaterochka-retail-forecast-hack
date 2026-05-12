import pandas as pd

from .config import FOLDS, ID_COL, MONTH_COL, TARGET_COL
from .features import build_features, get_cat_features, get_feature_columns
from .metrics import mape_percent
from .models import get_model


def run_time_validation(df: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    rows = []
    fold_frames = []
    for fold in FOLDS:
        valid_month = fold["valid_month"]
        train_end_month = fold["train_end_month"]
        fold_df = df[df[MONTH_COL] <= valid_month].copy()
        feat_df = build_features(fold_df)
        feature_columns = get_feature_columns(feat_df)
        cat_features = get_cat_features(feature_columns)
        train_mask = (feat_df[MONTH_COL] <= train_end_month) & feat_df[TARGET_COL].notna()
        valid_mask = feat_df[MONTH_COL] == valid_month
        fold_frames.append((fold, feat_df, feature_columns, cat_features, train_mask, valid_mask))

    for model_name in model_names:
        fold_scores = []
        for fold, feat_df, feature_columns, cat_features, train_mask, valid_mask in fold_frames:
            x_train = feat_df.loc[train_mask, feature_columns]
            y_train = feat_df.loc[train_mask, TARGET_COL]
            x_valid = feat_df.loc[valid_mask, feature_columns]
            y_valid = feat_df.loc[valid_mask, TARGET_COL]

            model = get_model(model_name)
            model.fit(x_train, y_train, cat_features=cat_features)
            pred = model.predict(x_valid)
            score = mape_percent(y_valid, pred)
            fold_scores.append(score)
            rows.append(
                {
                    "model": model_name,
                    "backend": getattr(model, "backend", model.__class__.__name__),
                    "fold": fold["fold"],
                    "train_months": f"<= {fold['train_end_month']}",
                    "valid_month": fold["valid_month"],
                    "mape_percent": score,
                }
            )

        rows.append(
            {
                "model": model_name,
                "backend": "mean",
                "fold": "mean",
                "train_months": "",
                "valid_month": "",
                "mape_percent": sum(fold_scores) / len(fold_scores),
            }
        )
    return pd.DataFrame(rows)


def summarize_train(df: pd.DataFrame) -> dict:
    months_per_store = df.groupby(ID_COL)[MONTH_COL].nunique()
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "month_min": int(df[MONTH_COL].min()),
        "month_max": int(df[MONTH_COL].max()),
        "unique_stores": int(df[ID_COL].nunique()),
        "missing": df.isna().sum().to_dict(),
        "months_per_store_min": int(months_per_store.min()),
        "months_per_store_max": int(months_per_store.max()),
        "stores_with_incomplete_history": int((months_per_store < df[MONTH_COL].nunique()).sum()),
    }
