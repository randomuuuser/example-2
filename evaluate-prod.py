"""
Block 4b - Cross-validation, ablation, metrics.

Everything here is grouped by call. Segments from one call share speaker,
channel, line quality and often topic, so a random split leaks and returns an
optimistic score that will not survive contact with a new call.

Two prediction targets are supported and reported side by side:

    'wer'     predict the rate directly. This is what Waheed et al. do - their
              regressor outputs aWER, an approximated rate - and it is the mode
              to use.
    'errors'  predict the absolute error count, then divide by the target
              hypothesis length. Exploratory variant with NO counterpart in the
              paper; do not present it as one.

They are not equivalent: the label is errors / n_ref_words, while the
recovered estimate is errors / n_hyp_words. When the target system deletes or
inserts a lot, the two denominators drift apart. denominator_bias() quantifies
that gap on your data instead of assuming it away.
"""

import numpy as np
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import block_of, build_features


def make_model(name="hgb", seed=0):
    """
    Small-data regressors. No early stopping: with ~500 rows the internal
    validation split is too small to be a reliable stopping signal.
    """
    if name == "hgb":
        return HistGradientBoostingRegressor(
            max_depth=3, max_iter=200, learning_rate=0.05,
            min_samples_leaf=20, l2_regularization=1.0,
            early_stopping=False, random_state=seed,
        )
    if name == "xgb": ## NEW
        from xgboost import XGBRegressor

        return XGBRegressor(
            max_depth=3, n_estimators=200, learning_rate=0.05,
            min_child_weight=20, reg_lambda=1.0, subsample=0.8,
            colsample_bytree=0.8, objective="reg:absoluteerror",
            importance_type="gain", random_state=seed, n_jobs=-1,
        )
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "dummy":
        return DummyRegressor(strategy="mean")
    raise ValueError(f"unknown model: {name}")


def select_blocks(X, names, blocks):
    """Keep only the columns belonging to the requested feature blocks."""
    if not blocks:
        return np.zeros((len(X), 1)), []
    keep = [j for j, name in enumerate(names) if block_of(name) in blocks]
    return np.asarray(X)[:, keep], [names[j] for j in keep]


def splitter(groups, n_splits=5):
    """GroupKFold, or leave-one-call-out when there are few calls."""
    n_groups = len(set(groups))
    if n_splits >= n_groups:
        return LeaveOneGroupOut()
    return GroupKFold(n_splits=n_splits)


def cross_validate(rows, blocks=("proxy", "text"), target="errors",
                   model="hgb", n_splits=5, seed=0, roles=None):
    """
    Out-of-fold predictions with a call-grouped split.

    Returns a dict holding the OOF WER estimate, the true WER, the groups and
    the per-fold scores, so everything downstream reads from one object.
    """
    from features import PROXY_ROLES

    X, y_errors, y_wer, groups, names = build_features(
        rows, blocks=("proxy", "text"), roles=roles or PROXY_ROLES
    )
    X, used_names = select_blocks(X, names, blocks)
    y_errors = np.asarray(y_errors, dtype=float)
    y_wer = np.asarray(y_wer, dtype=float)
    groups = np.asarray(groups)
    n_hyp = np.array([max(r.get("n_hyp_words", 0), 1) for r in rows], dtype=float)

    y = y_errors if target == "errors" else y_wer
    predictions = np.zeros(len(y), dtype=float)
    fold_scores = []

    cv = splitter(groups, n_splits)
    for train_index, test_index in cv.split(X, y, groups):
        estimator = make_model(model, seed)
        estimator.fit(X[train_index], y[train_index])
        fold_prediction = estimator.predict(X[test_index])
        predictions[test_index] = fold_prediction
        fold_scores.append(
            float(np.mean(np.abs(fold_prediction - y[test_index])))
        )

    if target == "errors":
        predicted_errors = np.clip(predictions, 0, None)
        predicted_wer = np.clip(predicted_errors / n_hyp, 0, 2.0)
    else:
        predicted_wer = np.clip(predictions, 0, 2.0)
        predicted_errors = predicted_wer * n_hyp

    return {
        "blocks": tuple(blocks), "target": target, "model": model,
        "feature_names": used_names, "n_features": len(used_names),
        "groups": groups,
        "y_wer": y_wer, "pred_wer": predicted_wer,
        "y_errors": y_errors, "pred_errors": predicted_errors,
        "durations": np.array([r["duration"] for r in rows], dtype=float),
        "fold_mae": fold_scores,
        "n_folds": len(fold_scores),
    }


def segment_metrics(result):
    """Segment-level quality of the WER estimate."""
    y, prediction = result["y_wer"], result["pred_wer"]
    return {
        "MAE_wer": float(np.mean(np.abs(prediction - y))),
        "RMSE_wer": float(np.sqrt(np.mean((prediction - y) ** 2))),
        "MAE_errors": float(np.mean(np.abs(
            result["pred_errors"] - result["y_errors"]))),
        "pearson": float(stats.pearsonr(y, prediction)[0]),
        "spearman": float(stats.spearmanr(y, prediction)[0]),
        "fold_mae_std": float(np.std(result["fold_mae"])),
    }


def call_metrics(result):
    """
    Call-level aggregation, duration-weighted.

    This is the number that matters operationally: per call, the estimated WER
    weighted by segment duration, compared to the true one. WERR is the mean
    relative error of that aggregate.
    """
    rows = []
    for group in sorted(set(result["groups"])):
        mask = result["groups"] == group
        weights = result["durations"][mask]
        total = weights.sum()
        if total <= 0:
            continue
        true_wer = float(np.sum(result["y_wer"][mask] * weights) / total)
        estimated = float(np.sum(result["pred_wer"][mask] * weights) / total)
        rows.append({
            "call": group, "n_segments": int(mask.sum()),
            "wer_true": round(true_wer, 4),
            "wer_pred": round(estimated, 4),
            "abs_error": round(abs(estimated - true_wer), 4),
            "rel_error": round(abs(estimated - true_wer) / true_wer, 4)
            if true_wer > 0 else None,
        })

    relatives = [r["rel_error"] for r in rows if r["rel_error"] is not None]
    true_values = np.array([r["wer_true"] for r in rows])
    predicted_values = np.array([r["wer_pred"] for r in rows])
    summary = {
        "n_calls": len(rows),
        "WERR_mean": float(np.mean(relatives)) if relatives else None,
        "MAE_call": float(np.mean(np.abs(predicted_values - true_values))),
    }
    if len(rows) > 2:
        summary["spearman_call"] = float(
            stats.spearmanr(true_values, predicted_values)[0]
        )
    return rows, summary


def denominator_bias(rows):
    """
    How far n_ref_words and n_hyp_words drift apart.

    The label divides by the reference length; the deployed estimate divides
    by the hypothesis length. A ratio far from 1 means the 'errors' target
    carries a systematic bias that the 'wer' target avoids.
    """
    ratios = np.array([
        r["n_ref_words"] / max(r.get("n_hyp_words", 0), 1) for r in rows
    ], dtype=float)
    return {
        "ratio_mean": float(ratios.mean()),
        "ratio_median": float(np.median(ratios)),
        "ratio_p10": float(np.percentile(ratios, 10)),
        "ratio_p90": float(np.percentile(ratios, 90)),
        "pct_within_10pct": float(np.mean(np.abs(ratios - 1) < 0.1)),
    }


ABLATION_PLAN = [
    ("baseline (mean)", (), "dummy"),
    ("text only", ("text",), "hgb"),
    ("proxy only", ("proxy",), "hgb"),
    ("proxy + text", ("proxy", "text"), "hgb"),
    ("proxy + text, ridge", ("proxy", "text"), "ridge"),
]


def ablation(rows, target="errors", n_splits=5, seed=0, plan=None):
    """Run the ablation and return one row of metrics per configuration."""
    table = []
    for label, blocks, model in (plan or ABLATION_PLAN):
        result = cross_validate(rows, blocks=blocks, target=target,
                                model=model, n_splits=n_splits, seed=seed)
        metrics = segment_metrics(result)
        _, call_summary = call_metrics(result)
        table.append({
            "config": label,
            "n_feat": result["n_features"],
            "MAE_wer": round(metrics["MAE_wer"], 4),
            "RMSE_wer": round(metrics["RMSE_wer"], 4),
            "pearson": round(metrics["pearson"], 4),
            "spearman": round(metrics["spearman"], 4),
            "WERR": round(call_summary["WERR_mean"], 4)
            if call_summary["WERR_mean"] is not None else None,
        })
    return table


def single_proxy_ablation(rows, target="errors", n_splits=5):
    """
    Contribution of each proxy taken alone, then together.

    Worth running when the proxies differ in nature - a proxy that shares the
    target's architecture agrees with it for reasons that have nothing to do
    with correctness, and this is where that shows up.
    """
    from features import PROXY_ROLES

    table = []
    for roles in [(PROXY_ROLES[0],), (PROXY_ROLES[1],), PROXY_ROLES]:
        result = cross_validate(rows, blocks=("proxy", "text"), target=target,
                                model="hgb", n_splits=n_splits, roles=roles)
        metrics = segment_metrics(result)
        _, call_summary = call_metrics(result)
        table.append({
            "proxies": "+".join(roles),
            "MAE_wer": round(metrics["MAE_wer"], 4),
            "spearman": round(metrics["spearman"], 4),
            "WERR": round(call_summary["WERR_mean"], 4)
            if call_summary["WERR_mean"] is not None else None,
        })
    return table


def permutation_importance(rows, blocks=("proxy", "text"), target="errors",
                           n_splits=5, n_repeats=5, seed=0):
    """
    Drop in out-of-fold MAE when one feature is shuffled.

    Computed on the OOF predictions rather than on a single fit, so it reflects
    generalization rather than how hard the model leaned on a column in-sample.
    """
    rng = np.random.default_rng(seed)
    reference = cross_validate(rows, blocks, target, "hgb", n_splits, seed)
    base_mae = segment_metrics(reference)["MAE_wer"]

    from features import PROXY_ROLES

    X, y_errors, y_wer, groups, names = build_features(
        rows, blocks=("proxy", "text"), roles=PROXY_ROLES
    )
    X, used_names = select_blocks(X, names, blocks)
    y = np.asarray(y_errors if target == "errors" else y_wer, dtype=float)
    groups = np.asarray(groups)
    n_hyp = np.array([max(r.get("n_hyp_words", 0), 1) for r in rows], dtype=float)
    cv = splitter(groups, n_splits)

    scores = []
    for j, name in enumerate(used_names):
        deltas = []
        for _ in range(n_repeats):
            predictions = np.zeros(len(y))
            for train_index, test_index in cv.split(X, y, groups):
                estimator = make_model("hgb", seed)
                estimator.fit(X[train_index], y[train_index])
                shuffled = X[test_index].copy()
                shuffled[:, j] = rng.permutation(shuffled[:, j])
                predictions[test_index] = estimator.predict(shuffled)
            if target == "errors":
                predicted_wer = np.clip(np.clip(predictions, 0, None) / n_hyp, 0, 2)
            else:
                predicted_wer = np.clip(predictions, 0, 2)
            deltas.append(float(np.mean(np.abs(predicted_wer - y_wer))) - base_mae)
        scores.append((name, float(np.mean(deltas)), float(np.std(deltas))))

    scores.sort(key=lambda item: -item[1])
    return base_mae, scores






import numpy as np
from evaluate import cross_validate, make_model, select_blocks, splitter
from features import PROXY_ROLES, build_features
from sklearn.inspection import partial_dependence


def _matrix(rows, blocks, target, roles=PROXY_ROLES):
    """Feature matrix, target vector and groups, as used by cross_validate."""
    X, y_errors, y_wer, groups, names = build_features(rows, roles=roles)
    X, used = select_blocks(X, names, blocks)
    y = np.asarray(y_errors if target == "errors" else y_wer, dtype=float)
    return np.asarray(X), y, np.asarray(groups), used


def ridge_coefficients(rows, blocks=("proxy", "text"), target="wer", n_splits=5):
    """
    Ridge coefficients, refitted per fold and averaged.

    Features are standardized inside the pipeline, so coefficients compare
    directly: each is the change in predicted WER for a one-standard-deviation
    change in that feature. Fitting per fold gives the across-fold std - a
    coefficient whose std exceeds its mean is not a finding.
    """
    X, y, groups, names = _matrix(rows, blocks, target)
    cv = splitter(groups, n_splits)

    coefficients = []
    for train_index, _ in cv.split(X, y, groups):
        model = make_model("ridge").fit(X[train_index], y[train_index])
        coefficients.append(model[-1].coef_)      # [-1] = Ridge, [0] = scaler
    coefficients = np.array(coefficients)

    table = [
        {"feature": name,
         "coef": round(float(coefficients[:, j].mean()), 4),
         "std": round(float(coefficients[:, j].std()), 4),
         "stable": bool(abs(coefficients[:, j].mean()) > coefficients[:, j].std())}
        for j, name in enumerate(names)
    ]
    table.sort(key=lambda item: -abs(item["coef"]))
    return table


def partial_dependence_curve(rows, feature, blocks=("proxy", "text"),
                             target="wer", n_points=10, seed=0):
    """
    Shape of one feature's effect on the HGB prediction.

    Permutation importance says HOW MUCH a feature matters; this says IN WHICH
    DIRECTION and WITH WHAT SHAPE. HistGradientBoostingRegressor has no
    feature_importances_ attribute, so these two together are the full toolkit.
    """
    from sklearn.inspection import partial_dependence

    X, y, groups, names = _matrix(rows, blocks, target)
    if feature not in names:
        raise ValueError(f"unknown feature: {feature}")

    model = make_model("hgb", seed).fit(X, y)
    result = partial_dependence(model, X, [names.index(feature)],
                                grid_resolution=n_points, kind="average")

    return [{"value": round(float(v), 4), "predicted": round(float(p), 4)}
            for v, p in zip(result["grid_values"][0], result["average"][0])]


def xgb_importances(rows, blocks=("proxy", "text"), target="wer", seed=0):
    """Native gain-based importances of an XGBoost model fitted on all rows."""
    X, y, groups, names = _matrix(rows, blocks, target)
    model = make_model("xgb", seed).fit(X, y)
    table = sorted(zip(names, model.feature_importances_),
                   key=lambda item: -item[1])
    return [{"feature": n, "gain": round(float(g), 4)} for n, g in table]
