import numpy as np
from sklearn.linear_model import Ridge, RidgeCV

from cstims.encoding.ridge import (
    EncodingRidgeCV,
    EncodingRidgeReadoutCache,
    EncodingRidgeSpec,
    SchurCandidateReadoutCache,
    WeightedEncodingRidgeRefitCache,
    _build_kernel_eval_augmented_loo_cache,
)
from cstims.encoding.ridge_gcv_fast import RidgeCVFast


ALPHAS = np.array([0.05, 0.2, 1.0, 4.0])


def synthetic_problem(n_samples: int, n_features: int, *, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    coef = rng.normal(size=(n_features, 5))
    y = X @ coef + 0.4 * rng.normal(size=(n_samples, 5))
    sample_weight = rng.uniform(0.25, 2.5, size=n_samples)
    return X, y, sample_weight


def pearson_columns(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = prediction - prediction.mean(axis=0, keepdims=True)
    target = target - target.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(prediction * prediction, axis=0) * np.sum(target * target, axis=0))
    out = np.full(prediction.shape[1], np.nan, dtype=np.float64)
    ok = denom > 0
    out[ok] = np.sum(prediction[:, ok] * target[:, ok], axis=0) / denom[ok]
    return out


def score_columns(prediction: np.ndarray, target: np.ndarray, scoring: str) -> np.ndarray:
    if scoring == "pearson_r":
        return pearson_columns(prediction, target)
    return -np.mean((target - prediction) ** 2, axis=0)


def test_encoding_ridge_explicit_neg_mse_matches_sklearn_ridgecv():
    for n_samples, n_features, seed in [(36, 8, 1), (12, 36, 2)]:
        X, y, sample_weight = synthetic_problem(n_samples, n_features, seed=seed)
        for alpha_per_target in (False, True):
            ours = EncodingRidgeCV(
                alphas=ALPHAS,
                alpha_per_target=alpha_per_target,
                scoring="neg_mean_squared_error",
                standardize_x=False,
                standardize_y=False,
                fit_intercept=True,
            )
            ours.fit(X, y, sample_weight=sample_weight)

            sklearn_cv = RidgeCV(
                alphas=ALPHAS,
                scoring=None,
                alpha_per_target=alpha_per_target,
                fit_intercept=True,
            )
            sklearn_cv.fit(X, y, sample_weight=sample_weight)

            expected_alpha = np.asarray(sklearn_cv.alpha_, dtype=np.float64)
            if expected_alpha.ndim == 0:
                expected_alpha = np.full(y.shape[1], float(expected_alpha))
            np.testing.assert_allclose(ours.alpha_, expected_alpha, rtol=0, atol=0)
            np.testing.assert_allclose(ours.coef_, sklearn_cv.coef_, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(
                ours.intercept_,
                sklearn_cv.intercept_,
                rtol=1e-10,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                ours.predict(X),
                sklearn_cv.predict(X),
                rtol=1e-10,
                atol=1e-10,
            )


def test_encoding_ridge_default_pearson_matches_legacy_fast_path():
    X, y, _sample_weight = synthetic_problem(14, 40, seed=3)
    ours = EncodingRidgeCV(
        alphas=ALPHAS,
        alpha_per_target=True,
        standardize_x=False,
        standardize_y=False,
        fit_intercept=True,
    ).fit(X, y)
    legacy = RidgeCVFast(
        alphas=ALPHAS,
        scoring="pearson_r",
        alpha_per_target=True,
        fit_intercept=True,
    ).fit(X, y)
    np.testing.assert_allclose(ours.alpha_, np.asarray(legacy.alpha_), rtol=0, atol=0)
    np.testing.assert_allclose(ours.predict(X), legacy.predict(X), rtol=1e-9, atol=1e-9)


def test_encoding_ridge_fixed_alphas_match_explicit_ridge_refits():
    X, y, sample_weight = synthetic_problem(28, 9, seed=4)
    alpha_by_target = np.array([0.05, 0.2, 1.0, 4.0, 0.2])
    ours = EncodingRidgeCV(
        standardize_x=False,
        standardize_y=False,
        fit_intercept=True,
    ).fit_with_alphas(X, y, alpha_by_target, sample_weight=sample_weight)

    expected = np.empty_like(y)
    for target_idx, alpha in enumerate(alpha_by_target):
        ridge = Ridge(alpha=float(alpha), fit_intercept=True)
        ridge.fit(X, y[:, target_idx], sample_weight=sample_weight)
        expected[:, target_idx] = ridge.predict(X)
    np.testing.assert_allclose(ours.predict(X), expected, rtol=1e-10, atol=1e-10)


def test_readout_cache_selects_alphas_with_spec_scoring():
    rng = np.random.default_rng(5)
    X_train = rng.normal(size=(18, 7))
    X_val = rng.normal(size=(6, 7))
    X_eval = rng.normal(size=(4, 7))
    coef = rng.normal(size=(7, 4))
    y_train = X_train @ coef + 0.3 * rng.normal(size=(18, 4))
    y_val = X_val @ coef + 0.3 * rng.normal(size=(6, 4))

    for scoring in ("neg_mean_squared_error", "pearson_r"):
        spec = EncodingRidgeSpec.from_params(
            alphas=ALPHAS,
            scoring=scoring,
            standardize_x=False,
            standardize_y=False,
            fit_intercept=False,
        )
        cache = EncodingRidgeReadoutCache.from_features(
            spec=spec,
            X_train=X_train,
            X_val=X_val,
            eval_sets={"eval": X_eval},
            mode="independent",
            backend="kernel",
        )
        selection = cache.select_targetwise_alphas(y_train, y_val)
        pred_eval = cache.predict_eval(
            eval_key="eval",
            alpha_selection=selection,
            y_train=y_train,
        )
        assert pred_eval.shape == (X_eval.shape[0], y_train.shape[1])

        brute_scores = []
        for alpha in ALPHAS:
            ridge = Ridge(alpha=float(alpha), fit_intercept=False)
            ridge.fit(X_train, y_train)
            brute_scores.append(score_columns(ridge.predict(X_val), y_val, scoring))
        brute_scores = np.nan_to_num(np.stack(brute_scores, axis=0), nan=-np.inf)
        np.testing.assert_array_equal(selection.best_alpha_idx, np.argmax(brute_scores, axis=0))


def test_schur_candidate_cache_materializes_full_eval_augmented_ops():
    rng = np.random.default_rng(6)
    X_base = rng.normal(size=(12, 6))
    X_selected = rng.normal(size=(4, 6))
    X_candidates = rng.normal(size=(3, 6))

    cache = SchurCandidateReadoutCache.from_feature_blocks(
        X_base=X_base,
        X_selected=X_selected,
        X_candidates=X_candidates,
        alphas=ALPHAS,
    )
    for candidate_pos in range(X_candidates.shape[0]):
        materialized = cache.materialize_candidate(candidate_pos, dtype=np.float64)
        assert materialized is not None
        base_ops, eval_ops = materialized
        y_base = rng.normal(size=(X_base.shape[0], 5))
        y_eval = rng.normal(size=(X_selected.shape[0] + 1, 5))
        direct_pred = cache.predict_candidate(
            candidate_pos,
            y_base=y_base,
            y_eval=y_eval,
            dtype=np.float64,
        )
        assert direct_pred is not None
        op_pred = np.einsum("aeb,bt->aet", base_ops, y_base) + np.einsum(
            "aer,rt->aet",
            eval_ops,
            y_eval,
        )
        np.testing.assert_allclose(direct_pred, op_pred, rtol=1e-10, atol=1e-10)

        X_eval = np.vstack([X_selected, X_candidates[candidate_pos : candidate_pos + 1]])
        full_cache = _build_kernel_eval_augmented_loo_cache(
            X_train=X_base,
            X_val=X_selected[:2],
            X_base=X_base,
            eval_sets={"eval": X_eval},
            alphas=ALPHAS,
            backend="kernel",
        )
        for alpha_idx, alpha in enumerate(ALPHAS):
            expected_base, expected_eval = full_cache.materialized_eval_ops(
                float(alpha),
                "eval",
            )
            np.testing.assert_allclose(
                base_ops[alpha_idx],
                expected_base,
                rtol=1e-8,
                atol=1e-8,
            )
            np.testing.assert_allclose(
                eval_ops[alpha_idx],
                expected_eval,
                rtol=1e-8,
                atol=1e-8,
            )


def test_torch_backend_cpu_matches_sklearn_backend_when_available():
    try:
        import torch  # noqa: F401
    except Exception:
        return

    X, y, _sample_weight = synthetic_problem(12, 32, seed=7)
    sklearn_model = EncodingRidgeCV(
        alphas=ALPHAS,
        scoring="neg_mean_squared_error",
        standardize_x=True,
        standardize_y=False,
        fit_intercept=True,
        backend="sklearn",
    ).fit(X, y)
    torch_model = EncodingRidgeCV(
        alphas=ALPHAS,
        scoring="neg_mean_squared_error",
        standardize_x=True,
        standardize_y=False,
        fit_intercept=True,
        backend="torch",
        torch_device="cpu",
        torch_dtype="float64",
    ).fit(X, y)

    np.testing.assert_allclose(torch_model.alpha_, sklearn_model.alpha_, rtol=0, atol=0)
    np.testing.assert_allclose(
        torch_model.predict(X),
        sklearn_model.predict(X),
        rtol=1e-8,
        atol=1e-8,
    )


def test_weighted_refit_cache_matches_sklearn_refits_and_loso():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(18, 6))
    coef = rng.normal(size=(6, 4))
    y = X @ coef + 0.2 * rng.normal(size=(18, 4))
    weight = rng.uniform(0.2, 2.0, size=18)
    alphas = np.array([0.05, 0.2, 1.0, 4.0])
    X_eval = rng.normal(size=(5, 6))
    target_indices = np.array([1, 5, 12])
    X_target_eval = rng.normal(size=(3, 6))
    X_extra_eval = rng.normal(size=(4, 6))

    cache = WeightedEncodingRidgeRefitCache.fit(X, y, weight)
    heldout = cache.predict_heldout(X_eval, alphas)
    expected_heldout = np.empty_like(heldout, dtype=np.float64)
    for target_idx, alpha in enumerate(alphas):
        ridge = Ridge(alpha=float(alpha), fit_intercept=True)
        ridge.fit(X, y[:, target_idx], sample_weight=weight)
        expected_heldout[:, target_idx] = ridge.predict(X_eval)
    np.testing.assert_allclose(heldout, expected_heldout, rtol=1e-5, atol=1e-5)

    target_loso, extra_pred, target_hat = cache.predict_target_loso(
        target_indices=target_indices,
        X_target_eval=X_target_eval,
        alphas=alphas,
        X_extra_eval=X_extra_eval,
    )
    assert extra_pred is not None
    assert target_hat.shape == target_loso.shape

    expected_loso = np.empty_like(target_loso, dtype=np.float64)
    for row, train_idx in enumerate(target_indices):
        leave_weight = weight.copy()
        leave_weight[train_idx] = 0.0
        for target_idx, alpha in enumerate(alphas):
            ridge = Ridge(alpha=float(alpha), fit_intercept=True)
            ridge.fit(X, y[:, target_idx], sample_weight=leave_weight)
            expected_loso[row, target_idx] = ridge.predict(
                X_target_eval[row : row + 1]
            )[0]
    np.testing.assert_allclose(target_loso, expected_loso, rtol=1e-4, atol=1e-4)

    expected_extra = cache.predict_heldout(X_extra_eval, alphas)
    np.testing.assert_allclose(extra_pred, expected_extra, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    test_encoding_ridge_explicit_neg_mse_matches_sklearn_ridgecv()
    test_encoding_ridge_default_pearson_matches_legacy_fast_path()
    test_encoding_ridge_fixed_alphas_match_explicit_ridge_refits()
    test_readout_cache_selects_alphas_with_spec_scoring()
    test_schur_candidate_cache_materializes_full_eval_augmented_ops()
    test_torch_backend_cpu_matches_sklearn_backend_when_available()
    test_weighted_refit_cache_matches_sklearn_refits_and_loso()
    print("ok")
