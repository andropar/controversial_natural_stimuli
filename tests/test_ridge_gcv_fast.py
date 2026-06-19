import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import make_scorer

from cstims.encoding.ridge_gcv_fast import RidgeCVFast


ALPHAS = np.logspace(-2, 3, 8)


def pearson_score(y_true, y_pred, sample_weight=None):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = y_true - y_true.mean()
    y_pred = y_pred - y_pred.mean()
    denom = np.sqrt(np.sum(y_true * y_true) * np.sum(y_pred * y_pred))
    if denom == 0:
        return np.nan
    return float(np.sum(y_true * y_pred) / denom)


def synthetic_problem(n_samples: int, n_features: int, *, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    coef = rng.normal(size=(n_features, 4))
    y = X @ coef + 0.25 * rng.normal(size=(n_samples, 4))
    sample_weight = rng.uniform(0.2, 3.0, size=n_samples)
    return X, y, sample_weight


def assert_matches_sklearn_and_explicit_ridge(X, y, sample_weight):
    fast = RidgeCVFast(
        alphas=ALPHAS,
        scoring="pearson_r",
        alpha_per_target=True,
        fit_intercept=True,
    )
    fast.fit(X, y, sample_weight=sample_weight)

    sklearn_cv = RidgeCV(
        alphas=ALPHAS,
        scoring=make_scorer(pearson_score),
        alpha_per_target=True,
        fit_intercept=True,
        cv=None,
    )
    sklearn_cv.fit(X, y, sample_weight=sample_weight)

    np.testing.assert_allclose(fast.alpha_, sklearn_cv.alpha_, rtol=0, atol=1e-12)

    fast_pred = fast.predict(X)
    for target_idx, alpha in enumerate(np.atleast_1d(fast.alpha_)):
        ridge = Ridge(alpha=float(alpha), fit_intercept=True)
        ridge.fit(X, y[:, target_idx], sample_weight=sample_weight)
        expected = ridge.predict(X)
        np.testing.assert_allclose(
            fast_pred[:, target_idx],
            expected,
            rtol=1e-8,
            atol=1e-8,
        )


def test_ridgecvfast_unweighted_n_greater_than_p():
    X, y, _sample_weight = synthetic_problem(40, 7, seed=1)
    assert_matches_sklearn_and_explicit_ridge(X, y, sample_weight=None)


def test_ridgecvfast_unweighted_p_greater_than_n():
    X, y, _sample_weight = synthetic_problem(12, 40, seed=2)
    assert_matches_sklearn_and_explicit_ridge(X, y, sample_weight=None)


def test_ridgecvfast_weighted_n_greater_than_p():
    X, y, sample_weight = synthetic_problem(40, 7, seed=3)
    assert_matches_sklearn_and_explicit_ridge(X, y, sample_weight=sample_weight)


def test_ridgecvfast_weighted_p_greater_than_n():
    X, y, sample_weight = synthetic_problem(12, 40, seed=4)
    assert_matches_sklearn_and_explicit_ridge(X, y, sample_weight=sample_weight)


if __name__ == "__main__":
    test_ridgecvfast_unweighted_n_greater_than_p()
    test_ridgecvfast_unweighted_p_greater_than_n()
    test_ridgecvfast_weighted_n_greater_than_p()
    test_ridgecvfast_weighted_p_greater_than_n()
    print("ok")
