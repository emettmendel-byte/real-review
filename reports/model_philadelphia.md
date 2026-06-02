# Suspicion model — philadelphia

LightGBM trained on Snorkel `p_suspicious` soft labels (`cross_entropy` objective), time-split at 2020.

## Test metrics (2020+ holdout)

| metric | value |
|---|---|
| train reviews (pre-2020) | 1,664,006 |
| test reviews (2020+) | 266,153 |
| test positive rate (p≥0.5) | 0.3021 |
| ROC AUC | 0.9992 |
| average precision | 0.9983 |
| logloss vs soft target | 0.2615 |
| Pearson(pred, soft) | 0.9978 |
| precision@100 | 1.000 |
| precision@1000 | 1.000 |

> **AUC is against the LabelModel's own binarized output**, so it measures how well the discriminative model reproduces and generalizes the weak labels — not ground truth. There is no labeled ground truth here; treat `p_fake` as a calibrated second opinion, never a verdict.

## Feature importance (gain)

| feature | gain frac | splits |
|---|---|---|
| total_reviews | 0.645 | 15928 |
| friend_count | 0.166 | 9150 |
| reviews_per_month | 0.056 | 16775 |
| account_age_days_at_review | 0.033 | 15613 |
| fan_count | 0.031 | 5346 |
| posting_hour_entropy | 0.025 | 23420 |
| fraction_extreme_ratings | 0.015 | 13926 |
| stars | 0.006 | 1906 |
| rating_skew | 0.006 | 21222 |
| rating_variance | 0.003 | 16326 |
| stars_delta_from_user_mean | 0.003 | 7319 |
| photo_count | 0.003 | 2108 |
| char_length | 0.002 | 6597 |
| stars_delta_from_business_mean | 0.002 | 4802 |
| max_sim_to_user_history | 0.001 | 8700 |
| exclamation_ratio | 0.001 | 5160 |
| avg_review_length | 0.001 | 14968 |
| word_count | 0.000 | 3532 |
| business_review_count_at_time | 0.000 | 3992 |
| max_sim_to_business_reviews | 0.000 | 3391 |

## SHAP global importance (mean \|SHAP\|, 5k-row sample)

| feature | mean abs SHAP |
|---|---|
| total_reviews | 1.2389 |
| friend_count | 0.7107 |
| account_age_days_at_review | 0.2863 |
| fan_count | 0.2606 |
| posting_hour_entropy | 0.2395 |
| fraction_extreme_ratings | 0.2136 |
| char_length | 0.1194 |
| reviews_per_month | 0.1153 |
| stars | 0.0906 |
| stars_delta_from_business_mean | 0.0900 |
| rating_skew | 0.0721 |
| stars_delta_from_user_mean | 0.0527 |
| photo_count | 0.0391 |
| rating_variance | 0.0389 |
| max_sim_to_user_history | 0.0227 |
| avg_review_length | 0.0157 |
| exclamation_ratio | 0.0154 |
| business_review_count_at_time | 0.0131 |
| word_count | 0.0119 |
| max_sim_to_business_reviews | 0.0033 |

## Validation — synthetic injection (feature-space)

Perturbed 2,000 benign test reviews into a botting feature profile (duplicate text, fresh account, no social, in burst, 5★).

| metric | value |
|---|---|
| base mean p_fake | 0.0322 |
| injected mean p_fake | 0.9958 |
| recall @ p_fake≥0.5 | 1.000 |
| recall @ p_fake≥0.8 | 1.000 |
| median lift | 0.9735 |

## Validation — held-out heuristic ablation (duplicate text)

Retrained without the embedding-similarity features; checked whether the model still ranks `lf_duplicate_text`-flagged reviews.

| model | AUC vs duplicate-text flag |
|---|---|
| full (with similarity features) | 0.5464 |
| ablated (no similarity features) | 0.4675 |

Duplicate-text positives in test: 1,919. Both AUCs sit near chance (0.5): `p_fake` barely tracks the duplicate-text flag even with the similarity features. The LabelModel weights this heuristic lightly and the gbdt gives the embedding-similarity features near-zero gain, so the duplicate-text signal is effectively independent of what the model learned — it is not reproduced rather than recovered-from-correlates.

## Manual audit — top 100 by p_fake (test set)

Eyeball these for precision@100. Columns: rank, p_fake, soft label, stars, char_length, account age at review, in burst, max sim to user history.

| # | p_fake | p_susp | stars | chars | acct_age_d | burst | sim_user |
|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 1.000 | 5 | 208 | 0 | 0 | 0.00 |
| 2 | 1.000 | 1.000 | 5 | 136 | 46 | 0 | 0.00 |
| 3 | 1.000 | 1.000 | 5 | 93 | 262 | 0 | 0.00 |
| 4 | 1.000 | 1.000 | 5 | 490 | 22 | 0 | 0.00 |
| 5 | 1.000 | 1.000 | 1 | 926 | 0 | 0 | 0.00 |
| 6 | 1.000 | 1.000 | 1 | 1318 | 0 | 0 | 0.24 |
| 7 | 1.000 | 1.000 | 5 | 228 | 1612 | 0 | 0.00 |
| 8 | 1.000 | 1.000 | 5 | 285 | 1608 | 0 | 0.00 |
| 9 | 1.000 | 1.000 | 5 | 285 | 1210 | 0 | 0.00 |
| 10 | 1.000 | 1.000 | 5 | 314 | 1547 | 0 | 0.00 |
| 11 | 1.000 | 1.000 | 5 | 293 | 1676 | 0 | 0.00 |
| 12 | 1.000 | 1.000 | 5 | 200 | 1547 | 0 | 0.00 |
| 13 | 1.000 | 1.000 | 5 | 285 | 1477 | 0 | 0.00 |
| 14 | 1.000 | 1.000 | 5 | 305 | 1376 | 0 | 0.00 |
| 15 | 1.000 | 1.000 | 5 | 362 | 1484 | 0 | 0.00 |
| 16 | 1.000 | 1.000 | 5 | 278 | 1648 | 0 | 0.00 |
| 17 | 1.000 | 1.000 | 5 | 300 | 1617 | 0 | 0.00 |
| 18 | 1.000 | 1.000 | 5 | 320 | 1589 | 0 | 0.00 |
| 19 | 1.000 | 1.000 | 5 | 287 | 1556 | 0 | 0.00 |
| 20 | 1.000 | 1.000 | 5 | 334 | 1382 | 0 | 0.00 |
| 21 | 1.000 | 1.000 | 5 | 206 | 1195 | 0 | 0.00 |
| 22 | 1.000 | 1.000 | 5 | 301 | 1417 | 0 | 0.00 |
| 23 | 1.000 | 1.000 | 5 | 266 | 1694 | 0 | 0.00 |
| 24 | 1.000 | 1.000 | 5 | 229 | 1506 | 0 | 0.00 |
| 25 | 1.000 | 1.000 | 5 | 231 | 1542 | 0 | 0.00 |
| 26 | 1.000 | 1.000 | 5 | 327 | 1725 | 0 | 0.00 |
| 27 | 1.000 | 1.000 | 5 | 361 | 1539 | 0 | 0.00 |
| 28 | 1.000 | 1.000 | 5 | 342 | 1717 | 0 | 0.00 |
| 29 | 1.000 | 1.000 | 5 | 282 | 1684 | 0 | 0.00 |
| 30 | 1.000 | 1.000 | 5 | 303 | 1543 | 0 | 0.00 |
| 31 | 1.000 | 1.000 | 5 | 355 | 1400 | 0 | 0.00 |
| 32 | 1.000 | 1.000 | 5 | 265 | 1721 | 0 | 0.00 |
| 33 | 1.000 | 1.000 | 5 | 318 | 1214 | 0 | 0.00 |
| 34 | 1.000 | 1.000 | 5 | 358 | 1645 | 0 | 0.00 |
| 35 | 1.000 | 1.000 | 5 | 308 | 1655 | 0 | 0.00 |
| 36 | 1.000 | 1.000 | 5 | 184 | 1317 | 0 | 0.00 |
| 37 | 1.000 | 1.000 | 5 | 264 | 1481 | 0 | 0.00 |
| 38 | 1.000 | 1.000 | 5 | 280 | 1658 | 0 | 0.00 |
| 39 | 1.000 | 1.000 | 5 | 141 | 1395 | 0 | 0.00 |
| 40 | 1.000 | 1.000 | 5 | 235 | 1794 | 0 | 0.00 |
| 41 | 1.000 | 1.000 | 5 | 334 | 1397 | 0 | 0.00 |
| 42 | 1.000 | 1.000 | 5 | 238 | 1723 | 0 | 0.00 |
| 43 | 1.000 | 1.000 | 5 | 338 | 1416 | 0 | 0.00 |
| 44 | 1.000 | 1.000 | 5 | 230 | 1307 | 0 | 0.00 |
| 45 | 1.000 | 1.000 | 5 | 316 | 1709 | 0 | 0.00 |
| 46 | 1.000 | 1.000 | 5 | 150 | 1282 | 0 | 0.00 |
| 47 | 1.000 | 1.000 | 5 | 152 | 1601 | 0 | 0.00 |
| 48 | 1.000 | 1.000 | 5 | 250 | 1706 | 0 | 0.00 |
| 49 | 1.000 | 1.000 | 5 | 404 | 1615 | 0 | 0.00 |
| 50 | 1.000 | 1.000 | 5 | 193 | 1646 | 0 | 0.00 |
| 51 | 1.000 | 1.000 | 5 | 199 | 1303 | 0 | 0.00 |
| 52 | 1.000 | 1.000 | 5 | 109 | 1642 | 0 | 0.00 |
| 53 | 1.000 | 1.000 | 5 | 322 | 1336 | 0 | 0.00 |
| 54 | 1.000 | 1.000 | 5 | 108 | 1486 | 0 | 0.00 |
| 55 | 1.000 | 1.000 | 5 | 318 | 1462 | 0 | 0.00 |
| 56 | 1.000 | 1.000 | 5 | 270 | 1765 | 0 | 0.00 |
| 57 | 1.000 | 1.000 | 5 | 442 | 1553 | 0 | 0.00 |
| 58 | 1.000 | 1.000 | 5 | 174 | 1589 | 0 | 0.00 |
| 59 | 1.000 | 1.000 | 5 | 242 | 1435 | 0 | 0.00 |
| 60 | 1.000 | 1.000 | 5 | 224 | 1765 | 0 | 0.00 |
| 61 | 1.000 | 1.000 | 5 | 273 | 1421 | 0 | 0.00 |
| 62 | 1.000 | 1.000 | 5 | 234 | 1588 | 0 | 0.00 |
| 63 | 1.000 | 1.000 | 5 | 407 | 1371 | 0 | 0.00 |
| 64 | 1.000 | 1.000 | 5 | 351 | 1445 | 0 | 0.00 |
| 65 | 1.000 | 1.000 | 5 | 335 | 1681 | 0 | 0.00 |
| 66 | 1.000 | 1.000 | 5 | 219 | 1476 | 0 | 0.00 |
| 67 | 1.000 | 1.000 | 5 | 116 | 1687 | 0 | 0.00 |
| 68 | 1.000 | 1.000 | 5 | 88 | 1467 | 0 | 0.00 |
| 69 | 1.000 | 1.000 | 5 | 93 | 1354 | 0 | 0.00 |
| 70 | 1.000 | 1.000 | 5 | 206 | 1513 | 0 | 0.00 |
| 71 | 1.000 | 1.000 | 5 | 401 | 1658 | 0 | 0.00 |
| 72 | 1.000 | 1.000 | 5 | 170 | 1721 | 0 | 0.00 |
| 73 | 1.000 | 1.000 | 5 | 297 | 1556 | 0 | 0.00 |
| 74 | 1.000 | 1.000 | 5 | 397 | 1618 | 0 | 0.00 |
| 75 | 1.000 | 1.000 | 5 | 299 | 1864 | 0 | 0.00 |
| 76 | 1.000 | 1.000 | 5 | 92 | 1714 | 0 | 0.00 |
| 77 | 1.000 | 1.000 | 5 | 158 | 1700 | 0 | 0.00 |
| 78 | 1.000 | 1.000 | 5 | 116 | 1623 | 0 | 0.00 |
| 79 | 1.000 | 1.000 | 5 | 399 | 1723 | 0 | 0.00 |
| 80 | 1.000 | 1.000 | 5 | 371 | 1748 | 0 | 0.00 |
| 81 | 1.000 | 1.000 | 5 | 227 | 1710 | 0 | 0.00 |
| 82 | 1.000 | 1.000 | 5 | 388 | 1696 | 0 | 0.00 |
| 83 | 1.000 | 1.000 | 5 | 365 | 1355 | 0 | 0.00 |
| 84 | 1.000 | 1.000 | 5 | 439 | 1630 | 0 | 0.00 |
| 85 | 1.000 | 1.000 | 5 | 329 | 1729 | 0 | 0.00 |
| 86 | 1.000 | 1.000 | 5 | 245 | 1771 | 0 | 0.00 |
| 87 | 1.000 | 1.000 | 5 | 229 | 1378 | 0 | 0.00 |
| 88 | 1.000 | 1.000 | 5 | 277 | 1309 | 0 | 0.00 |
| 89 | 1.000 | 1.000 | 5 | 182 | 1399 | 0 | 0.00 |
| 90 | 1.000 | 1.000 | 5 | 279 | 1712 | 0 | 0.00 |
| 91 | 1.000 | 1.000 | 5 | 106 | 1426 | 0 | 0.00 |
| 92 | 1.000 | 1.000 | 5 | 310 | 1617 | 0 | 0.00 |
| 93 | 1.000 | 1.000 | 5 | 175 | 1477 | 0 | 0.00 |
| 94 | 1.000 | 1.000 | 5 | 374 | 1593 | 0 | 0.00 |
| 95 | 1.000 | 1.000 | 5 | 366 | 1716 | 0 | 0.00 |
| 96 | 1.000 | 1.000 | 5 | 237 | 1767 | 0 | 0.00 |
| 97 | 1.000 | 1.000 | 5 | 136 | 1543 | 0 | 0.00 |
| 98 | 1.000 | 1.000 | 5 | 139 | 1611 | 0 | 0.00 |
| 99 | 1.000 | 1.000 | 5 | 142 | 1662 | 0 | 0.00 |
| 100 | 1.000 | 1.000 | 5 | 141 | 1721 | 0 | 0.00 |

## Known limitations

- **Temporal leakage in per-user aggregates.** These features are computed over each user's full history through the Jan-2022 snapshot, so a pre-2020 training row reflects later behavior: total_reviews, fan_count, photo_count, friend_count, reviews_per_month, rating_variance, rating_skew, avg_review_length, fraction_extreme_ratings, posting_hour_entropy. `account_age_days_snapshot` was dropped in favour of the point-in-time `account_age_days_at_review`; the rest are retained for signal and flagged here.

- **No ground truth.** Labels are weak (Snorkel over 10 heuristics); every metric is relative to those heuristics, not verified fake/genuine reviews.

- **Model settings.** 30 features, params: `{'bagging_fraction': 0.5780093202212182, 'bagging_freq': 1, 'feature_fraction': 0.7993292420985183, 'lambda_l1': 0.004207053950287938, 'lambda_l2': 0.0017073967431528124, 'learning_rate': 0.030710573677773714, 'min_child_samples': 144, 'num_leaves': 223, 'seed': 42}`.
