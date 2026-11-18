"""Transformer tùy biến tương thích scikit-learn (fit CHỈ trên train → transform val/test)."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class IQRClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Winsorizing theo hàng rào Tukey [Q1 − k·IQR, Q3 + k·IQR] có hai chốt an toàn.

    1. `max_tail`: không bao giờ cắt quá `max_tail` (mặc định 1%) số mẫu ở mỗi đuôi. Biến có phân phối
       hẹp nhưng đuôi dày (vd. `upper_ratio`) sẽ không bị "nghiền" mất thông tin.
    2. IQR = 0 (biến đếm lệch nặng, đa số = 0): hàng rào trùng Q1/Q3 sẽ biến cột thành hằng số.
       Khi đó dùng phân vị `fallback_q` (0.1% / 99.9%), nếu vẫn trùng thì dùng min/max (không cắt).

    Thuộc tính sau fit: `lower_`, `upper_` (biên cắt cho từng cột).
    """

    def __init__(self, k: float = 3.0, max_tail: float = 0.01, fallback_q: float = 0.001):
        self.k = k
        self.max_tail = max_tail
        self.fallback_q = fallback_q

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=np.float64, ensure_all_finite=True)
        q1, q3 = np.percentile(X, [25, 75], axis=0)
        iqr = q3 - q1
        tail_lo, tail_hi = np.quantile(X, [self.max_tail, 1 - self.max_tail], axis=0)
        fb_lo, fb_hi = np.quantile(X, [self.fallback_q, 1 - self.fallback_q], axis=0)
        xmin, xmax = X.min(axis=0), X.max(axis=0)

        upper = np.maximum(q3 + self.k * iqr, tail_hi)                       # chốt (1)
        lower = np.minimum(q1 - self.k * iqr, tail_lo)
        upper = np.where(upper <= q3, np.where(fb_hi > q3, fb_hi, xmax), upper)  # chốt (2)
        lower = np.where(lower >= q1, np.where(fb_lo < q1, fb_lo, xmin), lower)
        self.lower_, self.upper_ = lower, upper
        return self

    def transform(self, X):
        check_is_fitted(self, ['lower_', 'upper_'])
        X = validate_data(self, X, dtype=np.float64, reset=False)
        return np.clip(X, self.lower_, self.upper_)

    def clipped_fraction(self, X) -> np.ndarray:
        """Tỉ lệ giá trị bị cắt ở mỗi cột (để báo cáo)."""
        X = np.asarray(X, dtype=np.float64)
        return ((X < self.lower_) | (X > self.upper_)).mean(axis=0)
