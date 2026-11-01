"""Small, explicit text preprocessing variants for controlled experiments."""

from sklearn.base import BaseEstimator, TransformerMixin


class WhitespaceNormalizer(TransformerMixin, BaseEstimator):
    """Collapse internal whitespace and trim its edges, without letter mapping."""

    def fit(self, texts, y=None):
        return self

    def transform(self, texts):
        return [" ".join(text.split()) for text in texts]
