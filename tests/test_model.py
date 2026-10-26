from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from generate_data import generate
from train import FEATURES, TARGET, build_pipeline


def test_dataset_schema():
    df = generate(n=600)
    assert set(FEATURES + [TARGET]).issubset(df.columns)
    assert df[TARGET].isin([0, 1]).all()


def test_model_beats_baseline():
    df = generate(n=1500)
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=0)
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    baseline = max(y_te.mean(), 1 - y_te.mean())
    assert pipe.score(X_te, y_te) > baseline
