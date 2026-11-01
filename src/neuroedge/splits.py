"""Explicit participant-disjoint outer and validation folds."""
import numpy as np
from sklearn.model_selection import KFold


def assert_disjoint(train, validation, test):
    a, b, c = map(set, (train, validation, test))
    if not a or not b or not c:
        raise ValueError("Each split needs participants.")
    if a & b or a & c or b & c:
        raise ValueError("Participant leakage between train, validation and test.")


def make_splits(subjects, n_folds, split_seed, validation_fraction):
    subjects = np.asarray(sorted(set(map(int, subjects))))
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=split_seed)
    splits = []
    for fold, (source_idx, test_idx) in enumerate(splitter.split(subjects)):
        source = subjects[source_idx]
        rng = np.random.default_rng(split_seed + fold)
        shuffled = rng.permutation(source)
        n_val = max(1, int(np.ceil(len(source) * validation_fraction)))
        if n_val >= len(source):
            raise ValueError("Validation split leaves no training subjects.")
        train = sorted(shuffled[n_val:].tolist())
        validation = sorted(shuffled[:n_val].tolist())
        test = subjects[test_idx].tolist()
        assert_disjoint(train, validation, test)
        splits.append({"fold": fold, "train": train, "validation": validation, "test": test})
    all_test = [s for split in splits for s in split["test"]]
    if sorted(all_test) != subjects.tolist():
        raise ValueError("Outer-test coverage is not exactly once per participant.")
    return splits

