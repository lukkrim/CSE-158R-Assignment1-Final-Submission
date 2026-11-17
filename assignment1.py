"""CSE 158R Assignment 1 - Goodreads recommendation (Tasks 1-6).

Each task validates its model locally, refits on all available data, and writes
my_prediction/task_<n>_predictions.csv. The models live in the `models/` package:

    models/rating_base.py      Surprise SVD helpers, history statistics, residual neighborhoods
    models/rating_stack.py     shared rating stack for Tasks 1 and 3 (kNN, ALS, SVD++,
                               Bayesian FM, timeSVD++, gradient-boosted stacker)
    models/temporal_rating.py  Task 3 timestamp features
    models/pairwise.py         pairwise interaction classifier for Tasks 2, 4 and 5
    models/sequential.py       Task 5 sequence (transition) features
    models/task6_*.py          Task 6 time-until-next-interaction model (v3 builds on v2/advanced)

Usage:
    python assignment1.py                      # all tasks, data in ./data
    python assignment1.py --tasks 1 3          # selected tasks only
    python assignment1.py --data-dir /path/to/csvs --output-dir my_prediction
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from models.pairwise import PairwiseFeatures, build_training_pairs, fit_pair_classifier, pair_probability, sample_pairs
from models.rating_stack import predict_rating_task, validate_rating_task
from models.sequential import SequentialFeatures
from models import task6_v3

ROOT = Path(__file__).resolve().parent
STRINGS = {'user': 'string', 'item': 'string', 'item1': 'string', 'item2': 'string'}


def read(data_dir, name):
    return pd.read_csv(Path(data_dir) / name, dtype=STRINGS)


def held_out_user_mask(users, seed=0):
    """Mark a random 25% of users for evaluation of a pairwise classifier."""
    unique = users.unique(); np.random.default_rng(seed).shuffle(unique)
    return users.isin(set(unique[:len(unique)//4])).to_numpy()


def write(frame, columns, out_dir, task):
    path = Path(out_dir) / f'task_{task}_predictions.csv'
    frame[columns].to_csv(path, index=False)
    print('Wrote', path, flush=True)


# ---------------------------------------------------------------------------
# Task 1 - Rating prediction
# The test sets contain no 0-star ratings (0 = shelved, not rated), so every rating
# model is trained and validated on rating > 0 and predictions are clipped to [1, 5].
# ---------------------------------------------------------------------------
def task1(data_dir, out_dir):
    ratings = read(data_dir, 'tasks_1_2_train.csv')
    print('Rating distribution:', ratings.rating.value_counts(normalize=True).sort_index().round(4).to_dict())
    stacker, columns, report = validate_rating_task(ratings, temporal=False)
    print(report.to_string())
    test = read(data_dir, 'task_1_test.csv')
    test['rating'] = predict_rating_task(ratings, test, stacker, columns, temporal=False)
    assert np.isfinite(test.rating).all()
    write(test, ['user', 'item', 'rating'], out_dir, 1)


# ---------------------------------------------------------------------------
# Task 2 - Interaction prediction (which of two books did the user read?)
# Training pairs: 5 rounds x 30k held-out interactions, base models refit each round;
# negatives are the item of a random interaction, matching the test construction.
# ---------------------------------------------------------------------------
def pairwise_task(data, temporal, label):
    pairs, a, b, ctx = build_training_pairs(data, temporal=temporal)
    y = pairs.label.to_numpy()
    ev = held_out_user_mask(pairs.user)
    check = fit_pair_classifier(a[~ev], b[~ev], ctx[~ev], y[~ev])
    print(f'{label} held-out-user accuracy:', np.mean((pair_probability(check, a[ev], b[ev], ctx[ev]) > .5) == y[ev]), flush=True)
    return fit_pair_classifier(a, b, ctx, y)


def task2(data_dir, out_dir):
    data = read(data_dir, 'tasks_1_2_train.csv')
    model = pairwise_task(data, temporal=False, label='Task 2')
    test = read(data_dir, 'task_2_test.csv')
    p = pair_probability(model, *PairwiseFeatures(temporal=False).fit(data).pair_features(test))  # includes the seen-item rule
    test['which'] = np.where(p > .5, test.item1, test.item2)
    write(test, ['user', 'item1', 'item2', 'which'], out_dir, 2)


# ---------------------------------------------------------------------------
# Task 3 - Rating prediction with timestamp
# Same stack as Task 1 plus timestamp features and timeSVD++ (temporal=True).
# ---------------------------------------------------------------------------
def task3(data_dir, out_dir):
    data = read(data_dir, 'tasks_3_4_train.csv')
    stacker, columns, report = validate_rating_task(data, temporal=True)
    print(report.to_string())
    test = read(data_dir, 'task_3_test.csv')
    test['rating'] = predict_rating_task(data, test, stacker, columns, temporal=True)
    assert np.isfinite(test.rating).all()
    write(test, ['user', 'item', 'timestamp', 'rating'], out_dir, 3)


# ---------------------------------------------------------------------------
# Task 4 - Timestamped interaction prediction
# Task 2 pipeline plus features at the query time t.
# ---------------------------------------------------------------------------
def task4(data_dir, out_dir):
    data = read(data_dir, 'tasks_3_4_train.csv')
    model = pairwise_task(data, temporal=True, label='Task 4')
    test = read(data_dir, 'task_4_test.csv')
    p = pair_probability(model, *PairwiseFeatures(temporal=True).fit(data).pair_features(test))
    test['which'] = np.where(p > .5, test.item1, test.item2)
    write(test, ['user', 'timestamp', 'item1', 'item2', 'which'], out_dir, 4)


# ---------------------------------------------------------------------------
# Task 5 - Sequential next-item prediction
# Validation holds out the final book of 60k users. Training timestamps are allowed
# (instructor Q&A), so the query time is each user's last training timestamp.
# ---------------------------------------------------------------------------
def task5(data_dir, out_dir):
    data = read(data_dir, 'task_5_train.csv')
    rng = np.random.default_rng(42)
    users = rng.choice(data.user.unique(), 60000, replace=False)
    last = data.groupby('user', sort=False).tail(1)
    hold = last[last.user.isin(set(users))]
    train = data.drop(index=hold.index).reset_index(drop=True)
    features = SequentialFeatures(temporal=True).fit(train)
    pairs = sample_pairs(hold.drop(columns='timestamp'), train.item.to_numpy(), 5)
    pairs['timestamp'] = pairs.user.map(train.groupby('user').timestamp.max()).to_numpy()
    a, b, ctx = features.pair_features(pairs)
    y = pairs.label.to_numpy()
    ev = np.zeros(len(y), bool); ev[rng.permutation(len(y))[:len(y)//4]] = True
    check = fit_pair_classifier(a[~ev], b[~ev], ctx[~ev], y[~ev])
    print('Task 5 held-out accuracy:', np.mean((pair_probability(check, a[ev], b[ev], ctx[ev]) > .5) == y[ev]), flush=True)
    model = fit_pair_classifier(a, b, ctx, y)

    test = read(data_dir, 'task_5_test.csv')
    test['timestamp'] = test.user.map(data.groupby('user').timestamp.max()).to_numpy()
    p = pair_probability(model, *SequentialFeatures(temporal=True).fit(data).pair_features(test))
    test['which'] = np.where(p > .5, test.item1, test.item2)
    write(test, ['user', 'item1', 'item2', 'which'], out_dir, 5)


# ---------------------------------------------------------------------------
# Task 6 - Time until the next (final) interaction
# The metric behaves like MAE, so absolute-error boosters predict conditional medians
# (see models/task6_v3.py).
# ---------------------------------------------------------------------------
def task6(data_dir, out_dir):
    state = task6_v3.prepare(data_dir)
    task6_v3.evaluate(state)
    task6_v3.predict(state, out_dir)


TASKS = {1: task1, 2: task2, 3: task3, 4: task4, 5: task5, 6: task6}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data', help='folder with the assignment CSV files')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'my_prediction')
    parser.add_argument('--tasks', type=int, nargs='+', default=list(TASKS), choices=list(TASKS))
    args = parser.parse_args()
    if not (args.data_dir / 'tasks_1_2_train.csv').is_file():
        parser.error(f'assignment CSV files not found in {args.data_dir} (use --data-dir)')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        print(f'===== Task {task} =====', flush=True)
        TASKS[task](args.data_dir, args.output_dir)


if __name__ == '__main__':
    main()
