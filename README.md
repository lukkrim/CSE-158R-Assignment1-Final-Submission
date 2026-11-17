# CSE 158R Assignment 1 - Goodreads recommendation

Code, write-up and predictions for all six tasks.

```
assignment1.py        main script: validates, refits and writes predictions for each task
writeup.txt           description of the approach for each task
requirements.txt      package versions used
models/               helper modules
  rating_base.py        Surprise SVD helpers, history statistics, residual neighborhoods
  rating_stack.py       shared rating stack for Tasks 1 and 3
  temporal_rating.py    Task 3 timestamp features
  pairwise.py           pairwise interaction classifier (Tasks 2, 4, 5)
  sequential.py         Task 5 sequence features
  task6_advanced.py     Task 6 gap-history features
  task6_v2.py           Task 6 median model (builds on task6_advanced)
  task6_v3.py           Task 6 final model (builds on task6_v2)
my_prediction/        submitted task_1..6_predictions.csv
```

## Reproducing

1. Install the packages (Python 3.12): `pip install -r requirements.txt`
2. Put the assignment CSV files (`tasks_1_2_train.csv`, `tasks_3_4_train.csv`,
   `task_5_train.csv`, `task_6_train.csv`, `task_1_test.csv` ... `task_6_test.csv`)
   in a `data/` folder next to `assignment1.py`, or pass `--data-dir`.
3. Run all tasks, or only some:

```
python assignment1.py
python assignment1.py --tasks 1 3 --data-dir /path/to/csvs
```

Predictions are written to `my_prediction/` (change with `--output-dir`).
Running every task takes several hours on a laptop; Tasks 1 and 3 are the slowest.
Random seeds are fixed, but multithreaded libraries (implicit, myfm) can introduce
small run-to-run differences.
