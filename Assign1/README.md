# DL-Assignment-1

## Running the code

1. From the project root, go into the `Assign1` folder:

```bash
cd Assign1
```

2. Create and activate a virtual environment, and install packages:

```bash
python -m venv .venv
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

3. You can now run any of the files using:

```bash
python example_file.py
```

## File structure

Below an overview of the content of the most important files:

- `tune_models.py` is used for selecting the best combinations of hyperparameters for both RNN and LSTM. It trains the models with different combinations of hyperparameters and stores the best hyperparameters into `results/best_(model).json`.
- `forecast_best_models.py` is used for forcasting and evaluating the models by comparing its predictions to the actual test values.
- `model.py` contains the $\texttt{pytorch}$ architecture of both the RNN and LSTM models.
