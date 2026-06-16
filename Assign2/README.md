# DL-Assignment-2

## Running the code

Code makes use of a `--type` parameter, which can have one of the following values (case-sensitive):

- `Intra`
- `Cross`

From the Assign2 project root the code can be executed by running

```sh
$ uv run src/main.py --type Intra # or Cross
```

## File structure

Below an overview of the content of the most important files located in the `src` folder:

- `main.py` is used for training and running the model with different hyperparameter combinations.
- `preprocess.py` contains functions that are used for the data preprocessing.
- `CNN.py` and `cnn_methods` contain the $\texttt{pytorch}$ architecture of the CNN model and additional functions used for training the model.
- `utils/windowed_dataset.py` contains the $\texttt{WindowedDataset}$ class that is used for creating different windows for the model.
