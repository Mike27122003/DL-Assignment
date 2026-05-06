import matplotlib.pyplot as plt
import json
import os
import numpy as np

def plot_history():
    os.makedirs("plots", exist_ok=True)
    rnn_hist = json.load(open("results/best_rnn_history.json"))
    lstm_hist = json.load(open("results/best_lstm_history.json"))

    plt.figure(figsize=(12, 8))

    # --- RNN ---
    plt.subplot(2, 2, 1)
    plt.plot(rnn_hist["train_mse"], label="Train MSE")
    plt.plot(rnn_hist["val_mse"], label="Val MSE")
    plt.title("RNN MSE (Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(rnn_hist["val_mae"], label="Val MAE", color="orange")
    plt.title("RNN Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()

    # --- LSTM ---
    plt.subplot(2, 2, 3)
    plt.plot(lstm_hist["train_mse"], label="Train MSE")
    plt.plot(lstm_hist["val_mse"], label="Val MSE")
    plt.title("LSTM MSE (Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(lstm_hist["val_mae"], label="Val MAE", color="orange")
    plt.title("LSTM Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()

    plt.tight_layout()

    plt.savefig("plots/epoch_history.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_forecasts(actual, rnn_in, rnn_future, lstm_in, lstm_future, window_size_rnn, window_size_lstm, predict_future):
    os.makedirs("plots", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    time_actual = np.arange(len(actual))
    time_future = np.arange(len(actual), len(actual) + predict_future)

    # -------------RNN--------------

    #Plot actual data
    ax1.plot(time_actual, actual, label="Actual Data", color="black", alpha=0.5)

    time_in_sample_rnn = np.arange(window_size_rnn, window_size_rnn + len(rnn_in))

    ax1.plot(
        time_in_sample_rnn,
        rnn_in,
        label="Dataset predictions",
        color="blue",
        linestyle="--",
        alpha=0.8,
    )

    # Plot Future predictions
    ax1.plot(
        time_future,
        rnn_future,
        label=f"Future predictions ({predict_future} steps)",
        color="red",
        linewidth=2,
    )

    # Vertical line to show where the real data ends
    ax1.axvline(
        x=len(actual), color="green", linestyle=":", label="Forecast Start"
    )

    ax1.set_title("Plotted data RNN")
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Value")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # -------------LSTM-------------

    #Plot actual data
    ax2.plot(time_actual, actual, label="Actual Data", color="black", alpha=0.5)

    time_in_sample_lstm = np.arange(window_size_lstm, window_size_lstm + len(lstm_in))

    ax2.plot(
        time_in_sample_lstm,
        lstm_in,
        label="Dataset predictions",
        color="blue",
        linestyle="--",
        alpha=0.8,
    )

    # Plot Future predictions
    ax2.plot(
        time_future,
        lstm_future,
        label=f"Future predictions ({predict_future} steps)",
        color="red",
        linewidth=2,
    )

    # Vertical line to show where the real data ends
    ax2.axvline(
        x=len(actual), color="green", linestyle=":", label="Forecast Start"
    )

    ax2.set_title("Plotted data LSTM")
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Value")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/data_forecast.png", dpi=300, bbox_inches="tight")
    plt.show()