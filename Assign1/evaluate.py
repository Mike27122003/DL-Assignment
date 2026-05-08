import numpy as  np
import scipy.io as sio
import matplotlib.pyplot as plt
import json
import os

def evaluate(rnn_future, lstm_future):
    os.makedirs("plots", exist_ok=True)
    # load test data
    mat = sio.loadmat("Xtest.mat")
    x_test = np.array(mat["Xtest"]).flatten()[-200:]

    rnn_future = rnn_future.flatten()
    
    lstm_future = lstm_future.flatten()

    # MAE and MSE
    rnn_mae = np.mean(np.abs(rnn_future - x_test))
    rnn_mse = np.mean((rnn_future - x_test) ** 2)

    lstm_mae = np.mean(np.abs(lstm_future - x_test))
    lstm_mse = np.mean((lstm_future - x_test) ** 2)

    print(f"RNN  — MAE: {rnn_mae:.6f} | MSE: {rnn_mse:.6f}")
    print(f"LSTM — MAE: {lstm_mae:.6f} | MSE: {lstm_mse:.6f}")

    # Save results
    results = {
        "RNN":  {"MAE": rnn_mae,  "MSE": rnn_mse},
        "LSTM": {"MAE": lstm_mae, "MSE": lstm_mse},
    }
    with open("results/test_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    time = np.arange(200)

    ax1.plot(time, x_test, label="Real", color="black")
    ax1.plot(time, rnn_future, label="RNN Predicted", color="red", linestyle="--")
    ax1.set_title(f"RNN — MAE: {rnn_mae:.6f} | MSE: {rnn_mse:.6f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(time, x_test, label="Real", color="black")
    ax2.plot(time, lstm_future, label="LSTM Predicted", color="blue", linestyle="--")
    ax2.set_title(f"LSTM — MAE: {lstm_mae:.6f} | MSE: {lstm_mse:.6f}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/test_evaluation.png", dpi=300, bbox_inches="tight")
    plt.show()

    return results

