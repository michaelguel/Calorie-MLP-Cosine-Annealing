# Calorie Burn Prediction using MLPRegressor

This project uses a Multi-Layer Perceptron (MLP) model built in PyTorch to predict calorie expenditure based on biometric and activity data. It includes feature engineering, model training with RMSLE loss, and a test prediction pipeline.

## Project Structure

- `train.csv`, `calories_original.csv`: Training datasets
- `test.csv`: Test dataset
- `pyversion.py`: Main training and inference script
- `mlp_test_predictions.csv`: Final test predictions output
- `best_mlp_model.pt`, `mlp_model_final.pt`: Saved model weights

## Features

- Advanced **feature engineering** that incorporates BMI, BMR, lean mass, and interaction terms between age, sex, duration, and weight.
- Custom **RMSLE loss function** for evaluating model predictions in log space, mitigating the influence of large errors.
- **Cosine Annealing with Warm Restarts** for efficient learning rate scheduling.
- Train/validation split with standardization using `StandardScaler`.
- Model checkpointing to save the best-performing model during training.
- Final predictions clipped to avoid negative values.

## Model Architecture

The MLPRegressor consists of:
- Input layer
- 3 hidden layers with sizes [256, 128, 64]
- Batch normalization and ReLU activation in each hidden layer
- Final output layer with a single neuron for regression

## Training Details

- Optimizer: Adam
- Initial LR: 8e-3
- Batch size: 256
- Epochs: 100
- Scheduler: `CosineAnnealingWarmRestarts` with `T_0=20`, `T_mult=2`, `eta_min=1e-5`

## How to Run

Ensure you have the required CSVs in the working directory:

```bash
python pyversion.py
```

This will:
- Load and process data
- Train the MLP model
- Save best and final model weights
- Generate predictions for the test set as `mlp_test_predictions.csv`

## Requirements

```bash
pip install pandas numpy torch scikit-learn tqdm
```

## Output

After running, you'll get:
- Training/validation RMSLE logs per epoch
- Final CSV with predicted calorie values

## Notes

- Assumes categorical encoding for `Sex` and `AgeGroup` using `.cat.codes`
- Uses Torch’s DataLoader with `persistent_workers=True` for efficiency
- Safe clamping applied to predictions to avoid invalid log operations