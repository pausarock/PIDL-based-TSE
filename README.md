# PIDL-JWZ for Traffic State Estimation on Freeway

This repository provides a modularized implementation of Physics-Informed Deep Learning (PIDL) for traffic state estimation on freeways

## Features

- **Data Management**: CSV-based configuration for topology and training parameters
- **Physics-Informed Neural Networks**: PINN model for traffic flow prediction
- **Modular Design**: Separated data loading, topology management, and model training
- **Automatic Testing**: Integrated MAPE and RMSE evaluation after training

## Project Structure

```

├── pidl_jwz_github/           # Main code directory
│   ├── __init__.py
│   ├── data_loader.py         # Data loading utilities
│   ├── topology_loader.py     # Topology and geometry management
│   ├── pinn.py               # PINN model implementation
│   ├── train.py              # Training script
│   ├── test.py               # Testing script
│   └── requirements.txt      # Dependencies
├── detector_positions.csv    # Detector locations (km)
├── ramp_info.csv             # Ramp locations and types
├── station_config.csv        # Station configuration (supervised/hidden)
├── train_config.csv          # Training parameters
└── data files/               # Traffic data
    ├── sh_flow_data.csv
    ├── sh_speed_data.csv
    └── ...
```

## Configuration Files

### Topology Configuration
- **`detector_positions.csv`**: Detector IDs and their longitudinal positions (km)
- **`ramp_info.csv`**: Ramp locations, types (+1 for on-ramp, -1 for off-ramp)
- **`station_config.csv`**: Station types (supervised/hidden) for training/testing

### Training Configuration
- **`train_config.csv`**: Default detector ID lists for training parameters

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
cd pidl_jwz_github
python train.py
```


### Testing
```bash
cd pidl_jwz_github
python test.py
```


### Command Line Options

```bash
python train.py --help
```

Key parameters:
- `--total_steps`: Number of training steps (default: 30000)
- `--layers`: Network architecture (default: "2,128,128,128,64,2")
- `--id_list`: Detector IDs for training (loaded from CSV)
- `--s_detect_id`: Supervised detector IDs (loaded from CSV)
- `--alpha1`, `--alpha2`, `--beta1`, `--beta2`, `--gamma`: Physics constraint weights

## Data Requirements

The system expects the following data files in the parent directory:
- `sh_flow_data.csv`: Traffic flow data
- `sh_speed_data.csv`: Traffic speed data

## Model Architecture


## Output Files

After training and testing:
- `best_model.pat`: Trained model weights
- `pred_flow_JWZ.csv`: Flow predictions
- `pred_speed_JWZ.csv`: Speed predictions
- Various visualization plots (PNG/SVG formats)

