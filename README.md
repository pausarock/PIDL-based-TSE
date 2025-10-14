# PIDL-JWZ for TSE on an freeway

This repo provides:
- Data loader for flow/speed (density excluded)
- Topology utilities (detectors, ramps, region function)
- PINN model
- Training and testing scripts

## Structure
```
pidl_jwz_github/
  __init__.py
  data_loader.py
  topology_loader.py
  pinn.py
  train.py
  test.py
```

Expected data files :
- `sh_flow_data.csv`
- `sh_speed_data.csv`


## Install
```bash
pip install -r requirements.txt
```

## Train
```bash
python train.py
```
This will save weights to `best_model.pat` and auto-run testing.

## Test
```bash
python test.py
```
It reports MAPE/RMSE and writes prediction CSVs.


