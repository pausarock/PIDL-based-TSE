# PIDL-JWZ for TSE on an expressway

Modularized from `PIDL_JWZ_withoutS.ipynb`. This repo provides:
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

Expected data files 
- `sh_flow_data.csv`
- `sh_speed_data.csv`

## Install
```bash
pip install -r requirements.txt
```

## Train
```bash
python -m pidl_jwz_github.train
```
This will save weights to `cv_v_modelq1.pat` and auto-run testing.

## Test
```bash
python -m pidl_jwz_github.test
```
It reports MAPE/RMSE and writes prediction CSVs.


