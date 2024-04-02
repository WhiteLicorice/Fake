@ECHO OFF

ECHO Running on Cruz_2020 dataset
START "Cruz Crossval" /WAIT python crossval.py Cruz
ECHO Running on Lupac_2024 dataset
START "Lupac Crossval" /WAIT python crossval.py Lupac
ECHO Running on combined dataset
START "Combined Crossval" /WAIT python crossval_combined_ds.py
PAUSE