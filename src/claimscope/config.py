"""Explicit feature allowlist is the main defense against accidental target leakage."""

NUMERIC = ["VehPower", "VehAge", "DrivAge", "Density"]
CATEGORICAL = ["Area", "VehBrand", "VehGas", "Region"]
FEATURES = NUMERIC + CATEGORICAL
SEED = 42
SOURCE_REVISION = "227fb56b8734bdb7c0327a41180e01d2ddaeaf26"
SOURCE_BASE = f"https://raw.githubusercontent.com/dutangc/CASdatasets/{SOURCE_REVISION}"
DATASETS = ["freMTPL2freq", "freMTPL2sev"]
