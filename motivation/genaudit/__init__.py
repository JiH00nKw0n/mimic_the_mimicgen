"""genaudit — generation-audit framework for success-filtering bias experiments.

Core modules are importable without any simulator. Simulator-facing modules
(mimicgen, robomimic, h5py) guard their imports and fail loudly with install
instructions when the dependency is missing.
"""

__version__ = "0.1.0"
