"""
Neural-field (INR) surrogate of the LRN-OGV cascade and the tools to use it in an optimisation.

Modules (all need PyTorch, which aero_optim itself does not require; importing this package alone imports nothing heavy):
- paths: where the surrogate's data, trained models and optimisation runs live (environment overrides).
- foam: reading one-cell-thick OpenFOAM cases (cell centres, blade wall, fields).
- recut: periodic re-cut of the passage so that the blade lies on the domain boundary.
- morph: RBF morphing T_geom of the re-cut passage onto the reference rectangle.
- data: point-wise dataset, T_bal balancing, normalisation and coordinate embedding.
- siren: the networks (shift-modulated SIREN and MLP with Fourier features).
- predict: QoI predictor for new blades from their POD coefficients (mesh-free), and its differentiable variant.
- torch_geometry: the geometric constraints of aero_optim.optim.cascade_foam in torch (autodiff gradients).
Scripts (training, evaluation, optimisation, CFD verification, HPO) are in examples/LRN-OGV_OpenFOAM/surrogate/.
"""
