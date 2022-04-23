# ASTR-6410
`lya_photons.py`
The package I wrote for reading simulation data and calculating surface brightness profiles. 



`ModelFitting_1D.py`; `ModelFitting_2D.py`

These are the files for Gaussian Process Regression and plots. The simulation data is on the CHPC server so these codes can only be run on the CHPC server.

"1D": predictions with one self variable (concentration).
"2D": predictions with two self variables. (concentration + halo mass).

They have the same structure and the same comments. Their differences are mainly on:
 	1. Different data structures when reading simulation data in.
 	2. Different plotting setup.
