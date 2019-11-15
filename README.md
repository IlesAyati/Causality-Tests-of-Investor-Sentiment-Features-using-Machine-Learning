# MasterThesis
Preliminary thesis. 

# Instructions
MasterThesisScript.py contains the entire script in one .py file.
ExecutionFile.py allows for execution of the script by module. There are 8 executable scripts. 
All scripts depend on data preperation module DataCollection.py,
and all machine learning modules additionally depend on MLPrep.py. 

# Modules
1. DataCollection.py : Loads data and defines variables. 
2. GLSreg.py : Runs GLS regressions, univariate and multivariate.
3. VAR.py : Runs VAR model and tests for Granger causality
4. MLPrep.py : Defines regression data for machine learning modules.
5. ML_OLS.py : SKlearn OLS regressions with and without principal components - One-way causality tests.
6. ML_RIDGELASSO.py : SKlearn Ridge and Lasso regressions with and without principal components - One-way causality tests.
7. ML_FOREST.py : SKlearn Random Forest Regressions with and without principal components - One-way causality tests.
8. TablesFigures.py : Tables and figures ordered by executable module. 
