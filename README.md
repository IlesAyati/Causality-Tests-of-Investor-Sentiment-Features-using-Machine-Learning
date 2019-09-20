# MasterThesis
Preliminary thesis.

# Instructions
MasterThesisScript.py contains the entire script in one .py file.
ExecutionFile.py allows for execution of the script by module. There are 9 executable modules.
Order is important: All modules must be executed in the preset order.
All regressions depend on data preperation modules DataCollection.py,
and all machine learning modules depend on LOGIT.py and MLPrep.py as well. 

# Modules
1. DataCollection.py : Loads data and defines variables. 
2. GLSreg.py : Runs GLS regressions, univariate and multivariate.
3. VAR.py : Runs VAR model and tests for Granger causality
4. LOGIT.py : To be deprecated: For now, just run it to prepare data for all ML_* .py files
5. MLPrep.py : Defines regression data for machine learning modules.
6. ML_OLS.py : SKlearn OLS regressions with and without principal components - One-way causality tests.
7. ML_RIDGELASSO.py : SKlearn Ridge and Lasso regressions with and without principal components - One-way causality tests.
8. ML_FOREST.py : SKlearn Random Forest Regressions with and without principal components - One-way causality tests.
9. TablesFigures.py : Tables and figures ordered by executable module. 
