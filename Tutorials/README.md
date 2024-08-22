# Completing a Sensitivity Analysis using ModularCirc
This tutorial explains how to complete a sensitivity analysis to identiy a subset of parameters which can be put into the model - that if changed represent change in the model and that can be matched to patient data.

## Why this is useful?
Involved in a larger project involving patients with pulmonary arterial hypertension that have implanted cardiac monitors and/or pulmonary artery pressure monitors – which give values for pulmonary arterial pressure and cardiac output.

Currently, while the monitors provide a lot of cardiac data, the usefulness of the data that they provide is limited, as it is difficult to interpret and use the information to make approprate changes to patient management.

The project focuses on the development of a digital twin to aid with interpretation of the cardiac data from these monitors. 

The sensitivity analysis aims to focus in on the parameters from the model that most affect pulmonary arterial pressure and cardiac output.

Using a reduced set of parameters, you can more efficiently fit the model to patient data, improving the performance of the fitting process.


 ## Prerequisites
 * For running this code we need to install [ModularCirc](https://github.com/MaxBalmus/ModularCirc) 

 * Please run the following command: `pip install git+https://github.com/MaxBalmus/ModularCirc.git`

 * For using autoemulate on an M1 mac, we need to install lightGBM via conda: `conda install lightgbm`

 * Or install autoemulate using this link: https://github.com/alan-turing-institute/autoemulate.git@remove-lightgbm

## Steps to run

![alt text](image-1.png)

 1) First step is within the "step1" notebook, to create a CSV containing randomised parameters to use in the emulator training.
 2) Next step is to run these input parameters through ModularCirc simulations to create pressure pulse dataset (corresponding variables).
 3) Use the pressure pulse dataset and completing PCA - to reduce the dataset.
 4) Complete K fold cross validation.
 5) Retrain model on all the data with the new reduced number of components.
 6) Use the reduced PCA results as output and the original parameter set you created as input for emulation - use Autoemulate to find the best emulator for the data.
 7) Conduct a sensitivity analysis using the results from emulation using SAlib.
