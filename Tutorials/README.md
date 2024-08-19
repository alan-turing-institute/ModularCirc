# Completing a Sensitivity Analysis using ModularCirc
Testing a simplified version of a circulation model
 ## Prerequisites
 * For running this code we need to install [ModularCirc](https://github.com/MaxBalmus/ModularCirc) 

 * Please run the following command: `pip install git+https://github.com/MaxBalmus/ModularCirc.git`

 * For using autoemulate on an M1 mac, we need to install lightGBM via conda: `conda install lightgbm`

 * Or install autoemulate using this link: https://github.com/alan-turing-institute/autoemulate.git@remove-lightgbm

## Steps to run
 1) First step is within the "creating_a_dataset" notebook, to create a CSV containing randomised parameters to use in the emulator training.
 2) Next step is to use the newly created CSV file to generate the corresponding target variables.