# Visual
**A Convolutional Neural Network based model to predict the peptide binding sites in proteins.**\
Please cite the associated publication when using these codes.\
**Citation:** W. Wardah, A. Dehzangi and G. Taherzadeh et al., Predicting protein-peptide binding sites with a deep convolutional neural network, Journal of Theoretical Biology, https://doi.org/10.1016/j.jtbi.2020.110278 \
**Environment:** The original experiment was run on GeForce GTX 1060 Ti graphics card.

## Data Files
The data files contain sample preprocessed protein data (details found in the JTB paper).
- train_7_set.csv and train_labels.txt
- val_7_set.csv and val_labels.txt
- test_7_set.csv and test_labels.txt
## Code Files
- experiment.py (This is the main file. Run this to start running the experiment.)
- models.py (This contains the model structure and details.)
- dataset.py (This file loads datasets.)
## Guide
Put all these files in one folder and run the experiment.py file.
The experiment will continue running iterations until you manually stop/kill it.
