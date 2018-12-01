# PDE Learning Example 

This is a snapshop of the deep-flow-prediction code from
(https://github.com/thunil/Deep-Flow-Prediction)

All scripts below assume they're executed from their respective directories.

## Data download


Below you can download a large-scale training data set, and a test data set.
Unpack them into `../data/train` and `../data/test`.

* Reduced data set with 6.4k samples plus test data (2GB): <http://ge.in.tum.de/download/data_6k.tar.gz>
* Full data set with 53.8k samples plus test data (10GB): <http://138.246.224.34/index.php/s/m1459172> 

## Convolutional neural network training

Switch to the directory containing the training scripts, i.e., `dl4g/pde_learning/`,
and execute `python ./runTrain.py`. By default, this will execute a short training  run
with 10k iterations, loading all data that is available in `../data/train`. The L1 
validation loss is printed during training, and should decrease significantly.
Once the script has finished, it will save the trained model as `modelG`.

A sample image will be generated for each epoch in the `results_train` directory.
Optionally, you can also save txt files with the loss progression (see `saveL1` in the script).
explain created files:

## Test evaluation

To compute relative inference errors for a test data set, you can use the `./runTest.py` script.
By default, it assumes that the test data samples (with the same file format as the training samples)
are located in `../data/test`. Hence, you either have to generate data in a new directory with the
`dataGen.py` script from above, or download the test data set via the link below.

Once the test data is in place, execute `python ./runTest.py`. This script can compute accuracy 
evaluations for a range of models, it will automatically evaluate the test samples for all existing model files
named `modelG`,
`modelGa`,
`modelGb`,
`modelGc`, etc.

The text output will also be written to a file `testout.txt`. In addition, visualized reference data
and corresponding inferred outputs are written to `results_test` as PNGs.

## Further steps

For further experiments, you can increase the `expo` parameter in `runTrain.py` and `runTest.py` (note, non-integers are allowed). For large models you'll need much more data, though, to avoid overfitting.

In addition, the `DfpNet.py` file is worth a look: it contains most of the non-standard code for the RANS flow prediction. E.g., here you can find the U-net setup and data normalization. Hence, this class is a good starting point for experimenting with different architectures.

Note that both the `runTrain.py` and  `runTest.py` scripts also accept a prefix as command line argument. 
This can come in handy for automated runs with varying parameters.

