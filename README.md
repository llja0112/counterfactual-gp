counterfactual-gp
===============================

version number: 0.0.1

author: Guangyi Zhang

Overview
--------

Counterfactual Gaussian Processes

Maximize the following MPP (marked point process) to build a counterfactual model.

* The event model, i.e., the point process, is omitted since it does not affect the outcome.
* The action model has separate parameters from the outcome model.
* The action model is time-invariant, i.e., playing a similar role as the mixture coefficients in mixture model of GPs.
* So far the action model is also binary and individual-invariant, which, however, can be easily extended.

![marked point](https://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20P%28y%2Cz_y%2Ca%2Cz_a%7Ct%29%20%26%3D%20P%28y%7Ct%2Cz_y%29%20P%28z_y%7Ct%29%20P%28a%2Cz_a%7Ct%29%5C%5C%20%5Ctext%7Bassume%20%7D%20%26%20P%28z_y%3D1%7Ct%29%3DP%28z_a%3D1%7Ct%29%3D1%2C%5C%5C%20%26z_a%2C%20z_y%20%5Cin%20%5C%7B0%2C1%5C%7D%2C%5C%5C%20%26a%20%5Cin%20%5C%7B0%2C1%2C...%2Ck%5C%7D%2C%5C%5C%20P%28y%2Cz_y%2Ca%2Cz_a%7Ct%29%20%26%3D%20P%28y%7Ct%29%20P%28a%7Ct%29%5C%5C%20max%20%5C%3B%20%5Csum_j%5E%7Bn_i%7D%20logP%28y_j%2Cz_%7Byj%7D%2Ca_j%2Cz_%7Baj%7D%7Ct_j%29%20%26%5CLeftrightarrow%20max%20%5C%3B%20%5Csum_j%5E%7Bn_i%7D%20log%20P%28y_j%7Ct_j%29%20&plus;%20%5C%3B%20%5Csum_j%5E%7Bn_i%7D%20log%20P%28a_j%7Ct_j%29%5C%5C%20%5Ctext%7Bwhere%20%7D%20%26%20P%28y%7Ct%29%20%3D%20%5Csum_%7Ba%7D%20P%28y%2Ca%7Ct%29%20%3D%20%5Csum_%7Ba%7D%20P%28y%7Ct%2Ca_%29P%28a%7Ct%29%20%5Cend%7Baligned%7D)

Reference: 

* Schulam, Peter, and Suchi Saria. "Reliable decision support using counterfactual models." Advances in Neural Information Processing Systems. 2017.

Usage
--------------------

TBD

Dev
------------

Build simulated dataset from Jupyter Notebooks.

To run tests:

    $ py.test


Contributing
------------

TBD

Example
-------

TBD
