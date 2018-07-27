counterfactual-gp
===============================

version number: 0.0.1

author: Guangyi Zhang

Overview
--------

Counterfactual Gaussian Processes

Maximize the following MPP (marked point process) to build a counterfactual model.
The event model, i.e., the point process is omitted since it does not affect the outcome.

![marked point](https://latex.codecogs.com/svg.latex?%5Cbegin%7Baligned%7D%20P%28y%2Cz_y%2Ca%2Cz_a%7Ct%29%20%26%3D%20P%28y%7Ct%2Cz_y%29%20P%28z_y%7Ct%29%20P%28a%2Cz_a%7Ct%29%5C%5C%20%26assume%20%5C%3B%20P%28z_y%3D1%7Ct%29%3D1%2C%5C%3B%20a%20%5Cin%20%5C%7B0%2C1%5C%7D%5C%5C%20P%28y%2Cz_y%2Ca%2Cz_a%7Ct%29%20%26%3D%20P%28y%7Ct%2Cz_y%29%20P%28z_a%7Ct%29%5C%5C%20%26assume%20%5C%3B%20P%28y%7Ct%2Cz_y%3B%20%5Ctheta_1%29%2C%20P%28z_a%7Ct%3B%20%5Ctheta_2%29%2C%20%5Ctheta_1%5Ccap%5Ctheta_2%3D%5Cemptyset%5C%5C%20max%20%5C%3B%20%5Csum_j%5E%7Bn_i%7D%20logP%28y_j%2Cz_%7Byj%7D%2Ca_j%2Cz_%7Baj%7D%7Ct_j%29%20%26%5CLeftrightarrow%20max%20%5C%3B%20%5Csum_j%5E%7Bn_i%7D%20log%20P%28y_j%7Ct_j%2Cz_%7Byj%7D%29%20&plus;%20%5C%3B%20%5Csum_j%5E%7Bn_i%7D%20log%20P%28z_%7Baj%7D%7Ct_j%29%5C%5C%20%26assume%20%5C%3B%20P%28y%7Ct%2Cz_y%29%20%3D%20%5Csum_%7Bz_a%7D%20P%28y%2Cz_a%7Ct%2Cz_y%29%20%3D%20%5Csum_%7Bz_a%7D%20P%28y%7Ct%2Cz_y%2Cz_a%29P%28z_a%7Ct%29%20%5Cend%7Baligned%7D)

Reference: 

* Schulam, Peter, and Suchi Saria. "Reliable decision support using counterfactual models." Advances in Neural Information Processing Systems. 2017.

Usage
--------------------

To install use pip:

    $ TBD

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
