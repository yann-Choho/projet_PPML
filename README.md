[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](http://opensource.org/licenses/MIT)

[![GitHub Issues](http://img.shields.io/github/issues/sdpython/teachcompute.svg)]([https://github.com/yann-Choho/projet_PPML](https://github.com/yann-Choho/projet_PPML/issues)/)

[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# ENSAE Paris | Institut Polytechnique de Paris

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/LOGO-ENSAE.png/900px-LOGO-ENSAE.png" width="300">

projet PPML
==============================

Optimize a machine learning model by fusing operations.


## Course : Parallel programming for Machine Learning
### Xavier Dupré and Matthieu Durut.
### Academic year: 2023-2024

### Realised by :

* Choho Yann Eric CHOHO
* Paul Guillermit 

If you want to run it manually on your computer, then follow the step bellow :

Clone this repository, then install the requirements library
```python
pip install -r requirements.txt
```

We mainly used the library Triton which requires Linux system : you can then Install WSL for Windows user 

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    │
    │
    ├── notebooks          <- Jupyter notebooks for the project. 
    │     ├── pytorch_kernel.py       <- kernel for Llama in pytorch 
    │     ├── triton_kernel.py        <- kernel for Llama in triton 
    │     ├── isolate_kernel.ipynb    <- notebook with explanation of our approach
    │     ├── utils.py                <- usefull python function for the kernel
    │     ├── figures                 <- figures produced after lauching the experiment (not alway reproductible)
    │     └── outputs                 <- other ouptut due to experiment 
    │
    ├── reports            <- our report explaining our approach, results and conclusion as PDF.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project. (just from cookiecutter :not realy used)
        └── __init__.py    <- Makes src a Python module


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
