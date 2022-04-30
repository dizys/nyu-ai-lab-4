# nyu-ai-lab-4

About NYU Artificial Intelligence Course Lab 4: kNN and Naive Bayes implementations

## Prerequisite

-   Python 3.8+

## Getting-started

### Switch to Python 3.8 on CIMS machines

The Python version has to at least have full type hint support, thus requiring Python 3.8+.

```bash
module load python-3.8
```

If successful, the command `python3 --version` should give you:

```bash
$ python3 --version
Python 3.8.6
```

### Install dependencies/packages

Install Python packages used in this lab by running the following command:

```bash
pip3 install -r requirements.txt
```

or if you want to specify pandas runtime manually:

```bash
pip3 install pandas==1.4.2
```

### Script usages

> The main entrance is a python script, not a binary. It is in Shebang style,
> thus can be executed directly.
> Use `./learn -h` command to see the usage:

```
usage: learn [-h] --train TRAIN --test TEST [-k K] [-c C] [-v]

kNN algorithm and Naive Bayes implementations.

optional arguments:
  -h, --help     show this help message and exit
  --train TRAIN  the training csv data file
  --test TEST    the testing csv data file
  -k K           if > 0 indicates to use kNN and also the value of K (if 0, do Naive Bayes')
  -c C           if > 0 indicates the Laplacian correction to use (0 means don't use one)
  -v, --verbose  outputs each predicted vs actual label
```

Examples:

```bash
$ ./learn --train data/knn3.train.txt --test data/knn3.test.txt -k 7
```

```bash
$ ./learn --train data/ex2_train.csv --test data/ex2_test.csv -c 2
```

## Project structure

```
project
├─ml                            ml python module
│  ├─__init__.py                    Module initialization
│  ├─evaluation.py                  Evaluation-related calculations
│  ├─knn.py                         kNN algorithm implementation
│  └─naive_bayes.py                 Naive Bayes algorithm implementation
│
├─learn                         Main entrance python script (shebang style)
├─requirements.txt              Python packages used in this project
└─README.md                     The file you're reading
```
