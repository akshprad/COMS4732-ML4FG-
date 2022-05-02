# COSSMO
Competitive Splice Site Model

This is the code used to train the models described in the paper

> Hannes Bretschneider, Shreshth Gandhi, Khalid Zuberi, Amit G Deshwar, and Brendan J Frey  
> COSSMO: Predicting Competitive Alternative Splice Site Selection using Deep Learning  
> Bioinformatics, Volume 34, Issue 13, 1 July 2018, Pages i429–i437,  
> https://doi.org/10.1093/bioinformatics/bty244

## Installation

### Dependencies
COSSMO requires Python 2.7 and TensorFlow 1.8. We provide a tested Conda
environment. 

### Anaconda Python
We recommend using COSSMO with the conda package manager from 
[Anaconda Python](https://www.anaconda.com/distribution/).

To create a Conda environment for running COSSMO, install either the
`environment.yml` or `environment_gpu.yml` (includes TensorFlow with GPU 
support) file:

    conda env create -f environment.yml
or 

    conda env create -f environment_gpu.yml  

Activate your environment with

    conda activate cossmo
    
### Pip
Alternatively you can install the dependencies via the `requirements.txt` file
for pip:

    pip install -f requirements.txt
    
### Install COSSMO
After creating the dependencies you can install the COSSMO package. The 
following command will symlink the COSSMO package into your environment. 

    pip install -e .
    
    
## Tests
To run the tests, first install a test runner in your environment. We recommend
py.test:

    conda install pytest
    
Running py.test from the root directory will discover all the test 
automatically:

    pytest
    
## Training

### Training script
The training script is at 
[`bin/train_cossmo.py`](https://github.com/PSI-Lab/COSSMO/blob/master/bin/train_cossmo.py).
You can call the script with the `--help` option to receive details of the
parameters:

    $ python bin/train_cossmo.py --help
    usage: train_cossmo.py [-h] [--configuration-file CONFIGURATION_FILE]
                           [--gpu GPU] [--intra-op-threads INTRA_OP_THREADS]
                           [--inter-op-threads INTER_OP_THREADS] [--test-only]
                           [--fold FOLD]
    
    optional arguments:
      -h, --help            show this help message and exit
      --configuration-file CONFIGURATION_FILE
                            Path to a configuration file, containing all
                            hyperparameters, model definitions, etc. See the
                            provided examples for details.
      --gpu GPU             GPU device ID to use for training. This is equivalent
                            to setting the CUDA_VISIBLE_DEVICES environment
                            variable. It is recommend to set this option when you
                            have more than one GPU device in your system to
                            prevent TensorFlow from claiming all devices.
      --intra-op-threads INTRA_OP_THREADS
                            See https://github.com/tensorflow/tensorflow/blob/26b4
                            dfa65d360f2793ad75083c797d57f8661b93/tensorflow/core/p
                            rotobuf/config.proto#L165 for the meaning of this
                            parameter.
      --inter-op-threads INTER_OP_THREADS
                            See https://github.com/tensorflow/tensorflow/blob/26b4
                            dfa65d360f2793ad75083c797d57f8661b93/tensorflow/core/p
                            rotobuf/config.proto#L165 for the meaning of this
                            parameter.
      --test-only           Don't train, only evaluate test set.
      --fold FOLD           Cross-validation fold to train on. When set, this
                            overrides the`cv_fold` key in the configuration file.

### Configuration files
Configuration files to replicate the models described in the paper are available
in [`configuration_files/`](https://github.com/PSI-Lab/COSSMO/blob/master/configuration_files).
Before you can use these configuration files, you must edit them and provide the
correct file paths for the dataset path and output path for your system.

### COSSMO with BERT (https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel)

To run the new COSSMO model.py from Colab notebook, we will need to install Tensorflow 1.13 for the tensorflow.contrib to work.

 pip install tensorflow==1.13.2

The next step would be to load the configuration file and store it in a variable config1, that will be passed as a parameter while calling the model to run.

The model can be called with 

c.main(configuration = config1, continue_from=None, intra_op_threads=2, inter_op_threads=6, test_only=False)

An example ipynb file is attached to this repository with the code for these steps.
