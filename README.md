# rld_diffusion

To run the code, you must do two things:

1) Change the flags on lines 10 and 11 of src/basis_funcs.py to determine where your data and output will be stored
2) Execute: python3 main.py ; there are several arguments you can add, including:
  -quickie to use only 100 pieces of training data
  -w to set w
  -reset to delete all the output you've previously had (use at your own peril!)
  -version to store different versions of the same run without over-writing them each time
  -dsName to specify either MNIST or CIFAR10
  -numEpochs to specify the number of training epochs of the diffusion model
