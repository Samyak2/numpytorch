# numpytorch

Simple neural network implementation in numpy with a PyTorch-like API

<sup><sub>Originally made for an assignment for the course "Machine Intelligence" at PES University. Although the commits are recent, most of the code was written during the course (Oct/Nov 2020) and moved from a different repo.</sub></sup>

## What can you do with it?

This is not meant to be used for serious workloads. You can use it as a learning tool though. For example, here is a list of things you could try learning from the code here:

 - Modular implementation of neural networks - each layer is a module with many trainable parameters. Refer [nn.py](./numpytorch/nn.py)
     - This implementation is also very extensible - you can make your modules with various behaviour, such as [Dense](https://github.com/Samyak2/numpytorch/blob/49ee7bb6681d2e1d56802a1e23d304151bcdc512/numpytorch/nn.py#L59) (a fully connected layer) and even something meta like [Sequential](https://github.com/Samyak2/numpytorch/blob/49ee7bb6681d2e1d56802a1e23d304151bcdc512/numpytorch/nn.py#L111) (a chain of layers).
     - Similarly, [activation functions](./numpytorch/activations.py), [loss functions](./numpytorch/losses.py) and [optimisers](./numpytorch/optim.py) are also modular and extensible.
 - Usage of Einstein summation operations in numpy (and in general). [Here's](https://stackoverflow.com/a/33641428/11199009) a nice reference for Einstein summation.
 - Type annotations in python - the codebase is almost completely [type-annotated](https://realpython.com/python-type-checking/). This makes the code a little easier to maintain and improves the editing experience significantly for users of the library. Although, [mypy](https://github.com/python/mypy) does report a few errors, most of the type annotations are correct (PRs are welcome to fix this).

## Some possible future plans

I don't plan to develop this further, but if you want to learn, you can try implementing the following (either in your own fork or send a PR!):

 - [ ] More activation functions. `numpytorch/activations.py` has a limited set of activation functions, there are many more you can add.
 - [ ] More loss functions. `numpytorch/losses.py` has only one loss function (binary cross-entropy).
 - [ ] More optimisers. `numpytorch/optim.py` has only one optimiser (Stochastic Gradient Descent, SGD) with support for L2 regularization and momentum. The [ADAM](https://arxiv.org/abs/1412.6980) optimiser would be a nice addition.
 - [ ] Automatic differentiation. Currently, backward passes (derivatives) have to be hand-coded into all the activation functions, layers, etc. Integrating some kind of automatic differentiation library (like [autograd](https://github.com/HIPS/autograd) or [autodidact](https://github.com/mattjj/autodidact)) would make this a lot less painful to customize. You could also try writing your own automatic differentiation library, that will be a fun project! ([ref](https://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/readings/L06%20Automatic%20Differentiation.pdf))
 - [ ] Other fancy layers like convolution, recurrent cells, etc.

## Acknowledgements

Team members [Aayush](https://github.com/NaikAayush/) and [Bhargav](https://github.com/bhargavsk1077/) for helping.

## License

numpytorch is [MIT Licensed](./LICENSE)
