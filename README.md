# Snapdragon: A From-Scratch Neural Network Implementation

This repository is a personal project aimed at deeply understanding the fundamental mechanics of
neural networks. It's an exploration into how these powerful models are represented, how they
compute predictions, and the intricate processes involved in their training.

## Motivation and Learning Journey

The primary motivation behind building Snapdragon was to gain a hands-on, ground-up understanding of
neural networks. By implementing core components from scratch, I aimed to demystify concepts such
as:

* **Neural Network Representation:** How layers, neurons, and connections are structured
  computationally.
* **Prediction Calculation:** The forward propagation process and activation functions.
* **Training Mechanisms:** Backpropagation, loss functions, and optimization algorithms.

During this journey, the learning extended beyond just neural networks. I gained significant
insights into the practical application and complexities of **matrices and tensors**, which are the
backbone of numerical computation in machine learning.

## Performance Optimization & Challenges

A significant challenge encountered during development was the **extremely slow training performance
** when attempting to process the MNIST dataset. To diagnose and address this bottleneck, I
leveraged specialized profiling tools:

1. **Async-profiler:** Used to pinpoint "hot spots" in the codebase, identifying the most
   time-consuming operations.
2. **JMH (Java Microbenchmark Harness):** Employed to rigorously compare the performance of the
   initial solution against improved implementations.

The profiling results indicated that the `Tensor` class's internal implementation was a major
performance bottleneck. To overcome this, the `Tensor` architecture was re-engineered:

* **Array-based Implementation:** Switched from a tree-based internal representation to a more
  performant array-based structure.
* **Multithreading:** Introduced multithreading when filling tensor values to leverage parallel
  processing.

The detailed benchmark results comparing these implementations are documented within the `Tensor`
class's Javadoc.

Despite these significant performance improvements, the project faced an ongoing challenge: **I was
still unable to successfully train the model on the full MNIST dataset within practical timeframes.
** While performance wasn't the sole goal, this highlights the immense computational demands of
neural network training, even for relatively small datasets, and the constant need for highly
optimized numerical libraries.

## Repository Contents

* **`src/`**: Contains the core Java source code for the neural network implementation.
* **`pom.xml`**: Maven project configuration file.
* **`*.idx*-ubyte` files**: These are the binary data files for the MNIST dataset (training images,
  training labels, test images, test labels).
