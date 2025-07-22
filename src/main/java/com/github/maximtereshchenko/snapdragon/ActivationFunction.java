package com.github.maximtereshchenko.snapdragon;

interface ActivationFunction {

    Tensor apply(Tensor tensor);

    Tensor derivative(Tensor tensor);
}
