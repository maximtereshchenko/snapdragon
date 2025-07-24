package com.github.maximtereshchenko.snapdragon;

interface ActivationFunction {

    Tensor apply(Tensor tensor);

    Tensor deltas(Tensor outputs, Tensor errorSignal);
}
