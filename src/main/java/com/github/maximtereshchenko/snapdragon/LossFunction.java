package com.github.maximtereshchenko.snapdragon;

interface LossFunction {

    Tensor loss(Tensor outputs, Tensor labels);

    Tensor derivative(Tensor outputs, Tensor labels);
}
