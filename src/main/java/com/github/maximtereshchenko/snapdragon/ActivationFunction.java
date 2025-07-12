package com.github.maximtereshchenko.snapdragon;

interface ActivationFunction {

    Matrix apply(Matrix matrix);

    Matrix derivative(Matrix matrix);
}
