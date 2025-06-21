package com.github.maximtereshchenko.snapdragon;

public interface ActivationFunction {

    Matrix apply(Matrix matrix);

    Matrix derivative(Matrix matrix);
}
