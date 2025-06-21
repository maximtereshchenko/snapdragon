package com.github.maximtereshchenko.snapdragon;

public interface LossFunction {

    Matrix derivative(Matrix outputs, Matrix labels);
}
