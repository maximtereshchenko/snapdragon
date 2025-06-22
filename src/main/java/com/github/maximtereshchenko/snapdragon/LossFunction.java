package com.github.maximtereshchenko.snapdragon;

public interface LossFunction {

    Matrix loss(Matrix outputs, Matrix labels);

    Matrix derivative(Matrix outputs, Matrix labels);
}
