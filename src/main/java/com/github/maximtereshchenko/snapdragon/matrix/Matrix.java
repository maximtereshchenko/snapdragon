package com.github.maximtereshchenko.snapdragon.matrix;

import java.util.Arrays;

public final class Matrix {

    private final double[][] values;

    private Matrix(double[][] values) {
        this.values = values;
    }

    public static Matrix from(double[][] matrix) {
        if (matrix.length == 0) {
            throw new IllegalArgumentException();
        }
        return new Matrix(matrix);
    }

    public static Matrix horizontalVector(double... values) {
        return from(new double[][]{values});
    }

    @Override
    public int hashCode() {
        return Arrays.deepHashCode(values);
    }

    @Override
    public boolean equals(Object object) {
        if (this == object) {
            return true;
        }
        return object instanceof Matrix matrix &&
                   Arrays.deepEquals(values, matrix.values);
    }

    @Override
    public String toString() {
        return Arrays.deepToString(values);
    }

    public Matrix product(Matrix matrix) {
        return new Matrix(new double[][]{{values[0][0] * matrix.values[0][0]}});
    }
}
