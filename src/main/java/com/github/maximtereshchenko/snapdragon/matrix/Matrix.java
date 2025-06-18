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
        for (var i = 1; i < matrix.length; i++) {
            if (matrix[i - 1].length != matrix[i].length) {
                throw new IllegalArgumentException();
            }
        }
        return new Matrix(matrix);
    }

    public static Matrix horizontalVector(double... values) {
        return from(new double[][]{values});
    }

    public static Matrix verticalVector(double... values) {
        var matrix = new double[values.length][];
        for (var i = 0; i < values.length; i++) {
            matrix[i] = new double[]{values[i]};
        }
        return from(matrix);
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
        if (columns() != matrix.rows()) {
            throw new IllegalArgumentException();
        }
        var product = new double[rows()][matrix.columns()];
        for (var rowIndex = 0; rowIndex < rows(); rowIndex++) {
            for (var columnIndex = 0; columnIndex < matrix.columns(); columnIndex++) {
                var row = row(rowIndex);
                var column = matrix.column(columnIndex);
                for (var i = 0; i < row.length; i++) {
                    product[rowIndex][columnIndex] += row[i] * column[i];
                }
            }
        }
        return new Matrix(product);
    }

    Matrix hadamardProduct(Matrix matrix) {
        if (rows() != matrix.rows() || columns() != matrix.columns()) {
            throw new IllegalArgumentException();
        }
        var hadamardProduct = new double[rows()][columns()];
        for (var rowIndex = 0; rowIndex < rows(); rowIndex++) {
            for (var columnIndex = 0; columnIndex < columns(); columnIndex++) {
                hadamardProduct[rowIndex][columnIndex] =
                    values[rowIndex][columnIndex] * matrix.values[rowIndex][columnIndex];
            }
        }
        return new Matrix(hadamardProduct);
    }

    private int rows() {
        return values.length;
    }

    private int columns() {
        return values[0].length;
    }

    private double[] row(int index) {
        return values[index];
    }

    private double[] column(int index) {
        var column = new double[rows()];
        for (var i = 0; i < rows(); i++) {
            column[i] = values[i][index];
        }
        return column;
    }
}
