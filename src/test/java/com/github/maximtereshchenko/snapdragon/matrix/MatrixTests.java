package com.github.maximtereshchenko.snapdragon.matrix;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

final class MatrixTests {

    @Test
    void givenEmptyArray_whenCreateMatrix_thenIllegalArgumentExceptionThrown() {
        assertThatThrownBy(() -> Matrix.from(new double[0][0]))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void givenVariableLengthMatrix_whenCreateMatrix_thenIllegalArgumentExceptionThrown() {
        assertThatThrownBy(() -> Matrix.from(new double[][]{{1}, {2, 3}}))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void givenIncompatibleMatrices_whenProduct_thenIllegalArgumentExceptionThrown() {
        var twoColumns = Matrix.horizontalVector(1, 2);
        var oneRow = Matrix.horizontalVector(1);
        assertThatThrownBy(() -> twoColumns.product(oneRow))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void givenSingleValueMatrices_whenProduct_thenValuesMultiplied() {
        assertThat(Matrix.horizontalVector(2).product(Matrix.horizontalVector(3)))
            .isEqualTo(Matrix.horizontalVector(6));
    }

    @Test
    void givenOneRowOneColumn_whenProduct_thenValuesMultiplied() {
        assertThat(Matrix.horizontalVector(1, 2).product(Matrix.verticalVector(3, 4)))
            .isEqualTo(Matrix.horizontalVector(1 * 3 + 2 * 4));
    }
}