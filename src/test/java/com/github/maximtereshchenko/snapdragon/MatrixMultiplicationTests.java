package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

final class MatrixMultiplicationTests {

    @Test
    void givenIncompatibleMatrices_whenProduct_thenIllegalArgumentExceptionThrown() {
        var twoColumns = Matrix.horizontalVector(1, 2);
        var oneRow = Matrix.horizontalVector(1);
        assertThatThrownBy(() -> twoColumns.product(oneRow))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void givenSingleValueMatrices_whenProduct_thenValuesMultiplied() {
        assertThat(
            Matrix.horizontalVector(2)
                .product(Matrix.horizontalVector(3))
        )
            .isEqualTo(Matrix.horizontalVector(2 * 3));
    }

    @Test
    void givenOneRowOneColumn_whenProduct_thenValuesMultiplied() {
        assertThat(
            Matrix.horizontalVector(1, 2)
                .product(Matrix.verticalVector(3, 4))
        )
            .isEqualTo(Matrix.horizontalVector(1 * 3 + 2 * 4));
    }

    @Test
    void givenTwoRowsOneColumn_whenProduct_thenValuesMultiplied() {
        assertThat(
            Matrix.verticalVector(1, 2)
                .product(Matrix.verticalVector(3))
        )
            .isEqualTo(Matrix.from(new double[][]{{1 * 3}, {2 * 3}}));
    }

    @Test
    void givenTwoRowsTwoColumns_whenProduct_thenValuesMultiplied() {
        assertThat(
            Matrix.from(
                    new double[][]{
                        {1, 2},
                        {3, 4}
                    }
                )
                .product(
                    Matrix.from(
                        new double[][]{
                            {5, 6},
                            {7, 8}
                        }
                    )
                )
        )
            .isEqualTo(
                Matrix.from(
                    new double[][]{
                        {1 * 5 + 2 * 7, 1 * 6 + 2 * 8},
                        {3 * 5 + 4 * 7, 3 * 6 + 4 * 8}
                    }
                )
            );
    }
}