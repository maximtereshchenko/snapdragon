package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

final class MatrixTranspositionTests {

    @Test
    void givenSingleValue_whenTransposed_thenSameMatrix() {
        var matrix = Matrix.horizontalVector(1);
        assertThat(matrix.transposed()).isEqualTo(matrix);
    }

    @Test
    void givenTransposedMatrix_whenTransposed_thenSameMatrix() {
        var matrix = Matrix.horizontalVector(1, 2);
        assertThat(matrix.transposed().transposed()).isEqualTo(matrix);
    }

    @Test
    void givenOneRow_whenTransposed_thenOneColumnMatrix() {
        assertThat(Matrix.horizontalVector(1, 2).transposed())
            .isEqualTo(Matrix.verticalVector(1, 2));
    }

    @Test
    void givenOneColumn_whenTransposed_thenOneRowMatrix() {
        assertThat(Matrix.verticalVector(1, 2).transposed())
            .isEqualTo(Matrix.horizontalVector(1, 2));
    }

    @Test
    void givenMultiDimensionalMatrix_whenTransposed_thenRowsAndColumnsFlipped() {
        assertThat(
            Matrix.from(
                    new double[][]{
                        {1, 2, 3},
                        {4, 5, 6}
                    }
                )
                .transposed()
        )
            .isEqualTo(
                Matrix.from(
                    new double[][]{
                        {1, 4},
                        {2, 5},
                        {3, 6},
                    }
                )
            );
    }
}