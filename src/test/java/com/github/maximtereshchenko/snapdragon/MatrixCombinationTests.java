package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

final class MatrixCombinationTests {

    @Test
    void givenDifferentHeight_whenCombined_thenIllegalArgumentExceptionThrown() {
        var oneRow = Matrix.horizontalVector(1);
        var twoRows = Matrix.verticalVector(1, 2);
        assertThatThrownBy(() -> oneRow.combined(twoRows, (a, b) -> a))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void givenDifferentWidth_whenCombined_thenIllegalArgumentExceptionThrown() {
        var oneColumn = Matrix.verticalVector(1);
        var twoColumns = Matrix.horizontalVector(1, 2);
        assertThatThrownBy(() -> oneColumn.combined(twoColumns, (a, b) -> a))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void givenSingleValueMatrices_whenCombined_thenValuesAdded() {
        assertThat(
            Matrix.horizontalVector(1)
                .combined(Matrix.horizontalVector(2), Double::sum)
        )
            .isEqualTo(Matrix.horizontalVector(1 + 2));
    }

    @Test
    void givenOneRow_whenCombined_thenValuesAdded() {
        assertThat(
            Matrix.horizontalVector(1, 2)
                .combined(Matrix.horizontalVector(3, 4), Double::sum)
        )
            .isEqualTo(Matrix.horizontalVector(1 + 3, 2 + 4));
    }

    @Test
    void givenTwoRows_whenCombined_thenValuesAdded() {
        assertThat(
            Matrix.verticalVector(1, 2)
                .combined(Matrix.verticalVector(3, 4), Double::sum)
        )
            .isEqualTo(Matrix.verticalVector(1 + 3, 2 + 4));
    }

    @Test
    void givenTwoRowsTwoColumns_whenCombined_thenValuesAdded() {
        assertThat(
            Matrix.from(
                    new double[][]{
                        {1, 2},
                        {3, 4}
                    }
                )
                .combined(
                    Matrix.from(
                        new double[][]{
                            {5, 6},
                            {7, 8}
                        }
                    ),
                    Double::sum
                )
        )
            .isEqualTo(
                Matrix.from(
                    new double[][]{
                        {1 + 5, 2 + 6},
                        {3 + 7, 4 + 8}
                    }
                )
            );
    }

    @Test
    void givenFunction_whenApplied_thenFunctionAppliedToEachValue() {
        assertThat(
            Matrix.from(
                    new double[][]{
                        {1, 2},
                        {3, 4}
                    }
                )
                .applied(value -> value + 1)
        )
            .isEqualTo(
                Matrix.from(
                    new double[][]{
                        {2, 3},
                        {4, 5}
                    }
                )
            );
    }
}