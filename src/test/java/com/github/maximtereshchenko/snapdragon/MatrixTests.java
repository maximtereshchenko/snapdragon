package com.github.maximtereshchenko.snapdragon;

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

    @Test
    void givenIncompatibleDimensions_whenBroadcasted_thenIllegalArgumentExceptionThrown() {
        var matrix = Matrix.from(
            new double[][]{
                {1, 2},
                {3, 4}
            }
        );
        assertThatThrownBy(() -> matrix.broadcasted(2, 3))
            .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void givenOneRow_whenBroadcasted_thenBroadcastedTwoRows() {
        assertThat(Matrix.horizontalVector(1, 2).broadcasted(2, 2))
            .isEqualTo(
                Matrix.from(
                    new double[][]{
                        {1, 2},
                        {1, 2}
                    }
                )
            );
    }

    @Test
    void givenOneColumn_whenBroadcasted_thenBroadcastedTwoColumns() {
        assertThat(Matrix.verticalVector(1, 2).broadcasted(2, 2))
            .isEqualTo(
                Matrix.from(
                    new double[][]{
                        {1, 1},
                        {2, 2}
                    }
                )
            );
    }
}