package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

final class MatrixBroadcastingTests {

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