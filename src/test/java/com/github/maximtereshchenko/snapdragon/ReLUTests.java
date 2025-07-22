package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

final class ReLUTests {

    private final ActivationFunction activationFunction = new ReLU();

    @ParameterizedTest
    @CsvSource(
        textBlock = """
                    -1, 0
                    -0.5, 0
                    0, 0
                    0.5, 0.5
                    1, 1
                    """
    )
    void givenSingleInput_whenApply_thenSingleActivatedOutput(double input, double output) {
        assertThat(activationFunction.apply(Tensor.horizontalVector(input)))
            .isEqualTo(Tensor.horizontalVector(output));
    }

    @Test
    void givenMultipleInputs_whenApply_thenExpectedOutputs() {
        assertThat(
            activationFunction.apply(
                Tensor.from(
                    List.of(2, 3),
                    -1, -0.5, 0,
                    0, 0.5, 1
                )
            )
        )
            .isEqualTo(
                Tensor.from(
                    List.of(2, 3),
                    0, 0, 0,
                    0, 0.5, 1
                )
            );
    }

    @ParameterizedTest
    @CsvSource(
        textBlock = """
                    -1, 0
                    -0.5, 0
                    0, 0
                    0.5, 1
                    1, 1
                    """
    )
    void givenSingleInput_whenDerivative_thenExpectedDerivativeValue(
        double input,
        double derivative
    ) {
        assertThat(activationFunction.derivative(Tensor.horizontalVector(input)))
            .isEqualTo(Tensor.horizontalVector(derivative));
    }

    @Test
    void givenMultipleInputs_whenApply_thenExpectedDerivatives() {
        assertThat(
            activationFunction.derivative(
                Tensor.from(
                    List.of(2, 3),
                    -1, -0.5, 0,
                    0, 0.5, 1
                )
            )
        )
            .isEqualTo(
                Tensor.from(
                    List.of(2, 3),
                    0, 0, 0,
                    0, 1, 1
                )
            );
    }
}