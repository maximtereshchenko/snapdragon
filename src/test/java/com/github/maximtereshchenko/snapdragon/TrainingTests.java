package com.github.maximtereshchenko.snapdragon;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;

final class TrainingTests {

    private final NeuralNetwork neuralNetwork = new FakeNeuralNetwork();

    @Test
    void givenZeroPatience_whenCompletedTraining_thenTwoEpochs() {
        assertThat(completedTraining(0, 10, 0.1, 0.5, 0.2, 0.5))
            .isEqualTo(
                new CompletedTraining(
                    neuralNetwork,
                    new Statistics(
                        List.of(
                            new EpochTrainingStatistics(List.of(0.1), 1.0),
                            new EpochTrainingStatistics(List.of(0.2), 1.0)
                        ),
                        List.of(
                            new EpochValidationStatistics(0.5, 1.0),
                            new EpochValidationStatistics(0.5, 1.0)
                        )
                    )
                )
            );
    }

    @Test
    void givenOneImprovementEpoch_whenCompletedTraining_thenThreeEpochs() {
        assertThat(completedTraining(0, 10, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1))
            .isEqualTo(
                new CompletedTraining(
                    neuralNetwork,
                    new Statistics(
                        List.of(
                            new EpochTrainingStatistics(List.of(0.4), 1),
                            new EpochTrainingStatistics(List.of(0.2), 1),
                            new EpochTrainingStatistics(List.of(0.1), 1)
                        ),
                        List.of(
                            new EpochValidationStatistics(0.3, 1),
                            new EpochValidationStatistics(0.1, 1),
                            new EpochValidationStatistics(0.1, 1)
                        )
                    )
                )
            );
    }

    @Test
    void givenOnePatienceNoImprovement_whenCompletedTraining_thenFourEpochs() {
        assertThat(
            completedTraining(1, 10, 0.6, 0.5, 0.4, 0.3, 0.2, 0.3, 0.1, 0.3)
        )
            .isEqualTo(
                new CompletedTraining(
                    neuralNetwork,
                    new Statistics(
                        List.of(
                            new EpochTrainingStatistics(List.of(0.6), 1),
                            new EpochTrainingStatistics(List.of(0.4), 1),
                            new EpochTrainingStatistics(List.of(0.2), 1),
                            new EpochTrainingStatistics(List.of(0.1), 1)
                        ),
                        List.of(
                            new EpochValidationStatistics(0.5, 1),
                            new EpochValidationStatistics(0.3, 1),
                            new EpochValidationStatistics(0.3, 1),
                            new EpochValidationStatistics(0.3, 1)
                        )
                    )
                )
            );
    }

    @Test
    void givenImprovingNeuralNetwork_whenCompletedTraining_thenTrainingFullyCompleted() {
        assertThat(completedTraining(0, 3, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1))
            .isEqualTo(
                new CompletedTraining(
                    neuralNetwork,
                    new Statistics(
                        List.of(
                            new EpochTrainingStatistics(List.of(0.6), 1),
                            new EpochTrainingStatistics(List.of(0.4), 1),
                            new EpochTrainingStatistics(List.of(0.2), 1)
                        ),
                        List.of(
                            new EpochValidationStatistics(0.5, 1),
                            new EpochValidationStatistics(0.3, 1),
                            new EpochValidationStatistics(0.1, 1)
                        )
                    )
                )
            );
    }

    @Test
    void givenWrongPredictions_whenCompletedTraining_thenLowAccuracy() {
        assertThat(
            completedTraining(
                Map.of(
                    new double[]{0.8, 0.2}, new double[]{0.0, 1.0},
                    new double[]{0.7, 0.3}, new double[]{0.0, 1.0},
                    new double[]{0.6, 0.4}, new double[]{0.0, 1.0},
                    new double[]{0.4, 0.6}, new double[]{0.0, 1.0}
                ),
                4,
                0,
                1,
                0.1,
                0.1
            )
        )
            .isEqualTo(
                new CompletedTraining(
                    neuralNetwork,
                    new Statistics(
                        List.of(
                            new EpochTrainingStatistics(List.of(0.1), 0.25)
                        ),
                        List.of(
                            new EpochValidationStatistics(0.1, 0.25)
                        )
                    )
                )
            );
    }

    @Test
    void givenBatchedTrainingDataset_whenCompletedTraining_thenTrainingLossesPerBatch() {
        assertThat(
            completedTraining(
                Map.of(
                    new double[]{0.1}, new double[]{0.1},
                    new double[]{0.2}, new double[]{0.2},
                    new double[]{0.3}, new double[]{0.3},
                    new double[]{0.4}, new double[]{0.4},
                    new double[]{0.5}, new double[]{0.5}
                ),
                2,
                0,
                1,
                0.3, 0.2, 0.1, 0.4, 0.3, 0.2
            )
        )
            .isEqualTo(
                new CompletedTraining(
                    neuralNetwork,
                    new Statistics(
                        List.of(new EpochTrainingStatistics(List.of(0.3, 0.2, 0.1), 1)),
                        List.of(new EpochValidationStatistics(0.3, 1))
                    )
                )
            );
    }

    private CompletedTraining completedTraining(
        int patience,
        int epochs,
        Double... losses
    ) {
        var doubles = new double[]{0};
        return completedTraining(Map.of(doubles, doubles), 1, patience, epochs, losses);
    }

    private CompletedTraining completedTraining(
        Map<double[], double[]> samples,
        int batchSize,
        int patience,
        int epochs,
        Double... losses
    ) {
        var labeledSamples = samples.entrySet()
                                 .stream()
                                 .<LabeledSample>map(entry ->
                                                         new StaticLabeledSample(
                                                             entry.getKey(),
                                                             entry.getValue()
                                                         )
                                 )
                                 .toList();
        return new Training(
            new TrainingDataset(() -> labeledSamples, batchSize, new Random(0)),
            new ValidationDataset(() -> labeledSamples, batchSize),
            new FakeLossFunction(losses),
            neuralNetwork,
            new LearningRate(1),
            new Patience(patience),
            epochs
        )
                   .completedTraining();
    }
}