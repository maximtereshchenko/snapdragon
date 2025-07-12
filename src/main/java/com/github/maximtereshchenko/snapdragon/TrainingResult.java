package com.github.maximtereshchenko.snapdragon;

sealed interface TrainingResult permits NextEpoch, TrainedNeuralNetwork {}
