package com.github.maximtereshchenko.snapdragon;

sealed interface TrainingResult permits EarlyStop, End, NextEpoch {}
