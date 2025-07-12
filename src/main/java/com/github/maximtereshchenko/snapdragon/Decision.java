package com.github.maximtereshchenko.snapdragon;

sealed interface Decision permits Improvement, NoImprovement, Stop {}
