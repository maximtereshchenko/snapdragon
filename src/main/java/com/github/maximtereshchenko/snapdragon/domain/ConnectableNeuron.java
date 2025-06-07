package com.github.maximtereshchenko.snapdragon.domain;

interface ConnectableNeuron {

    Connection connection(int destinationIndex);
}
