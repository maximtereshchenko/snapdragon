package com.github.maximtereshchenko.snapdragon.domain;

import java.util.List;

interface Connectable {

    List<Connection> connections(int destinationIndex);
}
