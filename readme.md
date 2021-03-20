# audio-graph

Work in progress audio graph implementation

## Goals:

- The graph uses an intermediate representation of nodes in a directed graph
- Buffers are abstract identified by indices
- The graph is "compiled" into a schedule, containing buffer allocations and process order

## Usage:

```rust
use audio_graph::*;

let mut graph = Graph::default();
let input1 = graph.node("Input 1");
let input2 = graph.node("Input 2");
let output = graph.node("Output");

let in_port1 = graph.port(input1, PortType::Audio, "input-1")?;
let in_port2 = graph.port(input2, PortType::Audio, "input-1")?;
let out_port = graph.port(output, PortType::Audio, "mixbus")?;

graph.connect(port1, out_port);
graph.connect(port2, out_port);

let schedule = graph.compile(output).collect::<Vec<_>>();
for entry in schedule {
    let node_name = graph.node_name(entry.node).unwrap();
    // inputs may have multiple buffers to handle.
    for (port, buffers) in entry.inputs {
        for b in buffers {
            // ... 
        }
    }
    // outputs have exactly one buffer they write to
    for (port, buf) in entry.outputs {
        // ... 
    }
}
```

## Constraints:

- Ports are typed, `PortType::Audio` or `PortType::Event`. Ports must be connected to ports of the same type.
- Ports can be bidirectional. This is a quirk of the current implementation, but don't rely on it.
- Delay compensation is currently unsound, it will result in invalid schedules.
- Cycles are not supported.
