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

let in_port1 = graph.port(input1, DefaultPortType::Audio, "input-1")?;
let in_port2 = graph.port(input2, DefaultPortType::Audio, "input-1")?;
let out_port = graph.port(output, DefaultPortType::Audio, "mixbus")?;

graph.connect(port1, out_port)?;
graph.connect(port2, out_port)?;

let schedule = graph.compile(output);
for entry in schedule {
    let node_name = entry.node;
    // inputs may have multiple buffers to handle.
    for (port_name, buffers) in entry.inputs {
        for b in buffers {
            // ... 
        }
    }
    // outputs have exactly one buffer they write to
    for (port_name, buf) in entry.outputs {
        // ... 
    }
}
```

## Constraints:

- Ports are typed, and only ports of the same type can be connected. By default there is the `DefaultPortType` enum which has `Audio` and `Event`, but you may define your own types if you wish.
- Ports can be bidirectional. This is a quirk of the current implementation, but don't rely on it.
- Delay compensation is currently unsound, it will result in invalid schedules.
- Cycles are not supported, and and error will be returned if tried. (However, cycle detection is currently broken atm).
