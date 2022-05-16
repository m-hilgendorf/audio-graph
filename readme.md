# audio-graph

Work in progress audio graph implementation

## Goals:

- The graph uses an intermediate representation of nodes in a directed graph
- Buffers are abstract identified by indices
- The graph is "compiled" into a schedule, containing buffer allocations and process order
- Delay compensation is automatically added to each needed input buffer inside the schedule
- The graph supports one-to-many and many-to-one connections
- The graph has built-in safeguards like cycle detection
- Multi-threaded schedules can be produced using any given number of threads

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

// set input_1 to have a delay of 2 samples
graph.set_delay(input_1, 2)?;

for entry in graph.compile() {
    let node_name = entry.node;

    // inputs may have multiple buffers to handle, in which case the
    // buffers will need to be mixed together into a single buffer
    // before being sent to the node
    for (port_name, buffers) in entry.inputs {
        for (buf, delay_comp) in buffers {
            // id (index) of the buffer
            let buffer_id = buf.buffer_id;

            if let Some(delay_comp) = delay_comp {
                // delay compensation that needs to be applied to the
                // contents of this buffer before being mixed & sent to
                // the node
                let delay = delay_comp.delay;
            }

            // ... 
        }
    }
    // outputs have exactly one buffer they write to
    for (port_name, buf) in entry.outputs {
        // id (index) of the buffer
        let buffer_id = buf.buffer_id;

        // ... 
    }
}
```

## Constraints:

- Ports are typed, and only ports of the same type can be connected. By default there is the `DefaultPortType` enum which has `Audio` and `Event`, but you may define your own types if you wish.
- Ports are unidirectional.
- Cycles are not supported, and an error will be returned if tried.
- Multithreaded schedules are not yet implemented.
