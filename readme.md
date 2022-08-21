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

(TODO)

## Constraints:

- Ports are typed, and only ports of the same type can be connected.
- Ports are unidirectional.
- Cycles are not supported, and an error will be returned if tried.
- Multithreaded schedules are not yet implemented.
