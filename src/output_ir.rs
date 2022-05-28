//! Output data structures from the audio graph compiler.
//!
use crate::input_ir::Edge;
use serde::{Deserialize, Serialize};

/// A [CompiledSchedule] is the output of the graph compiler.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompiledSchedule {
    /// A list of nodes, delays, and summing points to
    /// evaluate in order to render audio, in topological order.
    pub schedule: Vec<ScheduleEntry>,
    /// A list of delays that were inserted into the graph.
    pub delays: Vec<InsertedDelay>,
    /// The total number of buffers required to allocate, for
    /// each type of port.
    pub num_buffers: Vec<usize>,
}

/// A [ScheduleEntry] is one element of the schedule to evalute.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ScheduleEntry {
    /// One of the input nodes, to process
    Node(ScheduledNode),
    /// A delay that was inserted for latency compensation
    Delay(InsertedDelay),
    /// A sum that was inserted to merge multiple inputs into
    /// the same port.
    Sum(InsertedSum),
}

/// A [ScheduledNode] is a [Node] that has been assigned buffers
/// and a place in the schedule.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScheduledNode {
    /// The unique ID of this node.
    pub id: u64,
    /// The latency of this node. Kept for debugging and visualization.
    pub latency: f64,
    /// The assigned input buffers.
    pub input_buffers: Vec<BufferAssignment>,
    /// The assigned output buffers.
    pub output_buffers: Vec<BufferAssignment>,
}

/// An [InsertedDelay] represents a required delay node to be inserted
/// along some edge in order to compensate for different latencies along
/// paths of the graph.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct InsertedDelay {
    /// The edge that this delay corresponds to. Kept for debugging and visualization.
    pub edge: Edge,
    /// The amount of delay to apply to the input.
    pub delay: f64,
    /// The input data to read.
    pub input_buffer: BufferAssignment,
    /// The output buffer to write delayed into to.
    pub output_buffer: BufferAssignment,
}

/// An [InsertedSum] represents a point where multiple edges need to be merged
/// into a single buffer, in order to support multiple inputs into the same
/// port.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InsertedSum {
    /// The input buffers that will be summed
    pub input_buffers: Vec<BufferAssignment>,
    /// The output buffer to write to
    pub output_buffer: BufferAssignment,
}

/// A [Buffer Assignment] represents a single buffer assigned to an input
/// or output port.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct BufferAssignment {
    /// The index of the buffer assigned
    pub buffer_index: usize,
    /// The index of the type of data in this buffer
    pub type_index: usize,
    /// Whether the engine should clear the buffer before
    /// passing it to a process
    pub should_clear: bool,
    /// The ID of the port this buffer is mapped to
    pub port_id: usize,
    /// The ID of the node this buffer is mapped to
    pub node_id: usize,
    /// Buffers are reused, the "generation" represnts
    /// how many times this buffer has been used before
    /// this assignment. Kept for debugging and visualization.
    pub generation: usize,
}
