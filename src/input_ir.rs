//! Input data structures to the audio graph compiler.
//!
use serde::{Deserialize, Serialize};

/// The input IR used by the audio graph compiler.  
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioGraphCompilerInput {
    /// A list of nodes in the graph.
    pub nodes: Vec<Node>,
    /// A list of edges in the graph.
    pub edges: Vec<Edge>,
    /// The number of different port types used by the graph.
    pub num_port_types: usize,
}

/// A [Node] is a single process in the audio network.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    /// A globally unique identifier of the node.
    pub id: u64,
    /// A list of input ports used by the node
    pub inputs: Vec<Port>,
    /// A list of output ports used by the node.
    pub outputs: Vec<Port>,
    /// The latency this node adds to data flowing through it.
    pub latency: f64,
}

/// A [Port] is a single point of input or output data
/// for a node.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Port {
    /// A globally unique identifier of this port. Note: do not
    /// mix IDs for ports with IDs for nodes and edges.
    pub id: u64,
    /// A unique identifier for the type of data this port handles,
    /// for example nodes may have audio and event ports.
    pub type_idx: usize,
}

/// An [Edge] is a connection from source node and port to a
/// destination node and port.
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    /// A globally unique identifier for this connection. Note: do
    /// not mix IDs for edges with IDs for nodes or ports.
    pub id: u64,
    /// The ID of the source node of this edge.
    pub src_node: u64,
    /// The ID of the source port used by this edge.
    pub src_port: u64,
    /// The ID of the destination of this edge.
    pub dst_node: u64,
    /// The ID of the destination port used by this edge.
    pub dst_port: u64,
}
