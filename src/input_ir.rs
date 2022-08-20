//! Input data structures to the audio graph compiler.

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// A globally unique identifier for a [Node].
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeID(pub u64);

/// A globally unique identifier for a [Port].
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PortID {
    /// The ID of the [Node] this [Port] belongs to.
    pub node: NodeID,

    /// The (stable) ID for this [Port] on this [Node].
    ///
    /// This does not need to be a globally unique identifier,
    /// just unique to the [Node] it belongs to.
    pub stable_id: PortStableID,
}

/// The (stable) ID for a [Port] on a particular [Node].
///
/// This does not need to be a globally unique identifier,
/// just unique to the [Node] it belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PortStableID(pub u32);

/// A globally unique identifier for an [Edge].
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeID(pub u64);

/// The index of the port/buffer type.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeIdx(pub usize);

/// The input IR used by the audio graph compiler.  
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct AudioGraphCompilerInput {
    /// A list of nodes in the graph.
    pub nodes: Vec<Node>,
    /// A list of edges in the graph.
    pub edges: Vec<Edge>,
    /// The number of different port types used by the graph.
    pub num_port_types: usize,
}

/// A [Node] is a single process in the audio network.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct Node {
    /// A globally unique identifier of the node.
    pub id: NodeID,
    /// A list of input ports used by the node
    pub inputs: Vec<Port>,
    /// A list of output ports used by the node.
    pub outputs: Vec<Port>,
    /// The latency this node adds to data flowing through it.
    pub latency: f64,
}

/// A [Port] is a single point of input or output data
/// for a node.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
pub struct Port {
    /// A globally unique identifier of this port.
    pub id: PortID,
    /// A unique identifier for the type of data this port handles,
    /// for example nodes may have audio and event ports.
    pub type_idx: TypeIdx,
}

/// An [Edge] is a connection from source node and port to a
/// destination node and port.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub struct Edge {
    /// A globally unique identifier for this connection.
    pub id: EdgeID,
    /// The ID of the source port used by this edge.
    pub src_port: PortID,
    /// The ID of the destination port used by this edge.
    pub dst_port: PortID,
}
