use std::error::Error;
use std::fmt;

use crate::{Edge, NodeID, Port, PortID, TypeIdx};

/// An error occurred while attempting to add a port to the graph.
#[derive(Debug, Clone, Copy)]
pub enum AddPortError {
    /// The given node was not found in the graph.
    NodeNotFound(NodeID),
    /// An port with this ID already exists on this node.
    PortAlreadyExists(NodeID, PortID),
    /// The type index of this port is greater than or equal to
    /// the total number of port types set for this graph.
    TypeIndexOutOfBounds(TypeIdx, usize),
}

impl Error for AddPortError {}

impl fmt::Display for AddPortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeNotFound(node_id) => {
                write!(f, "Could not find node with ID {:?}", node_id)
            }
            Self::PortAlreadyExists(node_id, port_id) => {
                write!(
                    f,
                    "Could not add port: port with ID {:?} already exists in node with ID {:?}",
                    port_id, node_id,
                )
            }
            Self::TypeIndexOutOfBounds(type_idx, num_types) => {
                write!(
                    f,
                    "Type index {:?} is out of bounds in graph with {} types",
                    type_idx, num_types
                )
            }
        }
    }
}

/// An error occurred while attempting to remove a port from the
/// graph.
#[derive(Debug, Clone, Copy)]
pub enum RemovePortError {
    /// The given node was not found in the graph.
    NodeNotFound(NodeID),
    /// The given input port was not found in the graph.
    InPortNotFound(NodeID, PortID),
    /// The given output was not found in the graph.
    OutPortNotFound(NodeID, PortID),
}

impl Error for RemovePortError {}

impl fmt::Display for RemovePortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeNotFound(node_id) => {
                write!(f, "Could not find node with ID {:?}", node_id)
            }
            Self::InPortNotFound(node_id, port_id) => {
                write!(
                    f,
                    "Could not remove port: input port with ID {:?} was not found in node with ID {:?}",
                    port_id,
                    node_id,
                )
            }
            Self::OutPortNotFound(node_id, port_id) => {
                write!(
                    f,
                    "Could not remove port: output port with ID {:?} was not found in node with ID {:?}",
                    port_id,
                    node_id,
                )
            }
        }
    }
}

/// An error occurred while attempting to add an edge to the graph.
#[derive(Debug, Clone)]
pub enum AddEdgeError {
    /// The given source node was not found in the graph.
    SrcNodeNotFound(NodeID),
    /// The given destination node was not found in the graph.
    DstNodeNotFound(NodeID),
    /// The given source port was not found in the graph.
    SrcPortNotFound(NodeID, PortID),
    /// The given destination port was not found in the graph.
    DstPortNotFound(NodeID, PortID),
    /// The source port and the destination port have different
    /// type indexes.
    TypeMismatch {
        src_node_id: NodeID,
        src_port: Port,
        dst_node_id: NodeID,
        dst_port: Port,
    },
    /// The edge already exists in the graph.
    EdgeAlreadyExists(Edge),
    /// This edge would have created a cycle in the graph.
    CycleDetected,
}

impl Error for AddEdgeError {}

impl fmt::Display for AddEdgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SrcNodeNotFound(node_id) => {
                write!(
                    f,
                    "Could not add edge: could not find source node with ID {:?}",
                    node_id
                )
            }
            Self::DstNodeNotFound(node_id) => {
                write!(
                    f,
                    "Could not add edge: could not find destination node with ID {:?}",
                    node_id
                )
            }
            Self::SrcPortNotFound(node_id, port_id) => {
                write!(
                    f,
                    "Could not add edge: could not find source port with ID {:?} on node with ID {:?}",
                    port_id,
                    node_id,
                )
            }
            Self::DstPortNotFound(node_id, port_id) => {
                write!(
                    f,
                    "Could not add edge: could not find destination port with ID {:?} on node with ID {:?}",
                    port_id,
                    node_id,
                )
            }
            Self::TypeMismatch {
                src_node_id,
                src_port,
                dst_node_id,
                dst_port,
            } => {
                write!(
                    f,
                    "Could not add edge: source port {:?} on node {:?} is of type {:?} but destination port {:?} on node {:?} is of type {:?}", 
                    src_port.id,
                    src_node_id,
                    src_port.type_idx,
                    dst_port.id,
                    dst_node_id,
                    dst_port.type_idx
                )
            }
            Self::EdgeAlreadyExists(edge) => {
                write!(
                    f,
                    "Could not add edge: edge {:?} already exists in the graph",
                    edge
                )
            }
            Self::CycleDetected => {
                write!(f, "Could not add edge: cycle was detected")
            }
        }
    }
}
