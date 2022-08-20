use fnv::{FnvHashMap, FnvHashSet};
use std::error::Error;
use std::fmt;

use crate::{Edge, EdgeID, Node, NodeID, Port, PortID, TypeIdx};

#[derive(Debug, Clone)]
pub struct NodeEdges {
    pub incoming: Vec<Edge>,
    pub outgoing: Vec<Edge>,
}

impl NodeEdges {
    fn remove_incoming(&mut self, edge_id: EdgeID) {
        let mut found = None;
        for (i, e) in self.incoming.iter().enumerate() {
            if e.id == edge_id {
                found = Some(i);
                break;
            }
        }
        if let Some(i) = found {
            self.incoming.remove(i);
        }
    }

    fn remove_outgoing(&mut self, edge_id: EdgeID) {
        let mut found = None;
        for (i, e) in self.outgoing.iter().enumerate() {
            if e.id == edge_id {
                found = Some(i);
                break;
            }
        }
        if let Some(i) = found {
            self.outgoing.remove(i);
        }
    }
}

pub struct AudioGraphHelper {
    nodes: FnvHashMap<NodeID, Node>,
    edges: FnvHashMap<EdgeID, Edge>,

    node_edges: FnvHashMap<NodeID, NodeEdges>,

    next_node_id: u64,
    next_edge_id: u64,

    nodes_with_dirty_edges: FnvHashSet<NodeID>,
    needs_recompile: bool,

    num_port_types: usize,
}

impl AudioGraphHelper {
    pub fn new(num_port_types: usize) -> Self {
        assert_ne!(num_port_types, 0);

        Self {
            nodes: FnvHashMap::default(),
            edges: FnvHashMap::default(),
            node_edges: FnvHashMap::default(),
            next_node_id: 0,
            next_edge_id: 0,
            nodes_with_dirty_edges: FnvHashSet::default(),
            num_port_types,
            needs_recompile: false,
        }
    }

    pub fn add_new_node(&mut self, latency: f64) -> NodeID {
        let new_id = NodeID(self.next_node_id);
        self.next_node_id += 1;

        let new_node = Node {
            id: new_id,
            inputs: vec![],
            outputs: vec![],
            latency,
        };

        self.nodes.insert(new_id, new_node);

        self.nodes_with_dirty_edges.insert(new_id);

        self.needs_recompile = true;

        new_id
    }

    pub fn remove_node(&mut self, node_id: NodeID) -> Result<(), ()> {
        let node = self.nodes.remove(&node_id).ok_or(())?;
        self.node_edges.remove(&node_id).unwrap();

        for port in node.inputs.iter().chain(node.outputs.iter()) {
            self.remove_edges_with_port(port.id);
        }

        self.nodes_with_dirty_edges.remove(&node_id);

        self.needs_recompile = true;

        Ok(())
    }

    pub fn node_edges(&self, node_id: NodeID) -> Option<&NodeEdges> {
        self.node_edges.get(&node_id)
    }

    pub fn add_port(
        &mut self,
        port_id: PortID,
        port_type: TypeIdx,
        is_input: bool,
    ) -> Result<(), AddPortError> {
        if port_type.0 >= self.num_port_types {
            return Err(AddPortError::TypeIndexOutOfBounds(
                port_type,
                self.num_port_types,
            ));
        }

        let node = self
            .nodes
            .get_mut(&port_id.node)
            .ok_or(AddPortError::NodeNotFound(port_id.node))?;

        let new_port = Port {
            id: port_id,
            type_idx: port_type,
        };

        if is_input {
            for p in node.inputs.iter() {
                if p.id.stable_id == port_id.stable_id {
                    return Err(AddPortError::InPortAlreadyExists(port_id));
                }
            }

            node.inputs.push(new_port);
        } else {
            for p in node.outputs.iter() {
                if p.id.stable_id == port_id.stable_id {
                    return Err(AddPortError::OutPortAlreadyExists(port_id));
                }
            }

            node.inputs.push(new_port);
        }

        self.needs_recompile = true;

        Ok(())
    }

    pub fn remove_port(&mut self, port_id: PortID, is_input: bool) -> Result<(), RemovePortError> {
        let node = self
            .nodes
            .get_mut(&port_id.node)
            .ok_or(RemovePortError::NodeNotFound(port_id.node))?;

        if is_input {
            let mut found = None;
            for (i, p) in node.inputs.iter().enumerate() {
                if p.id.stable_id == port_id.stable_id {
                    found = Some(i);
                    break;
                }
            }
            if let Some(i) = found {
                node.inputs.remove(i);
            } else {
                return Err(RemovePortError::InPortNotFound(port_id));
            }
        } else {
            let mut found = None;
            for (i, p) in node.outputs.iter().enumerate() {
                if p.id.stable_id == port_id.stable_id {
                    found = Some(i);
                    break;
                }
            }
            if let Some(i) = found {
                node.outputs.remove(i);
            } else {
                return Err(RemovePortError::OutPortNotFound(port_id));
            }
        };

        self.remove_edges_with_port(port_id);

        self.needs_recompile = true;

        Ok(())
    }

    pub fn add_edge(
        &mut self,
        src_port_id: PortID,
        dst_port_id: PortID,
    ) -> Result<EdgeID, AddEdgeError> {
        let src_node = self
            .nodes
            .get(&src_port_id.node)
            .ok_or(AddEdgeError::SrcNodeNotFound(src_port_id.node))?;
        let dst_node = self
            .nodes
            .get(&dst_port_id.node)
            .ok_or(AddEdgeError::DstNodeNotFound(dst_port_id.node))?;

        let src_port = {
            let mut found = None;
            for p in src_node.outputs.iter() {
                if p.id.stable_id == src_port_id.stable_id {
                    found = Some(*p);
                    break;
                }
            }
            found.ok_or(AddEdgeError::SrcPortNotFound(src_port_id))
        }?;
        let dst_port = {
            let mut found = None;
            for p in dst_node.inputs.iter() {
                if p.id.stable_id == dst_port_id.stable_id {
                    found = Some(*p);
                    break;
                }
            }
            found.ok_or(AddEdgeError::DstPortNotFound(dst_port_id))
        }?;

        if src_port.type_idx != dst_port.type_idx {
            return Err(AddEdgeError::TypeMismatch {
                src_port_id,
                src_port_type: src_port.type_idx,
                dst_port_id,
                dst_port_type: dst_port.type_idx,
            });
        }

        let src_node_edges = self.node_edges.get_mut(&src_port_id.node).unwrap();

        for edge in src_node_edges.outgoing.iter() {
            if edge.dst_port == dst_port_id {
                return Err(AddEdgeError::EdgeAlreadyExists(*edge));
            }
        }

        if src_port_id.node == dst_port_id.node {
            return Err(AddEdgeError::CycleDetected);
        }

        // TODO: detect more cycles

        let new_edge_id = EdgeID(self.next_edge_id);
        self.next_edge_id += 1;

        let new_edge = Edge {
            id: new_edge_id,
            src_port: src_port.id,
            dst_port: dst_port.id,
        };

        src_node_edges.outgoing.push(new_edge);
        self.node_edges
            .get_mut(&dst_port_id.node)
            .unwrap()
            .incoming
            .push(new_edge);

        self.edges.insert(new_edge_id, new_edge);

        self.nodes_with_dirty_edges.insert(src_port_id.node);
        self.nodes_with_dirty_edges.insert(dst_port_id.node);

        self.needs_recompile = true;

        Ok(new_edge_id)
    }

    pub fn has_nodes_with_dirty_edges(&self) -> bool {
        !self.nodes_with_dirty_edges.is_empty()
    }

    pub fn nodes_with_dirty_edges(&mut self) -> Vec<NodeID> {
        self.nodes_with_dirty_edges.drain().collect()
    }

    fn remove_edges_with_port(&mut self, port_id: PortID) {
        let mut edges_to_remove: Vec<EdgeID> = Vec::new();

        // Remove all existing edges which have this port.
        for edge in self.edges.values() {
            if edge.src_port == port_id || edge.dst_port == port_id {
                edges_to_remove.push(edge.id);
            }
        }

        for edge_id in edges_to_remove.iter() {
            let edge = self.edges.remove(edge_id).unwrap();

            self.node_edges
                .get_mut(&edge.src_port.node)
                .unwrap()
                .remove_outgoing(edge.id);
            self.node_edges
                .get_mut(&edge.dst_port.node)
                .unwrap()
                .remove_incoming(edge.id);

            self.nodes_with_dirty_edges.insert(edge.src_port.node);
            self.nodes_with_dirty_edges.insert(edge.dst_port.node);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum AddPortError {
    NodeNotFound(NodeID),
    InPortAlreadyExists(PortID),
    OutPortAlreadyExists(PortID),
    TypeIndexOutOfBounds(TypeIdx, usize),
}

impl Error for AddPortError {}

impl fmt::Display for AddPortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeNotFound(node_id) => {
                write!(f, "Could not find node with ID {:?}", node_id)
            }
            Self::InPortAlreadyExists(port_id) => {
                write!(
                    f,
                    "Could not add input port: input port with ID {:?} already exists",
                    port_id
                )
            }
            Self::OutPortAlreadyExists(port_id) => {
                write!(
                    f,
                    "Could not add output port: output port with ID {:?} already exists",
                    port_id
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

#[derive(Debug, Clone, Copy)]
pub enum RemovePortError {
    NodeNotFound(NodeID),
    InPortNotFound(PortID),
    OutPortNotFound(PortID),
}

impl Error for RemovePortError {}

impl fmt::Display for RemovePortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeNotFound(node_id) => {
                write!(f, "Could not find node with ID {:?}", node_id)
            }
            Self::InPortNotFound(port_id) => {
                write!(
                    f,
                    "Could not remove port: input port with ID {:?} was not found",
                    port_id
                )
            }
            Self::OutPortNotFound(port_id) => {
                write!(
                    f,
                    "Could not remove port: output port with ID {:?} was not found",
                    port_id
                )
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum AddEdgeError {
    SrcNodeNotFound(NodeID),
    DstNodeNotFound(NodeID),
    SrcPortNotFound(PortID),
    DstPortNotFound(PortID),
    TypeMismatch {
        src_port_id: PortID,
        src_port_type: TypeIdx,
        dst_port_id: PortID,
        dst_port_type: TypeIdx,
    },
    EdgeAlreadyExists(Edge),
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
            Self::SrcPortNotFound(port_id) => {
                write!(
                    f,
                    "Could not add edge: could not find source port with ID {:?}",
                    port_id
                )
            }
            Self::DstPortNotFound(port_id) => {
                write!(
                    f,
                    "Could not add edge: could not find destination port with ID {:?}",
                    port_id
                )
            }
            Self::TypeMismatch {
                src_port_id,
                src_port_type,
                dst_port_id,
                dst_port_type,
            } => {
                write!(f, "Could not add edge: source port {:?} is of type {:?} but destination port {:?} is of type {:?}",  src_port_id, src_port_type, dst_port_id, dst_port_type)
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
