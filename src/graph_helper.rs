use fnv::{FnvHashMap, FnvHashSet};
use std::error::Error;
use std::fmt;

use crate::{Edge, EdgeID, Node, NodeID, Port, PortID, TypeIdx};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodePortStableID(pub u32);

#[derive(Debug, Clone)]
pub struct NodeState {
    pub inputs: FnvHashMap<NodePortStableID, Port>,
    pub outputs: FnvHashMap<NodePortStableID, Port>,

    pub edges: Vec<NodeEdge>,
}

impl NodeState {
    fn remove_edge(&mut self, edge_id: EdgeID) {
        let mut found_idx = None;
        for (i, edge) in self.edges.iter().enumerate() {
            if edge.id == edge_id {
                found_idx = Some(i);
                break;
            }
        }
        if let Some(i) = found_idx {
            self.edges.remove(i);
        }
    }
}

pub struct AudioGraphHelper {
    nodes: FnvHashMap<NodeID, Node>,
    edges: FnvHashMap<EdgeID, Edge>,

    node_states: FnvHashMap<NodeID, NodeState>,

    next_node_id: u64,
    next_port_id: u64,
    next_edge_id: u64,

    nodes_with_dirty_edges: FnvHashSet<NodeID>,

    num_port_types: usize,
}

impl AudioGraphHelper {
    pub fn new(num_port_types: usize) -> Self {
        assert_ne!(num_port_types, 0);

        Self {
            nodes: FnvHashMap::default(),
            edges: FnvHashMap::default(),
            node_states: FnvHashMap::default(),
            next_node_id: 0,
            next_port_id: 0,
            next_edge_id: 0,
            nodes_with_dirty_edges: FnvHashSet::default(),
            num_port_types,
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

        new_id
    }

    pub fn remove_node(&mut self, node_id: NodeID) -> Result<(), ()> {
        self.node_states.remove(&node_id).ok_or(())?;
        let node = self.nodes.remove(&node_id).unwrap();

        for port in node.inputs.iter().chain(node.outputs.iter()) {
            self.remove_edges_with_port(port.id);
        }

        self.nodes_with_dirty_edges.remove(&node_id);

        Ok(())
    }

    pub fn node_state(&self, node_id: NodeID) -> Option<&NodeState> {
        self.node_states.get(&node_id)
    }

    pub fn add_port_to_node(
        &mut self,
        node_id: NodeID,
        port_stable_id: NodePortStableID,
        port_type: TypeIdx,
        is_input: bool,
    ) -> Result<PortID, AddPortError> {
        if port_type.0 >= self.num_port_types {
            return Err(AddPortError::TypeIndexOutOfBounds(
                port_type,
                self.num_port_types,
            ));
        }

        let node_state = self
            .node_states
            .get_mut(&node_id)
            .ok_or(AddPortError::NodeNotFound(node_id))?;
        let node = self
            .nodes
            .get_mut(&node_id)
            .ok_or(AddPortError::NodeNotFound(node_id))?;

        let new_port_id = PortID(self.next_port_id);
        let new_port = Port {
            id: new_port_id,
            type_idx: port_type,
        };

        if is_input {
            if node_state.inputs.contains_key(&port_stable_id) {
                return Err(AddPortError::InPortAlreadyExists(node_id, port_stable_id));
            }

            node_state.inputs.insert(port_stable_id, new_port);
            node.inputs.push(new_port);
        } else {
            if node_state.outputs.contains_key(&port_stable_id) {
                return Err(AddPortError::OutPortAlreadyExists(node_id, port_stable_id));
            }

            node_state.outputs.insert(port_stable_id, new_port);
            node.outputs.push(new_port);
        }

        self.next_port_id += 1;

        Ok(new_port_id)
    }

    pub fn remove_port_from_node(
        &mut self,
        node_id: NodeID,
        port_stable_id: NodePortStableID,
        is_input: bool,
    ) -> Result<(), RemovePortError> {
        let node_state = self
            .node_states
            .get_mut(&node_id)
            .ok_or(RemovePortError::NodeNotFound(node_id))?;
        let node = self
            .nodes
            .get_mut(&node_id)
            .ok_or(RemovePortError::NodeNotFound(node_id))?;

        let port_id = if is_input {
            if let Some(removed_port) = node_state.inputs.remove(&port_stable_id) {
                let mut found_idx = None;
                for (i, port) in node.inputs.iter().enumerate() {
                    if port.id == removed_port.id {
                        found_idx = Some(i);
                        break;
                    }
                }
                if let Some(i) = found_idx {
                    node.inputs.remove(i);
                }

                removed_port.id
            } else {
                return Err(RemovePortError::InPortNotFound(node_id, port_stable_id));
            }
        } else {
            if let Some(removed_port) = node_state.outputs.remove(&port_stable_id) {
                let mut found_idx = None;
                for (i, port) in node.outputs.iter().enumerate() {
                    if port.id == removed_port.id {
                        found_idx = Some(i);
                        break;
                    }
                }
                if let Some(i) = found_idx {
                    node.outputs.remove(i);
                }

                removed_port.id
            } else {
                return Err(RemovePortError::OutPortNotFound(node_id, port_stable_id));
            }
        };

        self.remove_edges_with_port(port_id);

        Ok(())
    }

    pub fn add_edge(
        &mut self,
        src_node_id: NodeID,
        src_port_id: NodePortStableID,
        dst_node_id: NodeID,
        dst_port_id: NodePortStableID,
    ) -> Result<EdgeID, AddEdgeError> {
        let src_node_state = self
            .node_states
            .get(&src_node_id)
            .ok_or(AddEdgeError::SrcNodeNotFound(src_node_id))?;
        let dst_node_state = self
            .node_states
            .get(&dst_node_id)
            .ok_or(AddEdgeError::DstNodeNotFound(dst_node_id))?;

        let src_port = src_node_state
            .outputs
            .get(&src_port_id)
            .ok_or(AddEdgeError::SrcPortNotFound(src_node_id, src_port_id))?;
        let dst_port = dst_node_state
            .inputs
            .get(&dst_port_id)
            .ok_or(AddEdgeError::DstPortNotFound(dst_node_id, dst_port_id))?;

        for edge in src_node_state.edges.iter() {
            if edge.dst_node == dst_node_id
                && edge.src_port == src_port_id
                && edge.dst_port == dst_port_id
            {
                return Err(AddEdgeError::EdgeAlreadyExists(*edge));
            }
        }

        // TODO: detect cycles

        let new_edge_id = EdgeID(self.next_edge_id);
        self.next_edge_id += 1;

        self.edges.insert(
            new_edge_id,
            Edge {
                id: new_edge_id,
                src_node: src_node_id,
                src_port: src_port.id,
                dst_node: dst_node_id,
                dst_port: dst_port.id,
            },
        );

        let new_edge = NodeEdge {
            id: new_edge_id,
            src_node: src_node_id,
            src_port: src_port_id,
            dst_node: dst_node_id,
            dst_port: dst_port_id,
        };

        self.node_states
            .get_mut(&src_node_id)
            .unwrap()
            .edges
            .push(new_edge);
        self.node_states
            .get_mut(&dst_node_id)
            .unwrap()
            .edges
            .push(new_edge);

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

            self.node_states
                .get_mut(&edge.src_node)
                .unwrap()
                .remove_edge(edge.id);
            self.node_states
                .get_mut(&edge.dst_node)
                .unwrap()
                .remove_edge(edge.id);

            self.nodes_with_dirty_edges.insert(edge.src_node);
            self.nodes_with_dirty_edges.insert(edge.dst_node);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeEdge {
    pub id: EdgeID,
    pub src_node: NodeID,
    pub src_port: NodePortStableID,
    pub dst_node: NodeID,
    pub dst_port: NodePortStableID,
}

#[derive(Debug, Clone, Copy)]
pub enum AddPortError {
    NodeNotFound(NodeID),
    InPortAlreadyExists(NodeID, NodePortStableID),
    OutPortAlreadyExists(NodeID, NodePortStableID),
    TypeIndexOutOfBounds(TypeIdx, usize),
}

impl Error for AddPortError {}

impl fmt::Display for AddPortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeNotFound(node_id) => {
                write!(f, "Could not find node with ID {:?}", node_id)
            }
            Self::InPortAlreadyExists(node_id, stable_id) => {
                write!(f, "Could not add input port to node with ID {:?}: input port with stable ID {:?} already exists", node_id, stable_id)
            }
            Self::OutPortAlreadyExists(node_id, stable_id) => {
                write!(f, "Could not add output port to node with ID {:?}: output port with stable ID {:?} already exists", node_id, stable_id)
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
    InPortNotFound(NodeID, NodePortStableID),
    OutPortNotFound(NodeID, NodePortStableID),
}

impl Error for RemovePortError {}

impl fmt::Display for RemovePortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeNotFound(node_id) => {
                write!(f, "Could not find node with ID {:?}", node_id)
            }
            Self::InPortNotFound(node_id, stable_id) => {
                write!(f, "Could not remove input port from node with ID {:?}: input port with stable ID {:?} was not found", node_id, stable_id)
            }
            Self::OutPortNotFound(node_id, stable_id) => {
                write!(f, "Could not remove output port from node with ID {:?}: output port with stable ID {:?} was not found", node_id, stable_id)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum AddEdgeError {
    SrcNodeNotFound(NodeID),
    DstNodeNotFound(NodeID),
    SrcPortNotFound(NodeID, NodePortStableID),
    DstPortNotFound(NodeID, NodePortStableID),
    EdgeAlreadyExists(NodeEdge),
    CycleDetected,
}

impl Error for AddEdgeError {}

impl fmt::Display for AddEdgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SrcNodeNotFound(node_id) => {
                write!(f, "Could not find source node with ID {:?}", node_id)
            }
            Self::DstNodeNotFound(node_id) => {
                write!(f, "Could not find destination node with ID {:?}", node_id)
            }
            Self::SrcPortNotFound(node_id, stable_id) => {
                write!(f, "Could not find port from source node with ID {:?}: output port with stable ID {:?} was not found", node_id, stable_id)
            }
            Self::DstPortNotFound(node_id, stable_id) => {
                write!(f, "Could not find port from destination node with ID {:?}: input port with stable ID {:?} was not found", node_id, stable_id)
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
