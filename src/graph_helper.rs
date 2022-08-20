//! Structs that help construct and modify audio graphs.

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

use fnv::{FnvHashMap, FnvHashSet};

use crate::error::{AddEdgeError, AddPortError, RemovePortError};
use crate::{CompiledSchedule, Edge, EdgeID, GraphIR, Node, NodeID, Port, PortID, TypeIdx};

/// A helper struct to construct and modify audio graphs.
pub struct AudioGraphHelper {
    nodes: FnvHashMap<NodeID, Node>,
    edges: FnvHashMap<EdgeID, Edge>,

    node_edges: FnvHashMap<NodeID, NodeEdges>,

    next_node_id: u64,
    next_edge_id: u64,

    nodes_with_dirty_edges: FnvHashSet<NodeID>,
    needs_compile: bool,

    num_port_types: usize,
}

impl AudioGraphHelper {
    /// Construct a new [AudioGraphHelper].
    ///
    /// * `num_port_types` - The total number of port types that can
    /// exist in this audio graph. For example, if your graph can have
    /// an audio port type and an event port type, then this should be
    /// `2`. Ports of different types cannot be connected together.
    ///
    /// ## Panics
    ///
    /// This will panic if `num_port_types == 0`.
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
            needs_compile: false,
        }
    }

    /// Add a new [Node] the the audio graph.
    ///
    /// This will return the globally unique ID assigned to this node.
    pub fn add_new_node(&mut self, latency: u64) -> NodeID {
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

        self.needs_compile = true;

        new_id
    }

    /// Get info about a node.
    ///
    /// This will return `None` if a node with the given ID does not
    /// exist in the graph.
    pub fn get_node(&self, node_id: NodeID) -> Option<&Node> {
        self.nodes.get(&node_id)
    }

    /// Set the latency of the given [Node] in the audio graph.
    ///
    /// This will return an error if a node with the given ID does not
    /// exist in the graph.
    pub fn set_node_latency(&mut self, node_id: NodeID, latency: u64) -> Result<(), ()> {
        let node = self.nodes.get_mut(&node_id).ok_or(())?;

        if node.latency != latency {
            node.latency = latency;
            self.needs_compile = true;
        }

        Ok(())
    }

    /// Remove the given node from the graph.
    ///
    /// This will automatically remove all edges from the graph that
    /// were connected to this node.
    ///
    /// This will return an error if a node with the given ID does not
    /// exist in the graph.
    pub fn remove_node(&mut self, node_id: NodeID) -> Result<(), ()> {
        let node = self.nodes.remove(&node_id).ok_or(())?;
        self.node_edges.remove(&node_id).unwrap();

        for port in node.inputs.iter().chain(node.outputs.iter()) {
            self.remove_edges_with_port(node_id, port.id);
        }

        self.nodes_with_dirty_edges.remove(&node_id);

        self.needs_compile = true;

        Ok(())
    }

    /// Get the edges (port connections) that are currently connected
    /// to the given node.
    ///
    /// This will return `None` if a node with the given ID does not
    /// exist in the graph.
    pub fn node_edges(&self, node_id: NodeID) -> Option<&NodeEdges> {
        self.node_edges.get(&node_id)
    }

    /// Add a new [Port] to the graph.
    ///
    /// * `node_id` - The ID of the [Node] this port will be added to.
    /// * `port_id` - The identifier for this port. This does not need
    /// to be a globally unique identifier, just unique to the [Node]
    /// it belongs to.
    /// * `type_idx` - The type index of this port. This must be less
    /// than the value of `num_port_types` set in the constructor
    /// of this helper struct. Ports of different types cannot be
    /// connected to eachother.
    /// * `is_input` - `true` if this is an input port, `false` if this
    /// is an output port.
    ///
    /// If this returns an error, then the audio graph has not been
    /// modified.
    pub fn add_port(
        &mut self,
        node_id: NodeID,
        port_id: PortID,
        type_idx: TypeIdx,
        is_input: bool,
    ) -> Result<(), AddPortError> {
        if type_idx.0 >= self.num_port_types {
            return Err(AddPortError::TypeIndexOutOfBounds(
                type_idx,
                self.num_port_types,
            ));
        }

        let node = self
            .nodes
            .get_mut(&node_id)
            .ok_or(AddPortError::NodeNotFound(node_id))?;

        let new_port = Port {
            id: port_id,
            type_idx,
        };

        for p in node.inputs.iter().chain(node.outputs.iter()) {
            if p.id == port_id {
                return Err(AddPortError::PortAlreadyExists(node_id, port_id));
            }
        }

        if is_input {
            node.inputs.push(new_port);
        } else {
            node.outputs.push(new_port);
        }

        self.needs_compile = true;

        Ok(())
    }

    /// Remove the given port from the graph.
    ///
    /// This will automatically remove all edges from the graph that
    /// were connected to this port.
    ///
    /// * `node_id` - The ID of the node which the port belongs to.
    /// * `port_id` - The ID of the port to remove.
    /// * `is_input` - `true` if this is an input port, `false` if this
    /// is an output port.
    ///
    /// If this returns an error, then the audio graph has not been
    /// modified.
    pub fn remove_port(
        &mut self,
        node_id: NodeID,
        port_id: PortID,
        is_input: bool,
    ) -> Result<(), RemovePortError> {
        let node = self
            .nodes
            .get_mut(&node_id)
            .ok_or(RemovePortError::NodeNotFound(node_id))?;

        if is_input {
            let mut found = None;
            for (i, p) in node.inputs.iter().enumerate() {
                if p.id == port_id {
                    found = Some(i);
                    break;
                }
            }
            if let Some(i) = found {
                node.inputs.remove(i);
            } else {
                return Err(RemovePortError::InPortNotFound(node_id, port_id));
            }
        } else {
            let mut found = None;
            for (i, p) in node.outputs.iter().enumerate() {
                if p.id == port_id {
                    found = Some(i);
                    break;
                }
            }
            if let Some(i) = found {
                node.outputs.remove(i);
            } else {
                return Err(RemovePortError::OutPortNotFound(node_id, port_id));
            }
        };

        self.remove_edges_with_port(node_id, port_id);

        self.needs_compile = true;

        Ok(())
    }

    /// Add an [Edge] (port connection) to the graph.
    ///
    /// * `src_port_id` - The ID of the source port. This must be an output
    /// port on a node.
    /// * `src_port_id` - The ID of the destination port. This must be an
    /// input port on a node.
    ///
    /// If successful, this returns the globally unique identifier assigned
    /// to this edge.
    ///
    /// If this returns an error, then the audio graph has not been
    /// modified.
    pub fn add_edge(
        &mut self,
        src_node_id: NodeID,
        src_port_id: PortID,
        dst_node_id: NodeID,
        dst_port_id: PortID,
    ) -> Result<EdgeID, AddEdgeError> {
        let src_node = self
            .nodes
            .get(&src_node_id)
            .ok_or(AddEdgeError::SrcNodeNotFound(src_node_id))?;
        let dst_node = self
            .nodes
            .get(&dst_node_id)
            .ok_or(AddEdgeError::DstNodeNotFound(dst_node_id))?;

        let src_port = {
            let mut found = None;
            for p in src_node.outputs.iter() {
                if p.id == src_port_id {
                    found = Some(*p);
                    break;
                }
            }
            found.ok_or(AddEdgeError::SrcPortNotFound(src_node_id, src_port_id))
        }?;
        let dst_port = {
            let mut found = None;
            for p in dst_node.inputs.iter() {
                if p.id == dst_port_id {
                    found = Some(*p);
                    break;
                }
            }
            found.ok_or(AddEdgeError::DstPortNotFound(dst_node_id, dst_port_id))
        }?;

        if src_port.type_idx != dst_port.type_idx {
            return Err(AddEdgeError::TypeMismatch {
                src_node_id,
                src_port,
                dst_node_id,
                dst_port,
            });
        }

        let src_node_edges = self.node_edges.get_mut(&src_node_id).unwrap();

        for edge in src_node_edges.outgoing.iter() {
            if edge.dst_port == dst_port_id {
                return Err(AddEdgeError::EdgeAlreadyExists(*edge));
            }
        }

        if src_node_id == dst_node_id {
            return Err(AddEdgeError::CycleDetected);
        }

        let new_edge_id = EdgeID(self.next_edge_id);
        self.next_edge_id += 1;

        let new_edge = Edge {
            id: new_edge_id,
            src_node: src_node_id,
            src_port: src_port.id,
            dst_node: dst_node_id,
            dst_port: dst_port.id,
        };

        src_node_edges.outgoing.push(new_edge);
        self.node_edges
            .get_mut(&dst_node_id)
            .unwrap()
            .incoming
            .push(new_edge);

        self.edges.insert(new_edge_id, new_edge);

        if self.cycle_detected() {
            self.node_edges
                .get_mut(&src_node_id)
                .unwrap()
                .remove_outgoing(new_edge_id);
            self.node_edges
                .get_mut(&dst_node_id)
                .unwrap()
                .remove_incoming(new_edge_id);

            self.edges.remove(&new_edge_id);

            return Err(AddEdgeError::CycleDetected);
        }

        self.nodes_with_dirty_edges.insert(src_node_id);
        self.nodes_with_dirty_edges.insert(dst_node_id);

        self.needs_compile = true;

        Ok(new_edge_id)
    }

    /// Remove the given [Edge] (port connection) from the graph.
    ///
    /// This will return an error if the given edge does not exist in
    /// the graph. In this case the graph has not been modified.
    pub fn remove_edge(&mut self, edge_id: EdgeID) -> Result<(), ()> {
        let edge = self.edges.remove(&edge_id).ok_or(())?;

        self.node_edges
            .get_mut(&edge.src_node)
            .unwrap()
            .remove_outgoing(edge_id);
        self.node_edges
            .get_mut(&edge.dst_node)
            .unwrap()
            .remove_incoming(edge_id);

        self.nodes_with_dirty_edges.insert(edge.src_node);
        self.nodes_with_dirty_edges.insert(edge.dst_node);

        self.needs_compile = true;

        Ok(())
    }

    /// Compile the graph into a schedule.
    pub fn compile(&mut self) -> CompiledSchedule {
        self.needs_compile = false;

        // TODO: Make this more efficient by not constructing a new
        // `GraphIR` every time?

        GraphIR::preprocess(self.num_port_types, self.nodes.clone(), self.edges.clone())
            .sort_topologically()
            .solve_latency_requirements()
            .solve_buffer_requirements()
            .merge()
    }

    /// Returns `true` if `AudioGraphHelper::compile()` should be called
    /// again because the state of the graph has changed since the last
    /// compile.
    pub fn needs_compile(&self) -> bool {
        self.needs_compile
    }

    /// Returns `true` if any nodes have had the state of their edges
    /// (port connections) changed since the last call to
    /// `AudioGraphHelper::nodes_with_dirty_edges()`.
    ///
    /// If `true`, then you can call
    /// `AudioGraphHelper::nodes_with_dirty_edges()` to get the list of
    /// these nodes, and then call `AudioGraphHelper::node_edges()` on
    /// each one to retrieve its edges.
    pub fn has_nodes_with_dirty_edges(&self) -> bool {
        !self.nodes_with_dirty_edges.is_empty()
    }

    /// Returns a list of all nodes that have had the state of their
    /// edges (port connections) changed since the last call to this
    /// method.
    ///
    /// Call `AudioGraphHelper::node_edges()` on each node to retrieve
    /// its edges.
    pub fn nodes_with_dirty_edges(&mut self) -> Vec<NodeID> {
        self.nodes_with_dirty_edges.drain().collect()
    }

    /// The total number of port types that can exist in this audio
    /// graph. For example, if your graph can have an audio port type
    /// and an event port type, then this should be `2`.
    ///
    /// Ports of different types cannot be connected together.
    pub fn num_port_types(&self) -> usize {
        self.num_port_types
    }

    fn remove_edges_with_port(&mut self, node_id: NodeID, port_id: PortID) {
        let mut edges_to_remove: Vec<EdgeID> = Vec::new();

        // Remove all existing edges which have this port.
        for edge in self.edges.values() {
            if (edge.src_node == node_id && edge.src_port == port_id)
                || (edge.dst_node == node_id && edge.dst_port == port_id)
            {
                edges_to_remove.push(edge.id);
            }
        }

        for edge_id in edges_to_remove.iter() {
            let edge = self.edges.remove(edge_id).unwrap();

            self.node_edges
                .get_mut(&edge.src_node)
                .unwrap()
                .remove_outgoing(edge.id);
            self.node_edges
                .get_mut(&edge.dst_node)
                .unwrap()
                .remove_incoming(edge.id);

            self.nodes_with_dirty_edges.insert(edge.src_node);
            self.nodes_with_dirty_edges.insert(edge.dst_node);
        }
    }

    fn cycle_detected(&self) -> bool {
        // TODO: Make this more efficient by not constructing a new
        // `GraphIR` every time.

        let graph_ir =
            GraphIR::preprocess(self.num_port_types, self.nodes.clone(), self.edges.clone());
        graph_ir.tarjan() > 0
    }
}

/// The edges (port connections) that exist on a given [Node].
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct NodeEdges {
    /// The edges connected to this node's input ports.
    pub incoming: Vec<Edge>,
    /// The edges connected to this node's output ports.
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
