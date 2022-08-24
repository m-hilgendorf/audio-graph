//! The internal [GraphIR] datastructure used by the compiler passes.
//!
use crate::{
    buffer_allocator::{BufferAllocator, BufferRef},
    error::CompileGraphError,
    input_ir::*,
    output_ir::*,
};
use fnv::{FnvHashMap, FnvHashSet};
use std::rc::Rc;

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Internal IR used by the compiler algorithm. Built incrementally
/// via the compiler passes.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct GraphIR {
    /// The number of port types used by the graph.
    num_port_types: usize,
    /// A table of nodes in the graph.
    nodes: FnvHashMap<NodeID, Node>,
    /// Adjacency list table. Built internally.
    adjacent: FnvHashMap<NodeID, AdjacentEdges>,
    /// The topologically sorted schedule of the graph. Built internally.
    schedule: Vec<TempEntry>,
    /// The maximum number of buffers used for each port type. Built internally.
    max_num_buffers: Vec<usize>,
}

/// An entry in the schedule order. Since it is built incrementally, it
/// is not equivalent to a [ScheduledNode].
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub enum TempEntry {
    /// A node in the order that has not been completely scheduled yet.
    Node(Node),
    /// A completely scheduled node
    ScheduledNode(ScheduledNode),
    /// An inserted delay into the the order
    Delay(TempDelay),
    /// An inserted sum point into the order
    Sum(InsertedSum),
}

impl TempEntry {
    fn node(self) -> Node {
        if let Self::Node(node) = self {
            node
        } else {
            unreachable!()
        }
    }
}

/// A delay that has been inserted into the order but has
/// not yet been assigned i/o buffers.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
pub struct TempDelay {
    /// The edge that this delay corresponds to. Kept for debugging and visualization.
    pub edge: Edge,
    /// The amount of delay to apply to the input.
    pub delay: f64,
    /// The input data to read.
    pub input_buffer: Option<BufferAssignment>,
    /// The output buffer to write delayed into to.
    pub output_buffer: Option<BufferAssignment>,
}

/// The edges (port connections) that exist on a given [Node].
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Default, Debug, Clone)]
pub struct AdjacentEdges {
    /// The edges connected to this node's input ports.
    pub incoming: Vec<Edge>,
    /// The edges connected to this node's output ports.
    pub outgoing: Vec<Edge>,
}

/// Main compilation algorithm
pub fn compile<'a>(
    num_port_types: usize,
    nodes: impl IntoIterator<Item = &'a Node>,
    edges: impl IntoIterator<Item = &'a Edge>,
) -> Result<CompiledSchedule, CompileGraphError> {
    Ok(GraphIR::preprocess(num_port_types, nodes, edges)?
        .sort_topologically()?
        .solve_latency_requirements()
        .solve_buffer_requirements()?
        .merge())
}

impl GraphIR {
    /// Construct a [GraphIR] instance from lists of nodes and edges, building
    /// up the adjacency table and creating an empty schedule.
    pub fn preprocess<'a>(
        num_port_types: usize,
        nodes: impl IntoIterator<Item = &'a Node>,
        edges: impl IntoIterator<Item = &'a Edge>,
    ) -> Result<Self, CompileGraphError> {
        let mut nodes_map: FnvHashMap<NodeID, Node> = FnvHashMap::default();
        for node in nodes.into_iter() {
            if nodes_map.insert(node.id, node.clone()).is_some() {
                return Err(CompileGraphError::NodeIDNotUnique(node.id));
            }
        }

        let mut adjacent: FnvHashMap<NodeID, AdjacentEdges> = FnvHashMap::default();
        let mut edge_ids: FnvHashSet<EdgeID> = FnvHashSet::default();
        for edge in edges.into_iter() {
            if !nodes_map.contains_key(&edge.src_node) {
                return Err(CompileGraphError::NodeOnEdgeNotFound(*edge, edge.src_node));
            }
            if !nodes_map.contains_key(&edge.dst_node) {
                return Err(CompileGraphError::NodeOnEdgeNotFound(*edge, edge.dst_node));
            }
            if !edge_ids.insert(edge.id) {
                return Err(CompileGraphError::EdgeIDNotUnique(edge.id));
            }

            let src = adjacent
                .entry(edge.src_node)
                .or_insert_with(AdjacentEdges::default);
            src.outgoing.push(*edge);
            let dst = adjacent
                .entry(edge.dst_node)
                .or_insert_with(AdjacentEdges::default);
            dst.incoming.push(*edge);
        }

        Ok(Self {
            num_port_types,
            nodes: nodes_map,
            adjacent,
            schedule: vec![],
            max_num_buffers: vec![],
        })
    }

    /// Walk the nodes of the graph and add them to the schedule.
    pub fn sort_topologically(mut self) -> Result<Self, CompileGraphError> {
        if self.tarjan() != 0 {
            return Err(CompileGraphError::CycleDetected);
        }

        let mut stack = self.roots().cloned().collect::<Vec<_>>();
        let mut visited = FnvHashSet::default();
        visited.reserve(self.nodes.len());

        self.schedule.clear();

        while let Some(node) = stack.pop() {
            if !visited.contains(&node.id) {
                visited.insert(node.id);
                for next in self.outgoing(&node) {
                    stack.push(next.clone());
                }
                self.schedule.push(TempEntry::Node(node));
            }
        }

        Ok(self)
    }

    pub fn solve_latency_requirements(mut self) -> Self {
        let mut time_of_arrival = FnvHashMap::default();
        let mut new_schedule = Vec::with_capacity(self.schedule.capacity());
        for entry in self.schedule {
            let entry = entry.node(); // cast to a node

            let input_latencies = self.adjacent[&entry.id]
                .incoming
                .iter()
                .map(|edge| {
                    let node = edge.src_node;
                    (edge, time_of_arrival[&node])
                })
                .collect::<Vec<_>>();
            let max_input_latency = input_latencies
                .iter()
                .fold(0.0f64, |acc, lhs| acc.max(lhs.1));
            time_of_arrival.insert(entry.id, max_input_latency + entry.latency);
            let delays = input_latencies.into_iter().filter_map(|(edge, delay)| {
                if delay.abs() > f64::EPSILON {
                    let inserted = TempDelay {
                        delay,
                        edge: *edge,
                        input_buffer: None,
                        output_buffer: None,
                    };
                    Some(inserted)
                } else {
                    None
                }
            });

            for delay in delays {
                new_schedule.push(TempEntry::Delay(delay));
            }
            new_schedule.push(TempEntry::Node(entry));
        }
        self.schedule = new_schedule;
        self
    }

    pub fn solve_buffer_requirements(mut self) -> Result<Self, CompileGraphError> {
        let mut new_schedule = Vec::with_capacity(self.schedule.capacity());
        let mut allocator = BufferAllocator::new(self.num_port_types);
        let mut assignment_table = FnvHashMap::default();

        // hack to get around the borrow checker. This is sound because we do not mutate
        // the schedule internally, but keep the shared state bundled in the same data structure.

        for entry in &self.schedule {
            match entry {
                TempEntry::Node(node) => {
                    let (scheduled, sums) =
                        self.assign_node_buffers(&node, &mut allocator, &mut assignment_table)?;
                    for sum in sums {
                        new_schedule.push(TempEntry::Sum(sum));
                    }
                    new_schedule.push(TempEntry::ScheduledNode(scheduled));
                }
                TempEntry::Delay(delay) => {
                    let delay =
                        self.assign_delay_buffers(*delay, &mut allocator, &mut assignment_table);
                    new_schedule.push(TempEntry::Delay(delay));
                }
                _ => unreachable!(),
            }
        }

        self.schedule = new_schedule;
        self.max_num_buffers = allocator.num_buffers_per_type();
        Ok(self)
    }

    #[allow(unreachable_code)]
    pub fn assign_node_buffers(
        &self,
        node: &Node,
        allocator: &mut BufferAllocator,
        assignment_table: &mut FnvHashMap<EdgeID, Rc<BufferRef>>,
    ) -> Result<(ScheduledNode, impl Iterator<Item = InsertedSum>), CompileGraphError> {
        // Allocate our output data structures, any summing nodes that need to
        // be inserted, the input buffers, and the output buffers.
        let mut summing_nodes = vec![];
        let mut input_buffers = vec![];
        let mut output_buffers = vec![];

        // Collect the inputs to the algorithm, the incoming/outgoing edges of this node.
        let adjacent_edges = &self.adjacent[&node.id];

        let mut buffers_to_release: Vec<Rc<BufferRef>> =
            Vec::with_capacity(node.inputs.len() + node.outputs.len());

        for port in node.inputs.iter() {
            if port.type_idx.0 >= self.num_port_types {
                return Err(CompileGraphError::PortTypeIndexOutOfBounds(
                    node.id,
                    *port,
                    self.num_port_types,
                ));
            }

            let type_index = port.type_idx;
            let edges = adjacent_edges
                .incoming
                .iter()
                .filter(|edge| edge.dst_port == port.id)
                .collect::<Vec<_>>();

            if edges.is_empty() {
                // Case 1: The port is an input and it is unconnected. Acquire a buffer, and
                //         assign it. The buffer must be cleared. Release the buffer once the
                //         node assignments are done.
                let buffer = allocator.acquire(type_index);
                input_buffers.push(BufferAssignment {
                    buffer_index: buffer.idx,
                    generation: buffer.generation,
                    type_index: buffer.type_idx,
                    port_id: port.id,
                    should_clear: true,
                });
                buffers_to_release.push(buffer);
            } else if edges.len() == 1 {
                // Case 2: The port is an input, and has exactly one incoming edge. Lookup the
                //         corresponding buffer and assign it. Buffer should not be cleared.
                //         Release the buffer once the node assignments are done.
                let buffer = assignment_table
                    .remove(&edges[0].id)
                    .expect("No buffer assigned to edge!");
                input_buffers.push(BufferAssignment {
                    buffer_index: buffer.idx,
                    type_index: buffer.type_idx,
                    generation: buffer.generation,
                    port_id: port.id,
                    should_clear: false,
                });
                buffers_to_release.push(buffer);
            } else {
                // Case 4: The port is an input with multiple incoming edges. Compute the
                //         summing point, and assign the input buffer assignment to the output
                //         of the summing point.
                let sum_buffer = allocator.acquire(type_index);
                let sum_output = BufferAssignment {
                    buffer_index: sum_buffer.idx,
                    type_index: sum_buffer.type_idx,
                    generation: sum_buffer.generation,
                    port_id: port.id, // only meaningful to the input port/node.
                    should_clear: false,
                };
                // The sum inputs are the corresponding output buffers of the incoming edges.
                let sum_inputs = edges
                    .iter()
                    .map(|edge| {
                        let buf = assignment_table
                            .remove(&edge.id)
                            .expect("No buffer assigned to edge!");
                        let assignment = BufferAssignment {
                            buffer_index: buf.idx,
                            type_index: buf.type_idx,
                            generation: buf.generation,
                            port_id: edge.src_port,
                            should_clear: false,
                        };
                        allocator.release(buf);
                        assignment
                    })
                    .collect();
                summing_nodes.push(InsertedSum {
                    input_buffers: sum_inputs,
                    output_buffer: sum_output,
                });
                // This node's input buffer is the sum output buffer. Release it once the node
                // assignments are done.
                input_buffers.push(sum_output);
                buffers_to_release.push(sum_buffer);
            }
        }

        for port in node.outputs.iter() {
            if port.type_idx.0 >= self.num_port_types {
                return Err(CompileGraphError::PortTypeIndexOutOfBounds(
                    node.id,
                    *port,
                    self.num_port_types,
                ));
            }

            let type_index = port.type_idx;
            let edges = adjacent_edges
                .outgoing
                .iter()
                .filter(|edge| edge.src_port == port.id)
                .collect::<Vec<_>>();

            if edges.is_empty() {
                // Case 5: The port is an output and it is unconnected. Acquire a buffer and
                //         assign it. The buffer does not need to be cleared. Release the
                //         buffer once the node assignments are done.
                let buffer = allocator.acquire(type_index);
                output_buffers.push(BufferAssignment {
                    buffer_index: buffer.idx,
                    generation: buffer.generation,
                    type_index: buffer.type_idx,
                    port_id: port.id,
                    should_clear: false,
                });
                buffers_to_release.push(buffer);
            } else {
                // Case 6: The port is an output. Acquire a buffer, and add to the assignment
                //         table with any corresponding edge IDs. For each edge, update the
                //         assigned buffer table. Buffer should not be cleared or released.
                let buffer = allocator.acquire(type_index);
                for edge in &edges {
                    assignment_table.insert(edge.id, buffer.clone());
                }
                output_buffers.push(BufferAssignment {
                    buffer_index: buffer.idx,
                    type_index: buffer.type_idx,
                    generation: buffer.generation,
                    port_id: port.id,
                    should_clear: false,
                });
            }
        }

        for buffer in buffers_to_release.drain(..) {
            allocator.release(buffer);
        }

        // Construct the output of the assignment, the scheduled node and any summing nodes we will use.
        let node = ScheduledNode {
            id: node.id,
            latency: node.latency,
            input_buffers,
            output_buffers,
        };

        // Return the result.
        Ok((node, summing_nodes.into_iter()))
    }

    pub fn assign_delay_buffers(
        &self,
        mut delay: TempDelay,
        allocator: &mut BufferAllocator,
        assignment_table: &mut FnvHashMap<EdgeID, Rc<BufferRef>>,
    ) -> TempDelay {
        let input_buffer = assignment_table
            .remove(&delay.edge.id)
            .expect("No buffer assigned to edge!");
        let output_buffer = allocator.acquire(input_buffer.type_idx);

        delay.input_buffer = Some(BufferAssignment {
            buffer_index: input_buffer.idx,
            type_index: input_buffer.type_idx,
            generation: input_buffer.generation,
            port_id: delay.edge.src_port,
            should_clear: false,
        });

        delay.output_buffer = Some(BufferAssignment {
            buffer_index: output_buffer.idx,
            type_index: output_buffer.type_idx,
            generation: output_buffer.generation,
            port_id: delay.edge.dst_port,
            should_clear: false,
        });

        assignment_table.insert(delay.edge.id, output_buffer);
        allocator.release(input_buffer);
        delay
    }

    /// Merge the GraphIR into a [CompiledSchedule].
    ///
    /// Algorithm :
    ///
    /// For each temporary entry:
    ///     - if entry is an unscheduled node, fail.
    ///     - if entry is a delay
    ///         - fail if buffers are unallocated
    ///         - add delay to list of added delays, insert delay to schedule
    ///     - if entry is a sum or scheduled node, add to schedule
    ///
    pub fn merge(self) -> CompiledSchedule {
        debug_assert!(
            self.max_num_buffers.len() == self.num_port_types,
            "Missing buffer allocations in output."
        );

        let mut delays = vec![];
        let mut schedule = vec![];

        for entry in self.schedule {
            let entry = match entry {
                TempEntry::Node(_) => {
                    debug_assert!(false, "Unscheduled node in output.");
                    unreachable!();
                }
                TempEntry::Delay(delay) => {
                    debug_assert!(
                        delay.input_buffer.is_some(),
                        "Unallocated input buffer in scheduled delay."
                    );
                    debug_assert!(
                        delay.output_buffer.is_some(),
                        "Unallocated output buffer in scheduled delay."
                    );
                    let delay = InsertedDelay {
                        edge: delay.edge,
                        delay: delay.delay,
                        input_buffer: delay.input_buffer.unwrap(),
                        output_buffer: delay.output_buffer.unwrap(),
                    };
                    delays.push(delay);
                    ScheduleEntry::Delay(delay)
                }
                TempEntry::ScheduledNode(node) => ScheduleEntry::Node(node),
                TempEntry::Sum(sum) => ScheduleEntry::Sum(sum),
            };
            schedule.push(entry);
        }

        CompiledSchedule {
            schedule,
            delays,
            num_buffers: self.max_num_buffers,
        }
    }

    /// List the adjacent nodes along outgoing edges of `n`.
    pub fn outgoing<'b>(&'b self, n: &'b Node) -> impl Iterator<Item = &'b Node> + 'b {
        self.adjacent[&n.id]
            .outgoing
            .iter()
            .map(move |e| &self.nodes[&e.dst_node])
    }

    /// List the adjacent nodes along incoming edges of `n`.
    pub fn incoming<'b>(&'b self, n: &'b Node) -> impl Iterator<Item = &'b Node> + 'b {
        self.adjacent[&n.id]
            .incoming
            .iter()
            .map(move |e| &self.nodes[&e.src_node])
    }

    /// List root nodes, or nodes which have indegree of 0.
    pub fn roots(&self) -> impl Iterator<Item = &Node> + '_ {
        self.nodes
            .values()
            .filter(move |n| self.incoming(*n).next().is_none())
    }

    /// List the sink nodes, or nodes which have outdegree of 0.
    pub fn sinks(&self) -> impl Iterator<Item = &Node> + '_ {
        self.nodes
            .values()
            .filter(move |n| self.outgoing(*n).next().is_none())
    }

    /// Consume the GraphIR returning a new instance with an updated schedule.
    pub fn with_schedule(mut self, i: impl IntoIterator<Item = TempEntry>) -> Self {
        self.schedule = i.into_iter().collect();
        self
    }

    /// Count the number of cycles in the graph using Tarjan's algorithm for
    /// strongly connected components.
    pub fn tarjan(&self) -> usize {
        let mut index = 0;
        let mut stack = Vec::with_capacity(self.nodes.len());
        let mut aux: FnvHashMap<NodeID, TarjanData> = self
            .nodes
            .iter()
            .map(|(k, _)| (*k, TarjanData::default()))
            .collect();

        let mut num_cycles = 0;
        fn strong_connect<'a>(
            graph: &'a GraphIR,
            aux: &mut FnvHashMap<NodeID, TarjanData>,
            node: &'a Node,
            index: &mut u64,
            stack: &mut Vec<&'a Node>,
            outgoing: impl Iterator<Item = &'a Node> + 'a,
            num_cycles: &mut usize,
        ) {
            aux.get_mut(&node.id).unwrap().index = Some(*index);
            aux.get_mut(&node.id).unwrap().low_link = *index;
            aux.get_mut(&node.id).unwrap().on_stack = true;
            stack.push(node);
            *index += 1;

            for next in outgoing {
                if aux[&next.id].index.is_none() {
                    strong_connect(
                        graph,
                        aux,
                        next,
                        index,
                        stack,
                        graph.outgoing(next),
                        num_cycles,
                    );
                    aux.get_mut(&node.id).unwrap().low_link =
                        aux[&node.id].low_link.min(aux[&next.id].low_link);
                } else if aux[&next.id].on_stack {
                    aux.get_mut(&node.id).unwrap().low_link =
                        aux[&node.id].low_link.min(aux[&next.id].index.unwrap());
                }
            }

            if aux[&node.id].index.unwrap() == aux[&node.id].low_link {
                let mut scc_count = 0;
                loop {
                    if let Some(scc) = stack.pop() {
                        if scc.id == node.id {
                            break;
                        } else {
                            scc_count += 1;
                        }
                    }
                }
                if scc_count != 0 {
                    *num_cycles += 1;
                }
            }
        }

        for (_, node) in self.nodes.iter() {
            strong_connect(
                self,
                &mut aux,
                node,
                &mut index,
                &mut stack,
                self.outgoing(node),
                &mut num_cycles,
            );
        }

        num_cycles
    }
}

#[derive(Default)]
struct TarjanData {
    index: Option<u64>,
    on_stack: bool,
    low_link: u64,
}
