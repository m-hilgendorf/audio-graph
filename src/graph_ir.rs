//! The internal [GraphIR] datastructure used by the compiler passes.
//!
use crate::{input_ir::*, output_ir::*};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Internal IR used by the compiler algorithm. Built incrementally
/// via the compiler passes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphIR {
    /// The number of port types used by the graph.
    pub num_port_types: usize,
    /// A table of nodes in the graph.
    pub nodes: HashMap<u64, Node>,
    /// A list of edges in the graph.
    pub edges: HashMap<u64, Edge>,
    /// Adjacency list table. Built internally.
    pub adjacent: HashMap<u64, Vec<Edge>>,
    /// The topologically sorted schedule of the graph. Built internally.
    pub schedule: Vec<TempEntry>,
    /// The maximum number of buffers used for each port type. Built internally.
    pub max_num_buffers: Vec<usize>,
}

/// An entry in the schedule order. Since it is built incrementally, it
/// is not equivalent to a [ScheduledNode].
#[derive(Clone, Debug, Serialize, Deserialize)]
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

/// A delay that has been inserted into the order but has
/// not yet been assigned i/o buffers.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
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

/// Main compilation algorithm
pub fn compile(
    num_port_types: usize,
    nodes: impl IntoIterator<Item = Node>,
    edges: impl IntoIterator<Item = Edge>,
) -> CompiledSchedule {
    GraphIR::preprocess(num_port_types, nodes, edges)
        .sort_topologically()
        .solve_latency_requirements()
        .solve_buffer_requirements()
        .merge()
}

impl GraphIR {
    /// Construct a [GraphIR] instance from lists of nodes and edges, building
    /// up the adjacency table and creating an empty schedule.
    pub fn preprocess(
        num_port_types: usize,
        nodes: impl IntoIterator<Item = Node>,
        edges: impl IntoIterator<Item = Edge>,
    ) -> Self {
        let nodes = nodes.into_iter().map(|n| (n.id, n)).collect();
        let edges: HashMap<u64, Edge> = edges.into_iter().map(|e| (e.id, e)).collect();
        let mut adjacent = HashMap::new();
        for edge in edges.values() {
            let src = adjacent.entry(edge.src_node).or_insert_with(Vec::new);
            src.push(*edge);
            let dst = adjacent.entry(edge.dst_node).or_insert_with(Vec::new);
            dst.push(*edge);
        }
        Self {
            num_port_types,
            nodes,
            edges,
            adjacent,
            schedule: vec![],
            max_num_buffers: vec![],
        }
    }

    /// Walk the nodes of the graph and add them to the schedule.
    pub fn sort_topologically(mut self) -> Self {
        debug_assert!(self.tarjan() == 0, "Graph contains cycles.");
        let mut stack = self.roots().cloned().collect::<Vec<_>>();
        let mut visited = HashSet::with_capacity(self.nodes.len());

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

        self
    }

    pub fn solve_latency_requirements(mut self) -> Self {
        let mut time_of_arrival = HashMap::new();
        let mut new_schedule = Vec::with_capacity(self.schedule.capacity());
        for entry in self.schedule {
            let entry = if let TempEntry::Node(n) = entry {
                n
            } else {
                unreachable!()
            };
            let incoming_edges = self.adjacent[&entry.id]
                .iter()
                .filter(|edge| edge.dst_node == entry.id);
            let input_latencies = incoming_edges
                .map(|edge| {
                    let node = edge.src_node;
                    (edge, time_of_arrival[&node])
                })
                .collect::<Vec<_>>();
            let max_input_latency = input_latencies
                .iter()
                .fold(f64::MIN, |acc, lhs| acc.max(lhs.1));
            time_of_arrival.insert(entry.id, max_input_latency + entry.latency);
            let delays = input_latencies.into_iter().filter_map(|(edge, delay)| {
                if delay != 0.0 {
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

    pub fn solve_buffer_requirements(mut self) -> Self {
        let mut new_schedule = Vec::with_capacity(self.schedule.capacity());
        for entry in &self.schedule {
            match entry {
                TempEntry::Node(node) => {
                    let (scheduled, sums) = self.assign_node_buffers(node);
                    for sum in sums {
                        new_schedule.push(TempEntry::Sum(sum));
                    }
                    new_schedule.push(TempEntry::ScheduledNode(scheduled));
                }
                TempEntry::Delay(delay) => {
                    let delay = self.assign_delay_buffers(*delay);
                    new_schedule.push(TempEntry::Delay(delay));
                }
                _ => unreachable!(),
            }
        }
        self.schedule = new_schedule;
        self
    }

    #[allow(unreachable_code)]
    pub fn assign_node_buffers(
        &self,
        _node: &Node,
    ) -> (ScheduledNode, impl Iterator<Item = InsertedSum>) {
        (todo!(), std::iter::from_fn(|| None))
    }

    pub fn assign_delay_buffers(&self, _delay: TempDelay) -> TempDelay {
        todo!()
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
    pub fn outgoing<'a>(&'a self, n: &'a Node) -> impl Iterator<Item = &'a Node> + 'a {
        self.adjacent[&n.id].iter().filter_map(move |e| {
            if e.src_node == n.id {
                Some(&self.nodes[&e.src_node])
            } else {
                None
            }
        })
    }

    /// List the adjacent nodes along incoming edges of `n`.
    pub fn incoming<'a>(&'a self, n: &'a Node) -> impl Iterator<Item = &'a Node> + 'a {
        self.adjacent[&n.id].iter().filter_map(move |e| {
            if e.dst_node == n.id {
                Some(&self.nodes[&e.src_node])
            } else {
                None
            }
        })
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
        let mut aux: HashMap<u64, TarjanData> = self
            .nodes
            .iter()
            .map(|(k, _)| (*k, TarjanData::default()))
            .collect();

        let mut num_cycles = 0;
        fn strong_connect<'a>(
            graph: &'a GraphIR,
            aux: &mut HashMap<u64, TarjanData>,
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
