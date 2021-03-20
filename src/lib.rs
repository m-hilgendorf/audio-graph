pub mod graph2;

use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::atomic::{AtomicU64, Ordering},
};

#[derive(Debug)]
pub enum Error {
    NodeDoesNotExist,
    PortDoesNotExist,
    Cycle,
    ConnectionDoesNotExist,
    RefDoesNotExist,
    InvalidPortType,
}

fn unique() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    COUNTER.fetch_add(1, Ordering::SeqCst)
}

#[derive(Clone, Copy, Debug)]
struct PortInfo {
    type_: PortType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PortType {
    Event,
    Audio,
}

#[derive(Clone, Debug)]
struct NodeInfo {
    id: u64,
    ports: Vec<(u64, PortInfo)>,
    connections: Vec<ConnectionInfo>,
    delay: u64,
    latency: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ConnectionInfo {
    src: (NodeRef, PortRef),
    dst: (NodeRef, PortRef),
    type_: PortType,
}

// impl ConnectionInfo {
//     fn is_incoming(&self, node: &NodeInfo) -> bool {
//         self.dst.0 == NodeRef(node.id)
//     }
//     fn is_outgoing(&self, node: &NodeInfo) -> bool {
//         self.src.0 == NodeRef(node.id)
//     }
// }

impl NodeInfo {
    fn incoming(&self) -> impl Iterator<Item = ConnectionInfo> + '_ {
        self.connections
            .iter()
            .filter(move |c| c.dst.0 .0 == self.id)
            .copied()
    }
    fn dependencies(&self) -> impl Iterator<Item = NodeRef> + '_ {
        self.connections.iter().filter_map(move |c| {
            if c.dst.0 .0 == self.id {
                Some(c.src.0)
            } else {
                None
            }
        })
    }
    fn dependents(&self) -> impl Iterator<Item = NodeRef> + '_ {
        self.connections.iter().filter_map(move |c| {
            if c.src.0 .0 == self.id {
                Some(c.dst.0)
            } else {
                None
            }
        })
    }
}

#[derive(Default)]
pub struct Graph {
    nodes: HashMap<u64, NodeInfo>,
    names: HashMap<u64, String>,
    ports_to_nodes: HashMap<u64, u64>,
}

struct BufferAllocator {
    event_buffer_count: u64,
    audio_buffer_count: u64,
    event_buffer_stack: Vec<u64>,
    audio_buffer_stack: Vec<u64>,
}

pub trait Ref
where
    Self: Into<u64>,
{
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferRef(u64, PortType);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeRef(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PortRef(u64);

impl Into<u64> for BufferRef {
    fn into(self) -> u64 {
        self.0
    }
}

impl Into<u64> for PortRef {
    fn into(self) -> u64 {
        self.0
    }
}

impl Into<u64> for NodeRef {
    fn into(self) -> u64 {
        self.0
    }
}

impl Ref for NodeRef {}
impl Ref for PortRef {}
impl Ref for BufferRef {}

impl BufferAllocator {
    fn new() -> Self {
        Self {
            event_buffer_count: 0,
            audio_buffer_count: 0,
            event_buffer_stack: vec![],
            audio_buffer_stack: vec![],
        }
    }

    fn acquire(&mut self, type_: PortType) -> BufferRef {
        let (count, stack) = match type_ {
            PortType::Event => (&mut self.event_buffer_count, &mut self.event_buffer_stack),
            PortType::Audio => (&mut self.audio_buffer_count, &mut self.audio_buffer_stack),
        };
        if let Some(thing) = stack.pop() {
            BufferRef(thing, type_)
        } else {
            let bref = BufferRef(*count, type_);
            *count += 1;
            bref
        }
    }

    fn release(&mut self, ref_: BufferRef) {
        let stack = match ref_.1 {
            PortType::Event => &mut self.event_buffer_stack,
            PortType::Audio => &mut self.audio_buffer_stack,
        };
        stack.push(ref_.0);
    }
}

impl Graph {
    /// Get the name of something in the graph.
    pub fn name<R: Ref>(&self, r: R) -> Option<&'_ str> {
        let id = r.into();
        self.names.get(&id).map(|s| s.as_str())
    }

    /// Delete something from the graph, such as a node or port. Returns an error if the thing did not exist or is not owned by the graph.
    pub fn delete<R: Ref>(&mut self, r: R) -> Result<(), Error> {
        let thing: u64 = r.into();
        if let Some(node) = self.nodes.remove(&thing) {
            let NodeInfo {
                id,
                ports,
                connections,
                ..
            } = node;
            for (p, _) in ports {
                let _ = self.ports_to_nodes.remove(&p);
            }
            for c in connections {
                let neighbor = if c.src.0 == NodeRef(id) {
                    c.dst.0 .0
                } else {
                    c.src.0 .0
                };
                if let Some(neighbor) = self.nodes.get_mut(&neighbor) {
                    if let Some(idx) = neighbor.connections.iter().position(|c_| *c_ == c) {
                        neighbor.connections.remove(idx);
                    }
                }
            }
            Ok(())
        } else if let Some(node) = self.ports_to_nodes.remove(&thing) {
            let connections = self
                .nodes
                .get_mut(&node)
                .unwrap()
                .connections
                .iter()
                .filter(|c| c.src.1 == PortRef(thing) || c.dst.1 == PortRef(thing))
                .copied()
                .collect::<Vec<_>>();
            for c in connections {
                let ConnectionInfo { src, dst, .. } = c;
                let _r = self.disconnect(src.1, dst.1);
                debug_assert!(_r.is_ok());
            }
            let parent_node = self.nodes.get_mut(&node).unwrap();
            let idx = parent_node
                .ports
                .iter()
                .position(|(p, _)| *p == thing)
                .unwrap();
            parent_node.ports.remove(idx);
            Ok(())
        } else {
            Err(Error::RefDoesNotExist)
        }
    }

    /// Create a new node in the graph.
    pub fn node(&mut self, name: &str) -> NodeRef {
        let id = unique();
        let _old_name = self.names.insert(id, name.to_owned());
        debug_assert!(_old_name.is_none());
        let info = NodeInfo {
            id,
            ports: vec![],
            connections: vec![],
            delay: 0,
            latency: None,
        };
        self.nodes.insert(id, info);
        NodeRef(id)
    }

    /// Create a new port for a node    
    pub fn port(
        &mut self,
        node: NodeRef,
        port_name: &str,
        port_type: PortType,
    ) -> Result<PortRef, Error> {
        let node = self.nodes.get_mut(&node.0).ok_or(Error::NodeDoesNotExist)?;
        let id = unique();
        let _old_name = self.names.insert(id, port_name.to_owned());
        debug_assert!(_old_name.is_none());
        let info = PortInfo { type_: port_type };
        node.ports.push((id, info));
        self.ports_to_nodes.insert(id, node.id);
        Ok(PortRef(id))
    }

    /// Update the delay of a node in the graph. Will invalidate any latencies internally.
    pub fn set_delay(&mut self, node: NodeRef, delay: u64) -> Result<(), Error> {
        let node = self.nodes.get_mut(&node.0).ok_or(Error::NodeDoesNotExist)?;
        node.delay = delay;
        let mut stack = vec![node.id];
        while let Some(next) = stack.pop() {
            let node = self.nodes.get_mut(&next).unwrap();
            for n in node.dependents() {
                stack.push(n.0);
            }
            node.latency = None;
        }
        Ok(())
    }

    /// Connect two ports. Returns an error if they do not exist or if their types
    /// do not match.
    pub fn connect(&mut self, src: PortRef, dst: PortRef) -> Result<(), Error> {
        let src_node_id = *self
            .ports_to_nodes
            .get(&src.0)
            .ok_or(Error::PortDoesNotExist)?;
        let dst_node_id = *self
            .ports_to_nodes
            .get(&dst.0)
            .ok_or(Error::PortDoesNotExist)?;
        fn cycle_check(graph: &Graph, src: u64, check: u64) -> Result<(), Error> {
            if src == check {
                return Err(Error::Cycle);
            }
            for node in graph
                .nodes
                .get(&src)
                .ok_or(Error::NodeDoesNotExist)?
                .dependents()
            {
                cycle_check(graph, node.0, check)?;
            }
            Ok(())
        }
        let src_node = self
            .nodes
            .get(&src_node_id)
            .ok_or(Error::NodeDoesNotExist)?;
        let dst_node = self
            .nodes
            .get(&dst_node_id)
            .ok_or(Error::NodeDoesNotExist)?;
        cycle_check(self, dst_node_id, src_node_id)?;
        let (_, src_port) = src_node
            .ports
            .iter()
            .find(|(id, _)| *id == src.0)
            .ok_or(Error::PortDoesNotExist)?;
        let (_, dst_port) = dst_node
            .ports
            .iter()
            .find(|(id, _)| *id == dst.0)
            .ok_or(Error::PortDoesNotExist)?;
        if src_port.type_ != dst_port.type_ {
            return Err(Error::InvalidPortType);
        }
        let connection = ConnectionInfo {
            src: (NodeRef(src_node_id), src),
            dst: (NodeRef(dst_node_id), dst),
            type_: src_port.type_,
        };
        self.nodes
            .get_mut(&src_node_id)
            .unwrap()
            .connections
            .push(connection);
        self.nodes
            .get_mut(&dst_node_id)
            .unwrap()
            .connections
            .push(connection);
        Ok(())
    }

    // Internal. Remove an edge from the graph.
    fn disconnect_edge(&mut self, edge: ConnectionInfo) -> Result<(), Error> {
        let (src_node, src_port) = edge.src;
        let (dst_node, dst_port) = edge.dst;
        let src_index = self
            .nodes
            .get(&src_node.0)
            .ok_or(Error::NodeDoesNotExist)?
            .connections
            .iter()
            .position(|c| c.src.1 == src_port)
            .ok_or(Error::ConnectionDoesNotExist)?;
        let dst_index = self
            .nodes
            .get(&dst_node.0)
            .ok_or(Error::NodeDoesNotExist)?
            .connections
            .iter()
            .position(|c| c.dst.1 == dst_port)
            .ok_or(Error::ConnectionDoesNotExist)?;
        self.nodes
            .get_mut(&src_node.0)
            .unwrap()
            .connections
            .remove(src_index);
        self.nodes
            .get_mut(&dst_node.0)
            .unwrap()
            .connections
            .remove(dst_index);
        Ok(())
    }

    /// Disconnect two ports, returning an error if they were not connected or did not exist.
    pub fn disconnect(&mut self, src: PortRef, dst: PortRef) -> Result<(), Error> {
        let src_node_id = *self
            .ports_to_nodes
            .get(&src.0)
            .ok_or(Error::PortDoesNotExist)?;
        let dst_node_id = *self
            .ports_to_nodes
            .get(&dst.0)
            .ok_or(Error::PortDoesNotExist)?;
        let src_node = self
            .nodes
            .get_mut(&src_node_id)
            .ok_or(Error::NodeDoesNotExist)?;
        let info = src_node
            .ports
            .iter()
            .find_map(|(id, info)| if *id == src.0 { Some(*info) } else { None })
            .ok_or(Error::PortDoesNotExist)?;
        let connection = ConnectionInfo {
            src: (NodeRef(src_node_id), src),
            dst: (NodeRef(dst_node_id), dst),
            type_: info.type_,
        };
        self.disconnect_edge(connection)
    }

    pub fn compile(&mut self, root: NodeRef, state: &mut State) -> Vec<ScheduleEntry> {
        // Once all our work is done, add a node to the schedule.
        #[inline]
        fn add_to_schedule(
            graph: &Graph,
            node: u64,
            schedule: &mut Vec<ScheduleEntry>,
            assignments: &HashMap<ConnectionInfo, (BufferRef, usize)>,
        ) {
            let entry = ScheduleEntry {
                node: NodeRef(node),
                buffers: graph
                    .nodes
                    .get(&node)
                    .unwrap()
                    .connections
                    .iter()
                    .map(|c| {
                        let port = if c.src.0 .0 == node { c.src.1 } else { c.dst.1 };
                        (port, assignments.get(c).unwrap().0)
                    })
                    .collect(),
            };
            schedule.push(entry);
        }
        // If a latency requirement is found, we need to insert a delay into the graph.
        #[inline]
        fn insert_delay(
            graph: &mut Graph,
            edge: ConnectionInfo,
            amount: u64,
            state: &mut State,
            updates: &mut Vec<DelayInfo>,
        ) -> NodeRef {
            // reuse the delay if it already exists
            if let Some(delay) = state.delays.get_mut(&edge) {
                delay.delay = amount;
                updates.push(*delay);
                delay.node
            } else {
                // otherwise, create a new node
                let (src_node, src_port) = edge.src;
                let (dst_node, dst_port) = edge.dst;
                let delay_node = graph.node(&format!(
                    "delay-{}.{}-{}.{}",
                    graph.name(src_node).unwrap(),
                    graph.name(src_port).unwrap(),
                    graph.name(dst_node).unwrap(),
                    graph.name(dst_port).unwrap(),
                ));
                let input = graph.port(delay_node, "input", edge.type_).unwrap();
                let output = graph.port(delay_node, "output", edge.type_).unwrap();
                graph.disconnect_edge(edge).unwrap();
                graph.connect(src_port, input).unwrap();
                graph.connect(output, dst_port).unwrap();
                let delay = DelayInfo {
                    delay: amount,
                    node: delay_node,
                };
                updates.push(delay);
                delay_node
            }
        }
        // Once we reach a node after all its dependencies, we compute its latency requirements
        // and return an iterator of delay nodes to be inserted
        #[inline]
        fn solve_latency_reqs<'a, 'b>(
            graph: &'a mut Graph,
            node: u64,
            state: &'a mut State,
            updates: &'a mut Vec<DelayInfo>,
        ) -> impl Iterator<Item = NodeRef> + 'a {
            dbg!(graph.name(NodeRef(node)));
            let node = graph.nodes.get_mut(&node).unwrap().clone();
            let incoming = node.dependencies();
            let latencies = incoming
                .map(|i| {
                    dbg!(graph.name(i));
                    let node = graph.nodes.get(&i.0).expect("node not in graph");
                    node.delay + node.latency.unwrap()
                })
                .collect::<Vec<_>>();
            let latency = latencies.iter().copied().max().unwrap_or(0);
            graph.nodes.get_mut(&node.id).unwrap().latency = Some(latency);
            latencies
                .into_iter()
                .zip(node.incoming().collect::<Vec<_>>().into_iter())
                .map(move |(input_latency, edge)| {
                    insert_delay(graph, edge, latency - input_latency, state, updates)
                })
        }
        // After latency requirements are solved we need to compute which buffers to allocate.
        #[inline]
        fn solve_buffer_reqs(
            graph: &mut Graph,
            node: u64,
            assignments: &mut HashMap<ConnectionInfo, (BufferRef, usize)>,
            allocator: &mut BufferAllocator,
        ) {
            let node = graph.nodes.get(&node).unwrap();
            for (port_id, port) in node.ports.iter().copied() {
                dbg!(graph.name(NodeRef(node.id)), graph.name(PortRef(port_id)));
                let buffer = allocator.acquire(port.type_);
                let mut has_outgoing = false;
                for c in node
                    .connections
                    .iter()
                    .filter(|c| c.src.1 == PortRef(port_id))
                {
                    has_outgoing = true;
                    debug_assert!(!assignments.contains_key(c));
                    let e = assignments.entry(*c).or_insert((buffer, 0));
                    e.1 += 1;
                }
                for _c in node.incoming() {
                    // if let Some((buffer, count)) = assignments.get_mut(&c) {
                    //     let (buffer, count) = assignments.get_mut(&c).unwrap();
                    //     *count -= 1;
                    //     if *count == 0 {
                    //         allocator.release(*buffer);
                    //     }
                    // } else {
                    //     // the input buffer may not be assigned if a delay node has been inserted after
                    //     // the source node has been created. In this case we need to walk back the edge,
                    //     // assign the buffer to the input, and move on.
                    // }
                }
                if !has_outgoing {
                    allocator.release(buffer);
                }
            }
        }
        // The main driver of the compilation pass
        fn driver(graph: &mut Graph, root: NodeRef, state: &mut State) -> Vec<ScheduleEntry> {
            let mut allocator = BufferAllocator::new();
            let mut schedule = Vec::with_capacity(graph.nodes.len());
            let mut visited: HashSet<u64> = HashSet::with_capacity(graph.nodes.len());
            let mut delay_updates = vec![];
            let mut buffer_assignments: HashMap<ConnectionInfo, (BufferRef, usize)> =
                HashMap::with_capacity(graph.nodes.len() * 3 / 2);
            let mut queue = VecDeque::with_capacity(graph.nodes.len() * 3 / 2);
            queue.push_back(root.0);
            while let Some(node) = queue.pop_front() {
                {
                    if visited.contains(&node) {
                        continue;
                    }
                    let stack_height = queue.len();
                    let node = graph.nodes.get(&node).unwrap();
                    queue.extend(node.dependencies().filter_map(|n| {
                        if visited.contains(&n.0) {
                            None
                        } else {
                            Some(n.0)
                        }
                    }));
                    if stack_height != queue.len() {
                        queue.push_back(node.id);
                        continue;
                    };
                    queue.extend(node.dependents().filter_map(|n| {
                        if visited.contains(&n.0) {
                            None
                        } else {
                            Some(n.0)
                        }
                    }));
                }
                for inserted in
                    solve_latency_reqs(graph, node, state, &mut delay_updates).collect::<Vec<_>>()
                {
                    dbg!(inserted);
                    solve_buffer_reqs(graph, inserted.0, &mut buffer_assignments, &mut allocator);
                    add_to_schedule(graph, inserted.0, &mut schedule, &buffer_assignments);
                }
                solve_buffer_reqs(graph, node, &mut buffer_assignments, &mut allocator);
                add_to_schedule(graph, node, &mut schedule, &buffer_assignments);
                visited.insert(node);
            }
            schedule
        }
        // Launch!
        driver(self, root, state)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DelayInfo {
    node: NodeRef,
    delay: u64,
}

#[derive(Default)]
pub struct State {
    delays: HashMap<ConnectionInfo, DelayInfo>,
}

#[derive(Debug)]
pub struct ScheduleEntry {
    pub node: NodeRef,
    pub buffers: Vec<(PortRef, BufferRef)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn basic_ops() {
        let mut graph = Graph::default();
        let a = graph.node("A");
        let b = graph.node("B");

        let a_in = graph
            .port(a, "input", PortType::Event)
            .expect("port was not created");
        let a_out = graph
            .port(a, "output", PortType::Audio)
            .expect("port was not created");
        let b_in = graph
            .port(b, "input", PortType::Audio)
            .expect("port was not created");

        graph.connect(a_out, b_in).expect("could not connect");
        graph
            .connect(a_in, b_in)
            .expect_err("connected mistyped ports");
        graph.delete(a_in).expect("could not delete port");
        graph
            .disconnect(a_out, b_in)
            .expect("could not disconnect ports");
        graph.delete(a).expect("could not delete");
        graph
            .connect(a_out, b_in)
            .expect_err("connected node that doesn't exist");
    }
    // #[test]
    // fn simple_graph() {
    //     let mut graph = Graph::default();
    //     let (a, b, c, d) = (
    //         graph.node("A"),
    //         graph.node("B"),
    //         graph.node("C"),
    //         graph.node("D"),
    //     );
    //     let (a_out, b_out, c_out) = (
    //         graph
    //             .port(a, "output", PortType::Audio)
    //             .expect("could not create output port"),
    //         graph
    //             .port(b, "output", PortType::Audio)
    //             .expect("could not create output port"),
    //         graph
    //             .port(c, "output", PortType::Audio)
    //             .expect("could not create output port"),
    //     );
    //     let (b_in, c_in, d_in) = (
    //         graph
    //             .port(b, "input", PortType::Audio)
    //             .expect("could not create input"),
    //         graph
    //             .port(c, "input", PortType::Audio)
    //             .expect("could not create input"),
    //         graph
    //             .port(d, "input", PortType::Audio)
    //             .expect("could not create input"),
    //     );
    //     graph.set_delay(b, 2).expect("could not update delay of b");
    //     graph.set_delay(c, 5).expect("coudl not update delay of c");
    //     graph.connect(a_out, b_in).expect("could not connect");
    //     graph.connect(a_out, c_in).expect("could not connect");
    //     graph.connect(c_out, d_in).expect("could not connect");
    //     graph.connect(b_out, d_in).expect("could not connect");
    //     let mut state = State::default();
    //     let schedule = graph.compile(d, &mut state);
    //     assert_eq!(schedule.len(), 4);
    //     for entry in schedule {
    //         println!(
    //             "process \"{}\" with buffers: {:?}",
    //             graph.name(entry.node).unwrap(),
    //             entry.buffers
    //         );
    //     }
    // }
}
