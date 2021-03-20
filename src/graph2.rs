use smallvec::smallvec as vec;
use smallvec::SmallVec;
use std::{
    borrow::Borrow,
    collections::{HashMap, HashSet, VecDeque},
};
type Vec<T> = SmallVec<[T; 16]>;
use crate::Error;
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeRef(usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PortRef(usize);
impl Borrow<usize> for NodeRef {
    fn borrow(&self) -> &'_ usize {
        &self.0
    }
}
impl Borrow<usize> for PortRef {
    fn borrow(&self) -> &'_ usize {
        &self.0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Edge {
    src_node: NodeRef,
    src_port: PortRef,
    dst_node: NodeRef,
    dst_port: PortRef,
    type_: PortType,
}

#[derive(Default)]
struct BufferAllocator {
    event_buffer_count: usize,
    audio_buffer_count: usize,
    event_buffer_stack: Vec<usize>,
    audio_buffer_stack: Vec<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PortType {
    Audio,
    Event,
}

#[derive(Default)]
pub struct Graph {
    edges: Vec<Vec<Edge>>,
    ports: Vec<Vec<PortRef>>,
    delays: Vec<u64>,
    port_data: Vec<(NodeRef, PortType)>,
    port_names:Vec<String>, 
    node_names:Vec<String>,
    free_nodes: Vec<NodeRef>,
    free_ports: Vec<PortRef>,
}

#[derive(Copy, Clone, Debug)]
pub struct Buffer {
    index: usize,
    type_: PortType,
}

impl BufferAllocator {
    fn acquire(&mut self, type_: PortType) -> Buffer {
        let (count, stack) = match type_ {
            PortType::Event => (&mut self.event_buffer_count, &mut self.event_buffer_stack),
            PortType::Audio => (&mut self.audio_buffer_count, &mut self.audio_buffer_stack),
        };
        if let Some(index) = stack.pop() {
            Buffer { index, type_ }
        } else {
            let buffer = Buffer {
                index: *count,
                type_,
            };
            *count += 1;
            buffer
        }
    }

    fn release(&mut self, ref_: Buffer) {
        let stack = match ref_.type_ {
            PortType::Event => &mut self.event_buffer_stack,
            PortType::Audio => &mut self.audio_buffer_stack,
        };
        stack.push(ref_.index);
    }
}

#[derive(Clone, Debug)]
pub struct Scheduled {
    pub node: NodeRef,
    pub inputs: Vec<(PortRef, Vec<Buffer>)>,
    pub outputs: Vec<(PortRef, Buffer)>,
}

impl Graph {
    pub fn node(&mut self, name:&str) -> NodeRef {
        if let Some(node) = self.free_nodes.pop() {
            let id = node.0;
            self.edges[id].clear();
            self.ports[id].clear();
            self.delays[id] = 0;
            self.node_names[id] = name.to_owned();
            node
        } else {
            let id = self.node_count();
            self.edges.push(vec![]);
            self.ports.push(vec![]);
            self.delays.push(0);
            self.node_names.push(name.to_owned());
            NodeRef(id)
        }
    }

    pub fn port(&mut self, node: NodeRef, type_: PortType, name:&str) -> Result<PortRef, Error> {
        if node.0 < self.node_count() && !self.free_nodes.contains(&node) {
            let port = self
                .free_ports
                .pop()
                .or_else(|| {
                    self.port_data.push((node, type_));
                    self.port_names.push(Default::default());
                    Some(PortRef(self.port_count() - 1))
                })
                .unwrap();
            self.ports[node.0].push(port);
            self.port_data[port.0] = (node, type_);
            self.port_names[port.0] = name.to_owned();
            Ok(port)
        } else {
            Err(Error::NodeDoesNotExist)
        }
    }

    pub fn delete_port(&mut self, p: PortRef) -> Result<(), Error> {
        self.port_check(p)?;
        let (node, _) = self.port_data[p.0];
        for e in self.edges[node.0]
            .clone()
            .into_iter()
            .filter(|e| e.dst_port == p || e.src_port == p)
        {
            let _e = self.remove_edge(e);
            debug_assert!(_e.is_ok());
        }
        let index = self.ports[node.0].iter().position(|p_| *p_ == p).ok_or(Error::PortDoesNotExist)?; 
        self.ports[node.0].remove(index);
        self.free_ports.push(p);
        Ok(())
    }

    pub fn delete_node(&mut self, n: NodeRef) -> Result<(), Error> {
        self.node_check(n)?;
        for p in self.ports[n.0].clone() {
            let _e = self.delete_port(p);
            debug_assert!(_e.is_ok());
        }
        self.free_nodes.push(n);
        Ok(())
    }

    pub fn connect(&mut self, src: PortRef, dst: PortRef) -> Result<(), Error> {
        self.port_check(src)?;
        self.port_check(dst)?;
        self.cycle_check(src, dst)?;
        let (src_node, src_type) = self.port_data[src.0];
        let (dst_node, dst_type) = self.port_data[dst.0];
        if src_type != dst_type {
            return Err(Error::InvalidPortType);
        }
        let edge = Edge {
            src_node,
            src_port: src,
            dst_node,
            dst_port: dst,
            type_: src_type,
        };
        println!("connection {}.{} to {}.{} with edge: {:?}", self.node_name(src_node).unwrap(), self.port_name(src).unwrap(), self.node_name(dst_node).unwrap(), self.port_name(dst).unwrap(), edge);
        self.edges[src_node.0].push(edge);
        self.edges[dst_node.0].push(edge);
        Ok(())
    }

    pub fn disconnect(&mut self, src: PortRef, dst: PortRef) -> Result<(), Error> {
        self.port_check(src)?;
        self.port_check(dst)?;
        let (src_node, _) = self.port_data[src.0];
        let (dst_node, _) = self.port_data[dst.0];
        let type_ = self.port_data[src.0].1;
        self.remove_edge(Edge {
            src_node,
            src_port: src,
            dst_node,
            dst_port: dst,
            type_,
        })
    }

    pub fn set_delay(&mut self, node:NodeRef, delay:u64) -> Result<(), Error> {
        self.node_check(node)?; 
        self.delays[node.0] = delay;
        Ok(())
    }

    pub fn port_name(&self, port:PortRef) -> Result<&'_ str, Error> {
        self.port_check(port)?; 
        Ok(&self.port_names[port.0])
    }

    pub fn node_name(&self, node:NodeRef) -> Result<&'_ str, Error> {
        self.node_check(node)?; 
        Ok(&self.node_names[node.0])
    }
    
    fn node_count(&self) -> usize {
        self.ports.len()
    }

    fn port_count(&self) -> usize {
        self.port_data.len()
    }

    fn node_check(&self, n: NodeRef) -> Result<(), Error> {
        if n.0 < self.node_count() && !self.free_nodes.contains(&n) {
            Ok(())
        } else {
            Err(Error::NodeDoesNotExist)
        }
    }

    fn port_check(&self, p: PortRef) -> Result<(), Error> {
        if p.0 < self.port_count() && !self.free_ports.contains(&p) {
            Ok(())
        } else {
            Err(Error::PortDoesNotExist)
        }
    }

    fn cycle_check(&self, src: PortRef, dst: PortRef) -> Result<(), Error> {
        let mut stack: Vec<PortRef> = vec![src];
        while let Some(src) = stack.pop() {
            if src == dst {
                return Err(Error::Cycle);
            }
            stack.extend(self.outgoing(src).map(|e| e.dst_port));
        }
        Ok(())
    }

    fn remove_edge(&mut self, edge: Edge) -> Result<(), Error> {
        let Edge {
            src_node, dst_node, ..
        } = edge;
        let src_index = self.edges[src_node.0].iter().position(|e| *e == edge);
        let dst_index = self.edges[dst_node.0].iter().position(|e| *e == edge);
        match (src_index, dst_index) {
            (Some(s), Some(d)) => {
                self.edges[src_node.0].remove(s);
                self.edges[dst_node.0].remove(d);
                Ok(())
            }
            _ => Err(Error::ConnectionDoesNotExist),
        }
    }

    fn incoming(&self, port: PortRef) -> impl Iterator <Item = Edge> + '_ {
        let node = self.port_data[port.0].0;
        self.edges[node.0]
            .iter()
            .filter(move |e| e.dst_port == port)
            .copied()
    }

    fn outgoing(&self, port: PortRef) -> impl Iterator<Item = Edge> + '_ {
        let node = self.port_data[port.0].0;
        self.edges[node.0]
            .iter()
            .filter(move |e| e.src_port == port)
            .copied()
    }

    fn dependencies(&self, node: NodeRef) -> impl Iterator<Item = NodeRef> + '_ {
        self.edges[node.0].iter().filter_map(move |e| {
            if e.dst_node == node {
                Some(e.src_node)
            } else {
                None
            }
        })
    }

    fn dependents(&self, node: NodeRef) -> impl Iterator<Item = NodeRef> + '_ {
        self.edges[node.0].iter().filter_map(move |e| {
            if e.src_node == node {
                Some(e.dst_node)
            } else {
                None
            }
        })
    }

    fn walk_mut(&mut self, root: NodeRef, mut f: impl FnMut(&mut Graph, NodeRef)) {
        let mut queue = VecDeque::new();
        let mut visited: HashSet<NodeRef> = HashSet::new();
        queue.push_back(root);
        while let Some(node) = queue.pop_front() {
            if visited.contains(&node) {
                continue;
            }
            let len = queue.len();
            queue.extend(self.dependencies(node).filter(|n| !visited.contains(n)));
            if queue.len() != len {
                queue.push_back(node);
                continue;
            }
            (&mut f)(self, node);
            queue.extend(self.dependents(node).filter(|n| !visited.contains(n)));
            visited.insert(node);
        }
    }

    pub fn compile(&mut self, root: NodeRef) -> impl Iterator<Item = Scheduled> + '_ {
        let mut all_latencies: Vec<Option<u64>> = vec![None; self.node_count()];
        let mut delays: HashMap<Edge, (NodeRef, u64)> = Default::default();
        let mut solve_latency_requirements = |graph: &mut Graph, node: NodeRef| {
            let _deps = graph.dependencies(node).collect::<Vec<_>>();
            let latencies = graph
                .dependencies(node)
                .map(|n| all_latencies[n.0].unwrap() + graph.delays[n.0])
                .collect::<Vec<_>>();
            let max_latency = latencies.iter().max().copied().or(Some(0));
            all_latencies[node.0] = max_latency;
            let mut insertions: Vec<NodeRef> = vec![];
            for (dep, latency) in graph
                .dependencies(node)
                .zip(latencies.into_iter())
                .collect::<Vec<_>>()
            {
                for edge in graph.edges[node.0]
                    .clone()
                    .into_iter()
                    .filter(move |e| e.src_node == dep)
                {
                    let compensation = max_latency.unwrap() - latency;
                    if compensation == 0 {
                        continue;
                    }
                    if let Some((node, delay)) = delays.get_mut(&edge) {
                        *delay = compensation;
                        graph.delays[node.0] = compensation;
                    } else {
                        graph.remove_edge(edge).unwrap();
                        let delay_node = graph.node(&format!("delay-{}.{}-{}.{}", 
                            &graph.node_names[edge.src_node.0], 
                            &graph.port_names[edge.src_port.0], 
                            &graph.node_names[edge.dst_node.0], 
                            &graph.port_names[edge.dst_port.0], 
                        ));
                        let delay_input = graph.port(delay_node, edge.type_, "input").unwrap();
                        let delay_output = graph.port(delay_node, edge.type_, "output").unwrap();
                        graph.connect(edge.src_port, delay_input).unwrap();
                        graph.connect(delay_output, edge.dst_port).unwrap();
                        graph.delays[delay_node.0] = compensation;
                        delays.insert(edge, (delay_node, compensation));
                        insertions.push(delay_node);
                    }
                }
            }
            insertions.into_iter()
        };

        let mut allocator = BufferAllocator::default();
        let mut input_assignments: HashMap<(NodeRef, PortRef), Vec<Buffer>> = Default::default();
        let mut output_assignments: HashMap<(NodeRef, PortRef), (Buffer, usize)> = Default::default();

        let mut solve_buffer_requirements = |graph: &Graph, node: NodeRef| {
            for port in &graph.ports[node.0] {
                let (_, type_) = graph.port_data[port.0];
                for output in graph.outgoing(*port) {
                    let (buffer, count) = output_assignments.entry((node, *port)).or_insert((allocator.acquire(type_), 0));
                    *count += 1;
                    input_assignments.entry((output.dst_node, output.dst_port))
                        .or_insert(vec![])
                        .push(*buffer);
                }
                for input in graph.incoming(*port) {
                    let (buffer, count) = output_assignments.get_mut(&(input.src_node, input.src_port)).expect("no output buffer assigned");
                    *count -=  1; 
                    if *count == 0 {
                        allocator.release(*buffer);
                    }
                }
                
            }
        };

        let mut schedule: Vec<NodeRef> = vec![];
        self.walk_mut(root, |graph, node| {
            println!("compiling {}", graph.node_name(node).unwrap());
            let insertions = solve_latency_requirements(graph, node);
            for insertion in insertions {
                solve_buffer_requirements(graph, insertion);
                schedule.push(insertion);
            }
            solve_buffer_requirements(graph, node);
            schedule.push(node);
        });

        schedule.into_iter()
            .rev()
            .map(move |node| {
                let inputs: Vec<(PortRef, Vec<Buffer>)> = 
                    self.ports[node.0]
                        .iter()
                        .filter_map(|port| 
                            input_assignments
                                .get(&(node, *port))
                                .map(|buffers| (*port, buffers.clone()))
                        ).collect::<Vec<_>>();
                let outputs = self.ports[node.0]
                        .iter()
                        .filter_map(|port|{
                            output_assignments.get(&(node, *port))
                                .map(|(buffer, _)| (*port,*buffer))
                        })
                        .collect::<Vec<_>>();
                Scheduled {
                    node, 
                    inputs, 
                    outputs,
                }
            })
    }
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
            .port(a, PortType::Event, "events")
            .expect("port was not created");
        let a_out = graph
            .port(a, PortType::Audio, "output")
            .expect("port was not created");
        let b_in = graph
            .port(b, PortType::Audio, "input")
            .expect("port was not created");

        dbg!(&graph.port_count());
        graph.connect(a_out, b_in).expect("could not connect");
        graph
            .connect(a_in, b_in)
            .expect_err("connected mistyped ports");
        graph.delete_port(a_in).expect("could not delete port");
        graph
            .disconnect(a_out, b_in)
            .expect("could not disconnect ports");
        graph.delete_node(a).expect("could not delete");
        graph
            .connect(a_out, b_in)
            .expect_err("connected node that doesn't exist");
    }
    
    #[test]
    fn simple_graph() {
        let mut graph = Graph::default();
        let (a, b, c, d) = (
            graph.node("A"),
            graph.node("B"),
            graph.node("C"),
            graph.node("D"),
        );
        let (a_out, b_out, c_out) = (
            graph
                .port(a, PortType::Audio, "output")
                .expect("could not create output port"),
            graph
                .port(b, PortType::Audio, "output")
                .expect("could not create output port"),
            graph
                .port(c, PortType::Audio, "output")
                .expect("could not create output port"),
        );

        let (b_in, c_in, d_in) = (
            graph
                .port(b, PortType::Audio, "input")
                .expect("could not create input"),
            graph
                .port(c, PortType::Audio, "input")
                .expect("could not create input"),
            graph
                .port(d, PortType::Audio, "input")
                .expect("could not create input"),
        );
        graph.set_delay(b, 2).expect("could not update delay of b");
        graph.set_delay(c, 5).expect("coudl not update delay of c");
        graph.connect(a_out, b_in).expect("could not connect");
        graph.connect(a_out, c_in).expect("could not connect");
        graph.connect(c_out, d_in).expect("could not connect");
        graph.connect(b_out, d_in).expect("could not connect");
        let schedule = graph.compile(d).collect::<Vec<_>>();
        for entry in schedule {
            let node_name = graph.node_name(entry.node).unwrap(); 
            println!("process {}:", node_name);
            for (port, buffers) in entry.inputs {
                println!("    {} => ", graph.port_name(port).unwrap()); 
                for b in buffers {
                    println!("        {}", b.index);
                }
            }
            for (port, buf) in entry.outputs {
                println!("    {} => {}", graph.port_name(port).unwrap(), buf.index);
            }
        }
    }
}