use crate::cache::HeapStore;
use crate::error::Error;
use crate::port_type::PortType;
use crate::scheduled::ScheduledNode;
use crate::vec::Vec;
use fnv::FnvHashMap;
use std::fmt::Debug;
use std::{borrow::Borrow, collections::VecDeque};

/// A reference to a node in the graph
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct NodeRef(usize);

/// A reference to a port in the graph
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct PortRef(usize);

impl NodeRef {
    pub fn new(u: usize) -> Self {
        Self(u)
    }
}

impl PortRef {
    pub fn new(u: usize) -> Self {
        Self(u)
    }
}

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

impl From<NodeRef> for usize {
    fn from(node: NodeRef) -> Self {
        node.0
    }
}

impl From<PortRef> for usize {
    fn from(port: PortRef) -> Self {
        port.0
    }
}

#[derive(Clone, Copy, Debug)]
struct Edge<P> {
    src_node: NodeRef,
    src_port: PortRef,
    dst_node: NodeRef,
    dst_port: PortRef,
    type_: P,
}

impl<P> PartialEq for Edge<P>
where
    P: PortType + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        // For our purposes, comparing just src_port and dst_port is sufficient.
        // Ports can only be of one type, and they can only belong to one node.
        // Deleting ports should garauntee that all corresponding edges are also
        // deleted, so reusing ports should not cause a problem.
        self.src_port == other.src_port && self.dst_port == other.dst_port
    }
}

#[derive(Debug)]
pub struct Graph<N, P, PT>
where
    PT: PortType,
{
    edges: Vec<Vec<Edge<PT>>>,
    ports: Vec<Vec<PortRef>>,
    delays: Vec<u64>,
    port_data: Vec<(NodeRef, PT)>,
    port_identifiers: Vec<P>,
    node_identifiers: Vec<N>,
    free_nodes: Vec<NodeRef>,
    free_ports: Vec<PortRef>,
    heap_store: Option<HeapStore<N, P, PT>>,
}

impl<N, P, PT> Default for Graph<N, P, PT>
where
    N: Clone,
    P: Clone,
    PT: PortType,
{
    fn default() -> Self {
        Self {
            edges: Vec::new(),
            ports: Vec::new(),
            delays: Vec::new(),
            port_data: Vec::new(),
            port_identifiers: Vec::new(),
            node_identifiers: Vec::new(),
            free_nodes: Vec::new(),
            free_ports: Vec::new(),
            heap_store: Some(HeapStore::default()),
        }
    }
}

impl<N, P, PT> Graph<N, P, PT>
where
    N: Clone,
    P: Clone,
    PT: PortType,
{
    pub fn node(&mut self, ident: N) -> NodeRef {
        if let Some(node) = self.free_nodes.pop() {
            let id: usize = node.0;
            self.edges[id].clear();
            self.ports[id].clear();
            self.delays[id] = 0;
            self.node_identifiers[id] = ident;
            node
        } else {
            let id = self.node_count();
            self.edges.push(smallvec::smallvec![]);
            self.ports.push(smallvec::smallvec![]);
            self.delays.push(0);
            self.node_identifiers.push(ident);
            NodeRef::new(id)
        }
    }

    pub fn port(&mut self, node: NodeRef, type_: PT, ident: P) -> Result<PortRef, Error> {
        if node.0 < self.node_count() && !self.free_nodes.contains(&node) {
            if let Some(port) = self.free_ports.pop() {
                self.ports[node.0].push(port);
                self.port_data[port.0] = (node, type_);
                self.port_identifiers[port.0] = ident;
                Ok(port)
            } else {
                let port = PortRef::new(self.port_count());

                self.ports[node.0].push(port);
                self.port_data.push((node, type_));
                self.port_identifiers.push(ident);

                Ok(port)
            }
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
        let index = self.ports[node.0]
            .iter()
            .position(|p_| *p_ == p)
            .ok_or(Error::PortDoesNotExist)?;
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

        let (src_node, src_type) = self.port_data[src.0];
        let (dst_node, dst_type) = self.port_data[dst.0];
        if src_type != dst_type {
            return Err(Error::InvalidPortType);
        }

        for edge in self.incoming(dst) {
            if edge.src_port == src {
                // These two ports are already connected.
                return Ok(());
            }
        }

        self.cycle_check(src_node, dst_node)?;

        let edge = Edge {
            src_node,
            src_port: src,
            dst_node,
            dst_port: dst,
            type_: src_type,
        };

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

    pub fn set_delay(&mut self, node: NodeRef, delay: u64) -> Result<(), Error> {
        self.node_check(node)?;
        self.delays[node.0] = delay;
        Ok(())
    }

    pub fn port_ident(&self, port: PortRef) -> Result<&'_ P, Error> {
        self.port_check(port)?;
        Ok(&self.port_identifiers[port.0])
    }

    pub fn node_ident(&self, node: NodeRef) -> Result<&'_ N, Error> {
        self.node_check(node)?;
        Ok(&self.node_identifiers[node.0])
    }

    pub fn set_port_ident(&mut self, port: PortRef, ident: P) -> Result<(), Error> {
        self.port_check(port)?;
        self.port_identifiers[port.0] = ident;
        Ok(())
    }

    pub fn set_node_ident(&mut self, node: NodeRef, ident: N) -> Result<(), Error> {
        self.node_check(node)?;
        self.node_identifiers[node.0] = ident;
        Ok(())
    }

    pub fn node_check(&self, node: NodeRef) -> Result<(), Error> {
        if node.0 < self.node_count() && !self.free_nodes.contains(&node) {
            Ok(())
        } else {
            Err(Error::NodeDoesNotExist)
        }
    }

    pub fn port_check(&self, port: PortRef) -> Result<(), Error> {
        if port.0 < self.port_count() && !self.free_ports.contains(&port) {
            Ok(())
        } else {
            Err(Error::PortDoesNotExist)
        }
    }

    fn node_count(&self) -> usize {
        self.ports.len()
    }

    fn port_count(&self) -> usize {
        self.port_data.len()
    }

    /// Check that adding an edge `src` -> `dst` won't create a cycle. Must be called
    /// before each edge addition.
    ///
    /// TODO: Optimize for adding multiple edges at once. (pass over the whole graph)
    fn cycle_check(&mut self, src: NodeRef, dst: NodeRef) -> Result<(), Error> {
        // This won't panic because this is always `Some` on the user's end.
        let mut heap_store = self.heap_store.take().unwrap();
        let mut queue = heap_store.walk_queue.take().unwrap();
        let mut queued = heap_store.cycle_queued.take().unwrap();

        queue.clear();
        queued.clear();
        queue.push_back(dst);
        queued.insert(dst);

        while let Some(node) = queue.pop_front() {
            if node == src {
                heap_store.walk_queue = Some(queue);
                heap_store.cycle_queued = Some(queued);
                self.heap_store = Some(heap_store);

                return Err(Error::Cycle);
            }
            for dependent in self.dependents(node) {
                if !queued.contains(&dependent) {
                    queue.push_back(dependent);
                    queued.insert(dependent);
                }
            }
        }

        heap_store.walk_queue = Some(queue);
        heap_store.cycle_queued = Some(queued);
        self.heap_store = Some(heap_store);

        Ok(())
    }

    fn remove_edge(&mut self, edge: Edge<PT>) -> Result<(), Error> {
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

    fn incoming(&self, port: PortRef) -> impl Iterator<Item = Edge<PT>> + '_ {
        let node = self.port_data[port.0].0;
        self.edges[node.0]
            .iter()
            .filter(move |e| e.dst_port == port)
            .copied()
    }

    fn outgoing(&self, port: PortRef) -> impl Iterator<Item = Edge<PT>> + '_ {
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

    /// Walk graph in topological order using Kahn's algorithm.
    fn walk_mut(
        &mut self,
        heap_store: &mut HeapStore<N, P, PT>,
        queue: &mut VecDeque<NodeRef>,
        indegree: &mut FnvHashMap<NodeRef, usize>,
        mut f: impl FnMut(&mut Graph<N, P, PT>, NodeRef, &mut HeapStore<N, P, PT>),
    ) {
        queue.clear();
        indegree.clear();

        for node_index in 0..self.node_count() {
            indegree.insert(NodeRef::new(node_index), 0);
        }
        for node in &self.free_nodes {
            indegree.remove(node);
        }

        for (&node, value) in indegree.iter_mut() {
            *value = self.dependencies(node).count();
            if *value == 0 {
                queue.push_back(node);
            }
        }

        while let Some(node) = queue.pop_front() {
            (&mut f)(self, node, heap_store);
            for dependent in self.dependents(node) {
                let value = indegree
                    .get_mut(&dependent)
                    .expect("edge refers to freed node");
                *value = value.checked_sub(1).expect("corrupted graph");
                if *value == 0 {
                    queue.push_back(dependent);
                }
            }
        }
    }

    pub fn compile(&mut self) -> &[ScheduledNode<N, P, PT>] {
        let solve_latency_requirements =
            |graph: &mut Graph<N, P, PT>,
             node: NodeRef,
             heap_store: &mut HeapStore<N, P, PT>,
             delay_comps: &mut FnvHashMap<(PortRef, PortRef), u64>| {
                heap_store.deps.clear();
                heap_store.latencies.clear();
                for edge in graph.edges[node.0].iter().filter(|e| e.dst_node == node) {
                    heap_store.deps.push(edge.src_node);

                    heap_store.latencies.push(
                        heap_store.all_latencies[edge.src_node.0].unwrap()
                            + graph.delays[edge.src_node.0],
                    );
                }

                let max_latency = heap_store.latencies.iter().max().copied().or(Some(0));

                heap_store.all_latencies[node.0] = max_latency;

                for (dep, latency) in heap_store.deps.iter().zip(heap_store.latencies.iter()) {
                    let compensation = max_latency.unwrap() - latency;
                    if compensation != 0 {
                        for edge in graph.edges[node.0].iter().filter(|e| e.src_node == *dep) {
                            let _ =
                                delay_comps.insert((edge.src_port, edge.dst_port), compensation);
                        }
                    }
                }
            };

        let solve_buffer_requirements =
            |graph: &Graph<N, P, PT>, node: NodeRef, heap_store: &mut HeapStore<N, P, PT>| {
                for port in &graph.ports[node.0] {
                    let (_, type_) = graph.port_data[port.0];

                    for output in graph.outgoing(*port) {
                        let (buffer, count) = heap_store
                            .output_assignments
                            .entry((node, *port))
                            .or_insert((heap_store.allocator.acquire(type_), 0));
                        *count += 1;
                        heap_store
                            .input_assignments
                            .entry((output.dst_node, output.dst_port))
                            .or_insert(smallvec::smallvec![])
                            .push((*buffer, (output.src_port, output.dst_port)));
                    }
                    for input in graph.incoming(*port) {
                        let (buffer, count) = heap_store
                            .output_assignments
                            .get_mut(&(input.src_node, input.src_port))
                            .expect("no output buffer assigned");
                        *count -= 1;
                        if *count == 0 {
                            heap_store.allocator.release(*buffer);
                        }
                    }
                }
            };

        // This won't panic because this is always `Some` on the user's end.
        let mut heap_store = self.heap_store.take().unwrap();

        let mut scheduled = heap_store.scheduled.take().unwrap_or_default();
        scheduled.clear();

        heap_store.all_latencies.clear();
        heap_store.all_latencies.resize(self.node_count(), None);

        let mut delay_comps = heap_store.delay_comps.take().unwrap();
        delay_comps.clear();

        heap_store.allocator.clear();
        heap_store.input_assignments.clear();
        heap_store.output_assignments.clear();

        let mut scheduled_nodes = heap_store.scheduled_nodes.take().unwrap();
        scheduled_nodes.clear();

        let mut queue = heap_store.walk_queue.take().unwrap();
        let mut indegree = heap_store.walk_indegree.take().unwrap();

        self.walk_mut(
            &mut heap_store,
            &mut queue,
            &mut indegree,
            |graph, node, heap_store| {
                // TODO: Maybe use the log crate for this to avoid polluting the user's output?
                // println!("compiling {}", graph.node_name(node).unwrap());

                solve_latency_requirements(graph, node, heap_store, &mut delay_comps);
                solve_buffer_requirements(graph, node, heap_store);
                scheduled_nodes.push(node);
            },
        );

        for node in scheduled_nodes.iter() {
            let node_ident = self.node_identifiers[node.0].clone();

            let inputs = self.ports[node.0]
                .iter()
                .filter_map(|port| {
                    heap_store
                        .input_assignments
                        .get(&(*node, *port))
                        .map(|buffers| {
                            let buffers = buffers
                                .iter()
                                .map(|(buffer, ports)| {
                                    let delay_comp = delay_comps.get(ports).copied().unwrap_or(0);

                                    (*buffer, delay_comp)
                                })
                                .collect();

                            (self.port_identifiers[port.0].clone(), buffers)
                        })
                })
                .collect::<Vec<_>>();
            let outputs = self.ports[node.0]
                .iter()
                .filter_map(|port| {
                    heap_store
                        .output_assignments
                        .get(&(*node, *port))
                        .map(|(buffer, _)| (self.port_identifiers[port.0].clone(), *buffer))
                })
                .collect::<Vec<_>>();

            scheduled.push(ScheduledNode {
                node: node_ident,
                inputs,
                outputs,
            });
        }

        heap_store.scheduled = Some(scheduled);
        heap_store.delay_comps = Some(delay_comps);
        heap_store.scheduled_nodes = Some(scheduled_nodes);
        heap_store.walk_queue = Some(queue);
        heap_store.walk_indegree = Some(indegree);

        self.heap_store = Some(heap_store);

        self.heap_store
            .as_ref()
            .unwrap()
            .scheduled
            .as_ref()
            .unwrap()
    }
}
