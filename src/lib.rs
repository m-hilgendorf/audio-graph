use fnv::{FnvHashMap, FnvHashSet};
use smallvec::smallvec as vec;
use smallvec::SmallVec;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::{borrow::Borrow, collections::VecDeque};

type Vec<T> = SmallVec<[T; 16]>;

#[derive(Debug, Clone, Copy)]
pub enum Error {
    NodeDoesNotExist,
    PortDoesNotExist,
    Cycle,
    ConnectionDoesNotExist,
    RefDoesNotExist,
    InvalidPortType,
    DstPortAlreadyConnected,
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NodeDoesNotExist => write!(f, "Audio graph node does not exist"),
            Error::PortDoesNotExist => write!(f, "Audio graph port does not exist"),
            Error::Cycle => write!(f, "Audio graph cycle detected"),
            Error::ConnectionDoesNotExist => write!(f, "Audio graph connection does not exist"),
            Error::RefDoesNotExist=> write!(f, "Audio graph reference does not exist"),
            Error::InvalidPortType => write!(f, "Cannot connect audio graph ports. Ports are a different type"),
            Error::DstPortAlreadyConnected => write!(f, "Cannot connect audio graph ports. The destination port is already connected to another port"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum NodeIdent<T: Debug + Clone, PT: PortType + PartialEq> {
    DelayComp(PT),
    User(T),
}

#[derive(Debug, Clone)]
pub enum PortIdent<T: Debug + Clone> {
    DelayComp,
    User(T),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeRef(usize);

impl From<NodeRef> for usize {
    fn from(n: NodeRef) -> Self {
        n.0
    }
}

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
struct Edge<PT: PortType + PartialEq> {
    src_node: NodeRef,
    src_port: PortRef,
    dst_node: NodeRef,
    dst_port: PortRef,
    type_: PT,
}

struct BufferAllocator<PT: PortType + PartialEq> {
    buffer_count_stacks: Vec<(usize, Vec<usize>)>,
    _phantom_port_type: PhantomData<PT>,
}

impl<PT: PortType> BufferAllocator<PT> {
    fn clear(&mut self) {
        for (c, s) in self.buffer_count_stacks.iter_mut() {
            *c = 0;
            s.clear();
        }
    }
}

impl<PT: PortType> Default for BufferAllocator<PT> {
    fn default() -> Self {
        let num_types = PT::num_types();
        let mut buffer_count_stacks = Vec::<(usize, Vec<usize>)>::new();
        for _ in 0..num_types {
            buffer_count_stacks.push((0, Vec::new()));
        }
        Self {
            buffer_count_stacks,
            _phantom_port_type: PhantomData::default(),
        }
    }
}

pub trait PortType: Debug + Clone + Copy + Eq + std::hash::Hash {
    fn into_index(&self) -> usize;
    fn num_types() -> usize;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DefaultPortType {
    Audio,
    Event,
}

impl PortType for DefaultPortType {
    #[inline]
    fn into_index(&self) -> usize {
        match self {
            DefaultPortType::Audio => 0,
            DefaultPortType::Event => 1,
        }
    }

    fn num_types() -> usize {
        2
    }
}

pub struct Graph<N, P, PT>
where
    N: Debug + Clone,
    P: Debug + Clone,
    PT: PortType + PartialEq,
{
    edges: Vec<Vec<Edge<PT>>>,
    ports: Vec<Vec<PortRef>>,
    delays: Vec<u64>,
    port_data: Vec<(NodeRef, PT)>,
    port_identifiers: Vec<PortIdent<P>>,
    node_identifiers: Vec<NodeIdent<N, PT>>,
    free_nodes: Vec<NodeRef>,
    free_ports: Vec<PortRef>,

    mem_cache: Option<MemCache<N, P, PT>>,
}

impl<N, P, PT> Clone for Graph<N, P, PT>
where
    N: Debug + Clone,
    P: Debug + Clone,
    PT: PortType + PartialEq,
{
    fn clone(&self) -> Self {
        Self {
            edges: self.edges.clone(),
            ports: self.ports.clone(),
            delays: self.delays.clone(),
            port_data: self.port_data.clone(),
            port_identifiers: self.port_identifiers.clone(),
            node_identifiers: self.node_identifiers.clone(),
            free_nodes: self.free_nodes.clone(),
            free_ports: self.free_ports.clone(),

            // We don't want to clone the cache.
            mem_cache: None,
        }
    }
}

impl<N, P, PT> Default for Graph<N, P, PT>
where
    N: Debug + Clone,
    P: Debug + Clone,
    PT: PortType + PartialEq,
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
            mem_cache: Some(MemCache::default()),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Buffer<PT: PortType + PartialEq> {
    index: usize,
    type_: PT,
}

impl<PT: PortType + PartialEq> BufferAllocator<PT> {
    fn acquire(&mut self, type_: PT) -> Buffer<PT> {
        let type_index = type_.into_index();
        let (count, stack) = &mut self.buffer_count_stacks[type_index];

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

    fn release(&mut self, ref_: Buffer<PT>) {
        let type_index = ref_.type_.into_index();
        let stack = &mut self.buffer_count_stacks[type_index].1;
        stack.push(ref_.index);
    }
}

pub struct MemCache<N, P, PT>
where
    N: Debug + Clone,
    P: Debug + Clone,
    PT: PortType + PartialEq,
{
    walk_queue: Option<VecDeque<NodeRef>>,
    walk_visited: Option<FnvHashSet<NodeRef>>,

    latencies: Option<Vec<u64>>,
    insertions: Option<Vec<NodeRef>>,
    all_latencies: Vec<Option<u64>>,
    delays: FnvHashMap<Edge<PT>, (NodeRef, u64)>,
    deps: Option<Vec<NodeRef>>,
    edges: Option<Vec<Edge<PT>>>,
    allocator: BufferAllocator<PT>,
    input_assignments: FnvHashMap<(NodeRef, PortRef), Vec<Buffer<PT>>>,
    output_assignments: FnvHashMap<(NodeRef, PortRef), (Buffer<PT>, usize)>,
    scheduled_nodes: Option<Vec<NodeRef>>,

    delay_comp_graph: Option<Box<Graph<N, P, PT>>>,
    scheduled: Option<Vec<Scheduled<N, P, PT>>>,
}

impl<N, P, PT> Default for MemCache<N, P, PT>
where
    N: Debug + Clone,
    P: Debug + Clone,
    PT: PortType + PartialEq,
{
    fn default() -> Self {
        Self {
            walk_queue: Some(VecDeque::new()),
            walk_visited: Some(FnvHashSet::default()),
            latencies: Some(Vec::new()),
            insertions: Some(Vec::new()),
            all_latencies: Vec::new(),
            delays: FnvHashMap::default(),
            deps: Some(Vec::new()),
            edges: Some(Vec::new()),
            allocator: BufferAllocator::default(),
            input_assignments: FnvHashMap::default(),
            output_assignments: FnvHashMap::default(),
            scheduled_nodes: Some(Vec::new()),
            delay_comp_graph: None,
            scheduled: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Scheduled<N, P, PT>
where
    N: Debug + Clone,
    P: Debug + Clone,
    PT: PortType + PartialEq,
{
    pub node: NodeIdent<N, PT>,
    pub inputs: Vec<(PortIdent<P>, Vec<Buffer<PT>>)>,
    pub outputs: Vec<(PortIdent<P>, Buffer<PT>)>,
}

impl<N, P, PT> Graph<N, P, PT>
where
    N: Debug + Clone,
    P: Debug + Clone,
    PT: PortType + PartialEq,
{
    pub fn node(&mut self, ident: N) -> NodeRef {
        self.node_(NodeIdent::User(ident))
    }

    fn node_(&mut self, ident: NodeIdent<N, PT>) -> NodeRef {
        if let Some(node) = self.free_nodes.pop() {
            let id = node.0;
            self.edges[id].clear();
            self.ports[id].clear();
            self.delays[id] = 0;
            self.node_identifiers[id] = ident;
            node
        } else {
            let id = self.node_count();
            self.edges.push(vec![]);
            self.ports.push(vec![]);
            self.delays.push(0);
            self.node_identifiers.push(ident);
            NodeRef(id)
        }
    }

    pub fn port(&mut self, node: NodeRef, type_: PT, ident: P) -> Result<PortRef, Error> {
        self.port_(node, type_, PortIdent::User(ident))
    }

    fn port_(&mut self, node: NodeRef, type_: PT, ident: PortIdent<P>) -> Result<PortRef, Error> {
        if node.0 < self.node_count() && !self.free_nodes.contains(&node) {
            if let Some(port) = self.free_ports.pop() {
                self.ports[node.0].push(port);
                self.port_data[port.0] = (node, type_);
                self.port_identifiers[port.0] = ident;
                Ok(port)
            } else {
                let port = PortRef(self.port_count());

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
        // This won't panic because this is always `Some` on the user's end.
        let mut mem_cache = self.mem_cache.take().unwrap();
        let res = self.connect_(src, dst, &mut mem_cache);
        self.mem_cache = Some(mem_cache);
        res
    }

    fn connect_(
        &mut self,
        src: PortRef,
        dst: PortRef,
        mem_cache: &mut MemCache<N, P, PT>,
    ) -> Result<(), Error> {
        self.port_check(src)?;
        self.port_check(dst)?;

        for edge in self.incoming(dst) {
            if edge.src_port == src {
                // These two ports are already connected.
                return Ok(());
            } else {
                // Multiple sources cannot merge into the same destination port.
                return Err(Error::DstPortAlreadyConnected);
            }
        }

        let (src_node, src_type) = self.port_data[src.0];
        let (dst_node, dst_type) = self.port_data[dst.0];
        if src_type != dst_type {
            return Err(Error::InvalidPortType);
        }

        let mut queue = mem_cache.walk_queue.take().unwrap_or_default();
        let mut queued = mem_cache.walk_visited.take().unwrap_or_default();
        self.cycle_check(src_node, dst_node, &mut queue, &mut queued)?;
        mem_cache.walk_queue = Some(queue);
        mem_cache.walk_visited = Some(queued);

        let edge = Edge {
            src_node,
            src_port: src,
            dst_node,
            dst_port: dst,
            type_: src_type,
        };

        /* Maybe use the log crate for this to avoid polluting the user's output?
        println!(
            "connection {}.{} to {}.{} with edge: {:?}",
            self.node_name(src_node).unwrap(),
            self.port_name(src).unwrap(),
            self.node_name(dst_node).unwrap(),
            self.port_name(dst).unwrap(),
            edge
        );
        */

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

    pub fn port_ident(&self, port: PortRef) -> Result<&'_ PortIdent<P>, Error> {
        self.port_check(port)?;
        Ok(&self.port_identifiers[port.0])
    }

    pub fn node_ident(&self, node: NodeRef) -> Result<&'_ NodeIdent<N, PT>, Error> {
        self.node_check(node)?;
        Ok(&self.node_identifiers[node.0])
    }

    pub fn node_check(&self, n: NodeRef) -> Result<(), Error> {
        if n.0 < self.node_count() && !self.free_nodes.contains(&n) {
            Ok(())
        } else {
            Err(Error::NodeDoesNotExist)
        }
    }

    pub fn port_check(&self, p: PortRef) -> Result<(), Error> {
        if p.0 < self.port_count() && !self.free_ports.contains(&p) {
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
    fn cycle_check(
        &self,
        src: NodeRef,
        dst: NodeRef,
        queue: &mut VecDeque<NodeRef>,
        queued: &mut FnvHashSet<NodeRef>,
    ) -> Result<(), Error> {
        queue.clear();
        queued.clear();
        queue.push_back(dst);
        queued.insert(dst);

        while let Some(node) = queue.pop_front() {
            if node == src {
                return Err(Error::Cycle);
            }
            queue.extend(self.dependents(node).filter(|n| {
                if queued.contains(n) {
                    false
                } else {
                    queued.insert(*n);
                    true
                }
            }));
        }
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

    fn walk_mut(
        &mut self,
        root: NodeRef,
        mem_cache: &mut MemCache<N, P, PT>,
        queue: &mut VecDeque<NodeRef>,
        visited: &mut FnvHashSet<NodeRef>,
        mut f: impl FnMut(&mut Graph<N, P, PT>, NodeRef, &mut MemCache<N, P, PT>),
    ) {
        queue.clear();
        visited.clear();

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
            (&mut f)(self, node, mem_cache);
            queue.extend(self.dependents(node).filter(|n| !visited.contains(n)));
            visited.insert(node);
        }
    }

    pub fn compile(&mut self, root: NodeRef) -> &[Scheduled<N, P, PT>] {
        let solve_latency_requirements =
            |graph: &mut Graph<N, P, PT>,
             node: NodeRef,
             mem_cache: &mut MemCache<N, P, PT>,
             latencies: &mut Vec<u64>,
             deps: &mut Vec<NodeRef>,
             edges: &mut Vec<Edge<PT>>,
             insertions: &mut Vec<NodeRef>| {
                *latencies = graph
                    .dependencies(node)
                    .map(|n| mem_cache.all_latencies[n.0].unwrap() + graph.delays[n.0])
                    .collect::<Vec<_>>();

                *deps = graph.dependencies(node).collect();

                let max_latency = latencies.iter().max().copied().or(Some(0));

                mem_cache.all_latencies[node.0] = max_latency;

                insertions.clear();

                for (dep, latency) in deps.iter().zip(latencies.iter()) {
                    *edges = graph.edges[node.0]
                        .iter()
                        .filter(|e| e.src_node == *dep)
                        .map(|e| *e)
                        .collect();

                    for edge in edges.iter() {
                        let compensation = max_latency.unwrap() - latency;
                        if compensation == 0 {
                            continue;
                        }
                        if let Some((node, delay)) = mem_cache.delays.get_mut(edge) {
                            *delay = compensation;
                            graph.delays[node.0] = compensation;
                        } else {
                            graph.remove_edge(*edge).unwrap();

                            let delay_node = graph.node_(NodeIdent::DelayComp(edge.type_));

                            let delay_input = graph
                                .port_(delay_node, edge.type_, PortIdent::DelayComp)
                                .unwrap();
                            let delay_output = graph
                                .port_(delay_node, edge.type_, PortIdent::DelayComp)
                                .unwrap();

                            graph
                                .connect_(edge.src_port, delay_input, mem_cache)
                                .unwrap();
                            graph
                                .connect_(delay_output, edge.dst_port, mem_cache)
                                .unwrap();
                            graph.delays[delay_node.0] = compensation;
                            mem_cache.delays.insert(*edge, (delay_node, compensation));
                            insertions.push(delay_node);
                        }
                    }
                }
            };

        let solve_buffer_requirements =
            |graph: &Graph<N, P, PT>, node: NodeRef, mem_cache: &mut MemCache<N, P, PT>| {
                for port in &graph.ports[node.0] {
                    let (_, type_) = graph.port_data[port.0];
                    for output in graph.outgoing(*port) {
                        let (buffer, count) = mem_cache
                            .output_assignments
                            .entry((node, *port))
                            .or_insert((mem_cache.allocator.acquire(type_), 0));
                        *count += 1;
                        mem_cache
                            .input_assignments
                            .entry((output.dst_node, output.dst_port))
                            .or_insert(vec![])
                            .push(*buffer);
                    }
                    for input in graph.incoming(*port) {
                        let (buffer, count) = mem_cache
                            .output_assignments
                            .get_mut(&(input.src_node, input.src_port))
                            .expect("no output buffer assigned");
                        *count -= 1;
                        if *count == 0 {
                            mem_cache.allocator.release(*buffer);
                        }
                    }
                }
            };

        // This won't panic because this is always `Some` on the user's end.
        let mut mem_cache = self.mem_cache.take().unwrap();

        let mut scheduled = mem_cache.scheduled.take().unwrap_or_default();
        scheduled.clear();

        mem_cache.all_latencies.clear();
        mem_cache.all_latencies.resize(self.node_count(), None);
        mem_cache.delays.clear();

        mem_cache.allocator.clear();

        mem_cache.input_assignments.clear();
        mem_cache.output_assignments.clear();

        let mut scheduled_nodes = mem_cache.scheduled_nodes.take().unwrap_or_default();
        scheduled_nodes.clear();

        // This is arguably quite expensive, but the reason is that we don't want to add any delay
        // compensation nodes to the user's graph (We only want to add them to the schedule).
        // The fact that we're memory caching the previous graph should help some.
        let mut delay_comp_graph =
            if let Some(mut delay_comp_graph) = mem_cache.delay_comp_graph.take() {
                *delay_comp_graph = self.clone();
                delay_comp_graph
            } else {
                Box::new(self.clone())
            };

        let mut latencies = mem_cache.latencies.take().unwrap_or_default();
        let mut deps = mem_cache.deps.take().unwrap_or_default();
        let mut edges = mem_cache.edges.take().unwrap_or_default();
        let mut insertions = mem_cache.insertions.take().unwrap_or_default();
        let mut queue = mem_cache.walk_queue.take().unwrap_or_default();
        let mut visited = mem_cache.walk_visited.take().unwrap_or_default();

        delay_comp_graph.walk_mut(
            root,
            &mut mem_cache,
            &mut queue,
            &mut visited,
            |graph, node, mem_cache| {
                // Maybe use the log crate for this to avoid polluting the user's output?
                // println!("compiling {}", graph.node_name(node).unwrap());

                solve_latency_requirements(
                    graph,
                    node,
                    mem_cache,
                    &mut latencies,
                    &mut deps,
                    &mut edges,
                    &mut insertions,
                );
                for insertion in insertions.iter() {
                    solve_buffer_requirements(graph, *insertion, mem_cache);
                    scheduled_nodes.push(*insertion);
                }
                solve_buffer_requirements(graph, node, mem_cache);
                scheduled_nodes.push(node);
            },
        );

        for node in scheduled_nodes.iter().rev() {
            let node_ident = delay_comp_graph.node_identifiers[node.0].clone();

            let inputs: Vec<(PortIdent<P>, Vec<Buffer<PT>>)> = delay_comp_graph.ports[node.0]
                .iter()
                .filter_map(|port| {
                    mem_cache
                        .input_assignments
                        .get(&(*node, *port))
                        .map(|buffers| {
                            (
                                delay_comp_graph.port_identifiers[port.0].clone(),
                                buffers.clone(),
                            )
                        })
                })
                .collect::<Vec<_>>();
            let outputs: Vec<(PortIdent<P>, Buffer<PT>)> = delay_comp_graph.ports[node.0]
                .iter()
                .filter_map(|port| {
                    mem_cache
                        .output_assignments
                        .get(&(*node, *port))
                        .map(|(buffer, _)| {
                            (delay_comp_graph.port_identifiers[port.0].clone(), *buffer)
                        })
                })
                .collect::<Vec<_>>();

            scheduled.push(Scheduled {
                node: node_ident,
                inputs,
                outputs,
            });
        }

        mem_cache.scheduled = Some(scheduled);
        mem_cache.scheduled_nodes = Some(scheduled_nodes);
        mem_cache.delay_comp_graph = Some(delay_comp_graph);
        mem_cache.latencies = Some(latencies);
        mem_cache.deps = Some(deps);
        mem_cache.edges = Some(edges);
        mem_cache.insertions = Some(insertions);
        mem_cache.walk_queue = Some(queue);
        mem_cache.walk_visited = Some(visited);

        self.mem_cache = Some(mem_cache);

        self.mem_cache.as_ref().unwrap().scheduled.as_ref().unwrap()
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
            .port(a, DefaultPortType::Event, "events")
            .expect("port was not created");
        let a_out = graph
            .port(a, DefaultPortType::Audio, "output")
            .expect("port was not created");
        let b_in = graph
            .port(b, DefaultPortType::Audio, "input")
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
                .port(a, DefaultPortType::Audio, "output")
                .expect("could not create output port"),
            graph
                .port(b, DefaultPortType::Audio, "output")
                .expect("could not create output port"),
            graph
                .port(c, DefaultPortType::Audio, "output")
                .expect("could not create output port"),
        );

        let (a_in, b_in, c_in, d_in, d_in_2) = (
            graph
                .port(a, DefaultPortType::Audio, "input")
                .expect("could not create input"),
            graph
                .port(b, DefaultPortType::Audio, "input")
                .expect("could not create input"),
            graph
                .port(c, DefaultPortType::Audio, "input")
                .expect("could not create input"),
            graph
                .port(d, DefaultPortType::Audio, "input")
                .expect("could not create input"),
            graph
                .port(d, DefaultPortType::Audio, "input_2")
                .expect("could not create input"),
        );
        graph.set_delay(b, 2).expect("could not update delay of b");
        graph.set_delay(c, 5).expect("coudl not update delay of c");
        graph.connect(a_out, b_in).expect("could not connect");
        graph.connect(a_out, c_in).expect("could not connect");
        graph.connect(c_out, d_in).expect("could not connect");
        graph.connect(b_out, d_in_2).expect("could not connect");

        graph
            .connect(b_out, d_in)
            .expect_err("Desination ports should not be able to have multiple source ports.");

        graph
            .connect(b_out, a_in)
            .expect_err("Cycles should not be allowed");

        for entry in graph.compile(d) {
            println!("process {:?}:", entry.node);
            for (port, buffers) in entry.inputs.iter() {
                println!("    {:?} => ", port);
                for b in buffers {
                    println!("        {}", b.index);
                }
            }
            for (port, buf) in entry.outputs.iter() {
                println!("    {:?} => {}", port, buf.index);
            }
        }
    }
}
