use crate::buffer_allocator::{Buffer, BufferAllocator};
use crate::graph::{NodeRef, PortRef};
use crate::port_type::PortType;
use crate::vec::Vec;
use fnv::{FnvHashMap, FnvHashSet};
use std::collections::VecDeque;

#[derive(Debug)]
pub(crate) struct HeapStore<PT>
where
    PT: PortType,
{
    pub walk_queue: Option<VecDeque<NodeRef>>,
    pub walk_indegree: Option<FnvHashMap<NodeRef, usize>>,
    pub cycle_queued: Option<FnvHashSet<NodeRef>>,
    pub latencies: Vec<u64>,
    pub all_latencies: Vec<Option<u64>>,
    pub deps: Vec<NodeRef>,
    pub allocator: BufferAllocator<PT>,
    pub delay_comps: Option<FnvHashMap<(PortRef, PortRef), u64>>,
    pub input_assignments:
        FnvHashMap<(NodeRef, PortRef), Vec<(Buffer<PT>, PortRef, PortRef, NodeRef)>>,
    pub output_assignments: FnvHashMap<(NodeRef, PortRef), (Buffer<PT>, usize)>,
    pub scheduled_nodes: Option<Vec<NodeRef>>,
}

impl<PT> Default for HeapStore<PT>
where
    PT: PortType,
{
    fn default() -> Self {
        Self {
            walk_queue: Some(VecDeque::new()),
            walk_indegree: Some(FnvHashMap::default()),
            cycle_queued: Some(FnvHashSet::default()),
            latencies: Vec::new(),
            all_latencies: Vec::new(),
            deps: Vec::new(),
            allocator: BufferAllocator::default(),
            delay_comps: Some(FnvHashMap::default()),
            input_assignments: FnvHashMap::default(),
            output_assignments: FnvHashMap::default(),
            scheduled_nodes: Some(Vec::new()),
        }
    }
}
