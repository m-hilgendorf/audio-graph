use smallvec::SmallVec;

use crate::buffer_allocator::Buffer;
use crate::port_type::PortType;

/// The compiled schedule
#[derive(Clone, Debug, Default)]
pub struct Schedule<N, P, PT>
where
    PT: PortType,
{
    /// The scheduled nodes in sequential order
    pub scheduled: Vec<ScheduledNode<N, P, PT>>,
    // TODO: Put the total number of each type of buffer the user needs
    // to allocate here for convenience?
}

/// A scheduled node is a node with buffers assigned to its input and output ports
#[derive(Clone, Debug)]
pub struct ScheduledNode<N, P, PT>
where
    PT: PortType,
{
    /// The node itself
    pub node: N,

    // Note that these SmallVecs will actually be allocated on the heap (inside a
    // std Vec of ScheduledNodes). SmallVecs can still be useful here since the
    // majority of nodes on average will have less the 5 inputs/outputs and atleast
    // one or two buffesr per input port, so all that memory allocated upfront will
    // avoid a bunch of smaller allocations when filling in the schedule later.
    /// A list of input buffers assigned to ports
    pub inputs: SmallVec<[(P, SmallVec<[(Buffer<PT>, Option<DelayCompInfo<N, P>>); 2]>); 4]>,
    // A list of output buffers assigned to ports
    pub outputs: SmallVec<[(P, Buffer<PT>); 4]>,
}

/// Information on delay compensation
#[derive(Clone, Debug)]
pub struct DelayCompInfo<N, P> {
    /// The delay compensation in frames
    pub delay: u64,
    /// The source node of this connection
    pub source_node: N,
    /// The source port of this connection
    pub source_port: P,
}
