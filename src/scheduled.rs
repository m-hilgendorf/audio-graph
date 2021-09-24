use crate::buffer_allocator::Buffer;
use crate::port_type::PortType;
use crate::vec::Vec;
use crate::{NodeRef, PortRef};

/// The compiled schedule
#[derive(Clone, Debug, Default)]
pub struct Schedule<PT>
where
    PT: PortType,
{
    /// The scheduled nodes in sequential order
    pub scheduled: Vec<ScheduledNode<PT>>,
    // TODO: Put the total number of each type of buffer the user needs
    // to allocate here for convenience?
}

/// A scheduled node is a node with buffers assigned to its input and output ports
#[derive(Clone, Debug)]
pub struct ScheduledNode<PT>
where
    PT: PortType,
{
    /// The node itself
    pub node: NodeRef,
    /// A list of input buffers assigned to ports
    pub inputs: Vec<(PortRef, Vec<(Buffer<PT>, Option<DelayCompInfo>)>)>,
    // A list of output buffers assigned to ports
    pub outputs: Vec<(PortRef, Buffer<PT>)>,
}

/// Information on delay compensation
#[derive(Clone, Debug)]
pub struct DelayCompInfo {
    /// The delay compensation in frames
    pub delay: u64,
    /// The source node of this connection
    pub source_node: NodeRef,
    /// The source port of this connection
    pub source_port: PortRef,
}
