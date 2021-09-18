use crate::buffer_allocator::Buffer;
use crate::port_type::PortType;
use crate::vec::Vec;

/// A scheduled node is a node with buffers assigned to its input and output ports
#[derive(Clone, Debug)]
pub struct ScheduledNode<N, P, PT>
where PT: PortType
{
    /// The node itself
    pub node: N,
    /// A list of input buffers assigned to ports
    pub inputs: Vec<(P, Vec<(Buffer<PT>, u64)>)>,
    // A list of output buffers assigned to ports
    pub outputs: Vec<(P, Buffer<PT>)>,
}
