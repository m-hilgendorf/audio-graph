use crate::buffer_allocator::Buffer;
use crate::port_type::PortType;
use crate::vec::Vec;
use std::fmt;

#[derive(Clone, Debug)]
pub struct Scheduled<N, P, PT>
where
    N: fmt::Debug + Clone,
    P: fmt::Debug + Clone,
    PT: PortType + PartialEq,
{
    pub node: N,
    pub inputs: Vec<(P, Vec<(Buffer<PT>, u64)>)>,
    pub outputs: Vec<(P, Buffer<PT>)>,
}
