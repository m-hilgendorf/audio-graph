use std::fmt;

/// An abstract port type.
pub trait PortType
where
    Self: Default + fmt::Debug + Clone + Copy + Eq + std::hash::Hash,
{
    /// The total number of all port types
    const NUM_TYPES: usize;
    /// The id of this port type.
    fn id(&self) -> usize;
}

/// A default port type for general purpose applications
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DefaultPortType {
    /// Audio ports
    Audio,
    /// Event (non-audio) ports
    Event,
}

impl Default for DefaultPortType {
    fn default() -> Self {
        DefaultPortType::Audio
    }
}

impl PortType for DefaultPortType {
    const NUM_TYPES: usize = 2;
    fn id(&self) -> usize {
        *self as usize
    }
}
