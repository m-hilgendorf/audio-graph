/// Errors that may arise when compiling a graph
#[derive(Debug, Clone, Copy)]
pub enum Error {
    /// A node does not exist, arises during connections
    NodeDoesNotExist,
    /// A port does not exist, arises during connections
    PortDoesNotExist,
    /// A cycle was detected when connecting ports
    Cycle,
    /// Attempted to disconnect ports that are not connected
    ConnectionDoesNotExist,
    /// The reference does not exist or was invalid
    RefDoesNotExist,
    /// The port types mismatched, arises during connections
    InvalidPortType,
}

impl std::error::Error for Error {}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NodeDoesNotExist => write!(f, "Audio graph node does not exist"),
            Error::PortDoesNotExist => write!(f, "Audio graph port does not exist"),
            Error::Cycle => write!(f, "Audio graph cycle detected"),
            Error::ConnectionDoesNotExist => write!(f, "Audio graph connection does not exist"),
            Error::RefDoesNotExist => write!(f, "Audio graph reference does not exist"),
            Error::InvalidPortType => write!(
                f,
                "Cannot connect audio graph ports. Ports are a different type"
            ),
        }
    }
}
