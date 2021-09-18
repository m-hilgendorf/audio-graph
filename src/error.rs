use std::fmt;
use std::error;

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

impl error::Error for Error {}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result {
        let error_string = 
            match self {
                Error::NodeDoesNotExist => "Node does not exist",
                Error::PortDoesNotExist => "Port does not exist",
                Error::Cycle => "Cycle detected",
                Error::ConnectionDoesNotExist => "Connection does not exist",
                Error::RefDoesNotExist => "Reference does not exist",
                Error::InvalidPortType => "Cannot connect ports. Ports are a different type"
            };
        write!(f, "Audio Graph Error: {}.", error_string)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errror_strings () {
        let errors = [
            Error::NodeDoesNotExist,
            Error::PortDoesNotExist,
            Error::Cycle,
            Error::ConnectionDoesNotExist,
            Error::RefDoesNotExist,
            Error::InvalidPortType
        ];
        let strings = [
            "Audio Graph Error: Node does not exist.",
            "Audio Graph Error: Port does not exist.",
            "Audio Graph Error: Cycle detected.",
            "Audio Graph Error: Connection does not exist.",
            "Audio Graph Error: Reference does not exist.",
            "Audio Graph Error: Cannot connect ports. Ports are a different type.",
        ];
        for (e, s) in errors.iter().zip(strings.iter()) {
            assert_eq!(e.to_string(), *s);
        }
    }
}
