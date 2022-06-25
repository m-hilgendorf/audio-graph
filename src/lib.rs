#![allow(clippy::type_complexity)]
mod buffer_allocator;
mod buffer_allocator2;
mod cache;
mod error;
mod graph;
pub mod graph_ir;
pub mod input_ir;
pub mod output_ir;
mod port_type;
mod scheduled;
mod vec;

pub use error::Error;
pub use graph::{Graph, NodeRef, PortRef};
pub use port_type::{DefaultPortType, PortType};
pub use scheduled::ScheduledNode;

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
                .port(d, DefaultPortType::Audio, "d_input_1")
                .expect("could not create input"),
            graph
                .port(d, DefaultPortType::Audio, "d_input_2")
                .expect("could not create input"),
        );
        graph.set_delay(b, 2).expect("could not update delay of b");
        graph.set_delay(c, 5).expect("could not update delay of c");
        graph.connect(a_out, b_in).expect("could not connect");
        graph.connect(a_out, c_in).expect("could not connect");
        graph.connect(b_out, d_in).expect("could not connect");
        graph.connect(c_out, d_in).expect("could not connect");
        graph.connect(b_out, d_in_2).expect("could not connect");

        graph
            .connect(b_out, a_in)
            .expect_err("Cycles should not be allowed");

        let mut last_node = None;
        for entry in graph.compile() {
            println!("process {:?}:", entry.node);
            for (port, buffers) in entry.inputs.iter() {
                println!("    {} => ", port);

                if *port == "d_input_1" {
                    for (b, delay_comp) in buffers {
                        println!("        index: {}", b.buffer_id);
                        println!("        delay_comp: {}", delay_comp);
                    }

                    // One of the buffers should have a delay_comp of 0, and one
                    // should have a delay_comp of 3
                    assert!(
                        (buffers[0].1 == 0 && buffers[1].1 == 3)
                            || (buffers[0].1 == 3 && buffers[1].1 == 0)
                    )
                } else {
                    for (b, delay_comp) in buffers {
                        println!("        index: {}", b.buffer_id);
                        println!("        delay_comp: {}", delay_comp);

                        if *port == "d_input_2" {
                            assert_eq!(*delay_comp, 3);
                        } else {
                            assert_eq!(*delay_comp, 0);
                        }
                    }
                }
            }
            for (port, buffer) in entry.outputs.iter() {
                println!("    {:?} => {}", port, buffer.buffer_id);
            }
            last_node = Some(entry.node.clone());
        }
        assert!(matches!(last_node, Some("D")));
    }
}
