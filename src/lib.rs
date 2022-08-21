#![allow(clippy::type_complexity)]

mod buffer_allocator;
mod graph_helper;
mod input_ir;
mod output_ir;

pub mod error;
pub mod graph_ir;

pub use graph_helper::*;
pub use graph_ir::*;
pub use input_ir::*;
pub use output_ir::*;

#[cfg(test)]
mod tests {
    use super::*;
    use fnv::{FnvHashMap, FnvHashSet};

    #[test]
    fn simplest_graph_compile_test() {
        let nodes = vec![
            Node {
                id: 0.into(),
                inputs: vec![Port {
                    id: 0.into(),
                    type_idx: 0.into(),
                }],
                outputs: vec![Port {
                    id: 1.into(),
                    type_idx: 0.into(),
                }],
                latency: 0,
            },
            Node {
                id: 1.into(),
                inputs: vec![Port {
                    id: 0.into(),
                    type_idx: 0.into(),
                }],
                outputs: vec![Port {
                    id: 1.into(),
                    type_idx: 0.into(),
                }],
                latency: 0,
            },
        ];

        let edges = vec![Edge {
            id: 0.into(),
            src_node: nodes[0].id,
            src_port: nodes[0].outputs[0].id,
            dst_node: nodes[1].id,
            dst_port: nodes[1].inputs[0].id,
        }];

        let schedule = compile(1, nodes.clone(), edges);

        dbg!(&schedule);

        assert_eq!(schedule.schedule.len(), 2);
        assert_eq!(schedule.delays.len(), 0);
        assert_eq!(schedule.num_buffers.len(), 1);
        assert!(schedule.num_buffers[0] > 0);

        let edge_src_buffer_id = if let ScheduleEntry::Node(scheduled_node) = &schedule.schedule[0]
        {
            verify_scheduled_node(scheduled_node, &nodes[0], &[(nodes[0].inputs[0].id, true)]);
            scheduled_node.output_buffers[0].buffer_index
        } else {
            panic!("first entry not a node");
        };
        let edge_dst_buffer_id = if let ScheduleEntry::Node(scheduled_node) = &schedule.schedule[1]
        {
            verify_scheduled_node(scheduled_node, &nodes[1], &[(nodes[1].inputs[0].id, false)]);
            scheduled_node.input_buffers[0].buffer_index
        } else {
            panic!("second entry not a node");
        };

        assert_eq!(edge_src_buffer_id, edge_dst_buffer_id);
    }

    fn verify_scheduled_node(
        scheduled_node: &ScheduledNode,
        src_node: &Node,
        in_ports_that_should_clear: &[(PortID, bool)],
    ) {
        assert_eq!(scheduled_node.id, src_node.id);
        assert_eq!(scheduled_node.latency, src_node.latency);
        assert_eq!(scheduled_node.input_buffers.len(), src_node.inputs.len());
        assert_eq!(scheduled_node.output_buffers.len(), src_node.outputs.len());

        assert_eq!(in_ports_that_should_clear.len(), src_node.inputs.len());

        struct PortCheckVal {
            should_clear: bool,
            assigned_a_buffer: bool,
            type_index: TypeIdx,
        }

        let mut port_check: FnvHashMap<PortID, PortCheckVal> = FnvHashMap::default();
        for (port_id, should_clear) in in_ports_that_should_clear.iter() {
            let mut found_port_type = None;
            for port in src_node.inputs.iter() {
                if port.id == *port_id {
                    found_port_type = Some(port.type_idx);
                    break;
                }
            }
            let type_index = found_port_type.unwrap();

            assert!(port_check
                .insert(
                    *port_id,
                    PortCheckVal {
                        should_clear: *should_clear,
                        assigned_a_buffer: false,
                        type_index
                    }
                )
                .is_none());
        }
        for port in src_node.outputs.iter() {
            assert!(port_check
                .insert(
                    port.id,
                    PortCheckVal {
                        should_clear: false,
                        assigned_a_buffer: false,
                        type_index: port.type_idx
                    }
                )
                .is_none());
        }

        let mut buffer_alias_check: FnvHashSet<BufferIdx> = FnvHashSet::default();

        for buffer in scheduled_node
            .input_buffers
            .iter()
            .chain(scheduled_node.output_buffers.iter())
        {
            assert!(buffer_alias_check.insert(buffer.buffer_index));

            let port_check_val = port_check
                .get_mut(&buffer.port_id)
                .expect("Buffer assigned to port that doesn't exist in node");

            assert_eq!(buffer.type_index, port_check_val.type_index);
            assert_eq!(buffer.should_clear, port_check_val.should_clear);

            if port_check_val.assigned_a_buffer {
                panic!("More than one buffer assigned to the same port in node");
            }

            port_check_val.assigned_a_buffer = true;
        }

        for port_check_val in port_check.values() {
            assert!(port_check_val.assigned_a_buffer);
        }
    }
}
