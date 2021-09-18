use crate::port_type::PortType;
use std::cmp::PartialEq;
use std::marker::PhantomData;

/// An abstract buffer. Represents an allocated collection of data
#[derive(Copy, Clone, Debug)]
pub struct Buffer<P>
where
    P: PortType,
{
    /// The ID of the buffer
    pub buffer_id: usize,
    /// The type of data contained in this buffer
    pub port_type: P,
}

/// The buffer allocator is a stack based allocator that reuses buffers as needed
#[derive(Debug)]
pub struct BufferAllocator<P> {
    buffer_count_stacks: Vec<(usize, Vec<usize>)>,
    _phantom_port_type: PhantomData<P>,
}

impl<P> BufferAllocator<P>
where
    P: PortType + PartialEq,
{
    /// Empty/reset the allocator. Invalidates all previously allocated buffers.
    pub fn clear(&mut self) {
        for (c, s) in self.buffer_count_stacks.iter_mut() {
            *c = 0;
            s.clear();
        }
    }

    /// Acquire a new unique buffer.
    pub fn acquire(&mut self, type_: P) -> Buffer<P> {
        let type_index = type_.id();
        let (count, stack) = &mut self.buffer_count_stacks[type_index];

        if let Some(index) = stack.pop() {
            Buffer {
                buffer_id: index,
                port_type: type_,
            }
        } else {
            let buffer = Buffer {
                buffer_id: *count,
                port_type: type_,
            };
            *count += 1;
            buffer
        }
    }

    /// Release a buffer to be reused later.
    pub fn release(&mut self, ref_: Buffer<P>) {
        let type_index = ref_.port_type.id();
        let stack = &mut self.buffer_count_stacks[type_index].1;
        stack.push(ref_.buffer_id);
    }
}

impl<P> Default for BufferAllocator<P>
where
    P: PortType,
{
    fn default() -> Self {
        let num_types = P::NUM_TYPES;
        let mut buffer_count_stacks = Vec::<(usize, Vec<usize>)>::new();
        for _ in 0..num_types {
            buffer_count_stacks.push((0, Vec::new()));
        }
        Self {
            buffer_count_stacks,
            _phantom_port_type: PhantomData::default(),
        }
    }
}
