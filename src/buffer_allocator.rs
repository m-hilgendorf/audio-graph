///! Buffer allocation internals v2.
///!
///! The [BufferAllocator] is implemented using a list of
///! stacks, with one stack for each port type. When a
///! new buffer is required, we first try and pop a buffer
///! off the stack. If it none are available, we allocate
///! a new buffer. When a buffer is released, it is
///! pushed to the top of the corresponding stack.
///!
///! There is some additional bookkeeping required for the
///! buffers. [BufferRef]s contain a `ref_count` field which
///! tracks the number of edges that still need the buffer
///! to be alive before it can be safely released. The
///! `generation` field is kept around for visualization
///! of the assigned buffers during debugging.
///!
///! Finally, the engine using the graph needs to know
///! the maximum number of buffers for each port type
///! to allocate during its prepare for playback operation.
///! We track this by counting each time a new buffer
///! is allocated for a type in the `counts` list.
///!
///! Since it is not valid for the buffer allocator to
///! keep allocating after the `counts` field has been
///! consumed, we require consuming `self` to retrieve it.
use std::rc::Rc;

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

use crate::{BufferIdx, TypeIdx};

/// A reference to an abstract buffer during buffer allocation.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
pub struct BufferRef {
    /// The index of the buffer
    pub idx: BufferIdx,
    /// The type index of the buffer
    pub type_idx: TypeIdx,
    /// The generation, or the nth time this buffer has
    /// been assigned to a different edge in the graph.
    pub generation: usize,
}

impl BufferRef {
    /// Create a new BufferRef
    pub fn new(idx: BufferIdx, type_idx: TypeIdx, generation: usize) -> Self {
        Self {
            idx,
            type_idx,
            generation,
        }
    }
}

/// An allocator for managing and reusing [BufferRef]s.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct BufferAllocator {
    /// A list of free buffers that may be reallocated, one list for
    /// each different port type.
    pub free_lists: Vec<Vec<FreeListEntry>>,
    /// A list of the maximum number of buffers used for each port type.
    pub counts: Vec<usize>,
}

/// A small helper struct for tracking the index and generation
/// of data in the free lists
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct FreeListEntry {
    pub idx: BufferIdx,
    pub generation: usize,
}

impl BufferAllocator {
    /// Create a new allocator, `num_types` defines the number
    /// of buffer types we may allocate.
    pub fn new(num_types: usize) -> BufferAllocator {
        Self {
            free_lists: vec![vec![]; num_types],
            counts: vec![0; num_types],
        }
    }

    /// Acquire a new buffer with a given type index. Panics if
    /// the type index is out of bounds.
    pub fn acquire(&mut self, type_idx: TypeIdx) -> Rc<BufferRef> {
        let entry = self.free_lists[type_idx.0].pop().unwrap_or_else(|| {
            let idx = self.counts[type_idx.0] + 1;
            self.counts[type_idx.0] = idx;
            FreeListEntry {
                idx: BufferIdx(idx),
                generation: 0,
            }
        });
        Rc::new(BufferRef::new(entry.idx, type_idx, entry.generation))
    }

    /// Release a BufferRef.
    pub fn release(&mut self, buffer_ref: Rc<BufferRef>) {
        if Rc::strong_count(&buffer_ref) == 1 {
            self.free_lists[buffer_ref.type_idx.0].push(FreeListEntry {
                idx: buffer_ref.idx,
                generation: buffer_ref.generation + 1,
            });
        }
    }

    /// Consume the allocator to return the maximum number of buffers used
    /// for each type.
    pub fn num_buffers_per_type(self) -> Vec<usize> {
        self.counts
    }
}
