//! Capability trait and registration macro for the Unified Capability Cortex.
//!
//! Each capability is a typed, zero-copy bridge between the MLX lazy graph
//! (which produces action tensors) and native Rust functions (which perform
//! work and write results back into the Fabric).
//!
//! # Yellowpaper Invariants
//!
//! - All argument and result types implement `Pod + Zeroable`
//! - Each capability has an exhaustive `CapabilityId`
//! - Dispatch is type-safe: `fabric_view` is lifetime-bound to the call
//! - Results are written into pre-allocated observation buffer slices

use crate::types::{CapabilityId, CortexError};
use bytemuck::{Pod, Zeroable};

/// The core capability trait.
///
/// Every capability registered with the Cortex must implement this trait.
/// The trait enforces:
/// - Fixed, `Pod`-compatible argument and result types (zero-copy serialization)
/// - A unique `CapabilityId` for dispatch
/// - A pure `execute` function that takes args and a mutable fabric view
///
/// # Example
///
/// ```rust,ignore
/// pub struct VectorSearch;
///
/// impl Capability for VectorSearch {
///     type Args = VectorSearchArgs;
///     type Result = VectorSearchResult;
///     const ID: CapabilityId = CapabilityId::VectorSearch;
///
///     fn execute(args: Self::Args, fabric_view: &mut [u8]) -> Result<Self::Result, CortexError> {
///         // Perform semantic search via MLX matmul
///         Ok(VectorSearchResult { ... })
///     }
/// }
/// ```
pub trait Capability: Send + Sync + 'static {
    /// The argument type, serialized as raw bytes from the action tensor
    type Args: Pod + Zeroable + Copy;
    /// The result type, written back into the observation buffer
    type Result: Pod + Zeroable + Copy;
    /// Unique identifier for this capability
    const ID: CapabilityId;

    /// Execute the capability.
    ///
    /// # Arguments
    ///
    /// - `args`: Deserialized from the Fabric's argument region
    /// - `arg_buf`: Read-only slice of the Fabric's argument region (for dynamic sized args like strings)
    /// - `res_buf`: Mutable slice of the observation buffer for writes
    ///
    /// # Returns
    ///
    /// The result to be written back into the Fabric, or a `CortexError`.
    fn execute(args: Self::Args, arg_buf: &[u8], res_buf: &mut [u8]) -> Result<Self::Result, CortexError>;
}

/// Type-erased handler function signature used in the Cortex dispatch map.
///
/// Takes raw byte slices for args and results, performs internal Pod casts.
pub type HandlerFn =
    Box<dyn Fn(&[u8], &mut [u8]) -> Result<(), CortexError> + Send + Sync>;

/// Generate a type-erased handler from a `Capability` implementor.
///
/// This function creates a closure that:
/// 1. Casts the arg bytes to the capability's Args type via `bytemuck`
/// 2. Calls `execute`
/// 3. Casts the result back to bytes and writes to the result slice
pub fn make_handler<C: Capability>() -> HandlerFn {
    Box::new(|arg_bytes: &[u8], res_bytes: &mut [u8]| {
        // Validate arg size
        let arg_size = std::mem::size_of::<C::Args>();
        if arg_bytes.len() < arg_size {
            return Err(CortexError::ArgOutOfBounds {
                offset: 0,
                size: arg_bytes.len(),
            });
        }

        // Zero-copy cast from bytes → Args
        let args: &C::Args = bytemuck::from_bytes(&arg_bytes[..arg_size]);

        // Execute the capability
        let result = C::execute(*args, arg_bytes, res_bytes)?;

        // Write result back to result slice
        let result_bytes = bytemuck::bytes_of(&result);
        let result_size = result_bytes.len();
        if res_bytes.len() < result_size {
            return Err(CortexError::ResultOutOfBounds {
                offset: 0,
                size: res_bytes.len(),
            });
        }
        res_bytes[..result_size].copy_from_slice(result_bytes);

        Ok(())
    })
}

/// Macro to register a capability with the Cortex.
///
/// Usage: `register_capability!(cortex, MyCapability);`
#[macro_export]
macro_rules! register_capability {
    ($cortex:expr, $cap:ty) => {
        $cortex.register(<$cap as $crate::capability::Capability>::ID,
                         $crate::capability::make_handler::<$cap>());
    };
}
