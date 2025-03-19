#![allow(clippy::deref_addrof)]
pub mod diis_incore;
pub mod diis_semi_incore;
pub mod util;

pub mod prelude {
    pub use crate::diis_incore::{DIISIncore, DIISIncoreFlags, DIISIncoreFlagsBuilder, DIISIncoreFlagsBuilderError};
    pub use crate::diis_semi_incore::{DIISSemiIncore, DIISSemiIncoreFlags, DIISSemiIncoreFlagsBuilder, DIISSemiIncoreFlagsBuilderError};
    pub use crate::util::{DIISAPI, DIISPopStrategy};
    pub(crate) use crate::util::{logger_init, ndarray_to_rstsr};
}

pub mod prelude_dev {
    pub use crate::prelude::*;
    pub use derive_builder::Builder;
    pub use std::collections::HashMap;
    pub use tracing_subscriber::fmt::time;
}
