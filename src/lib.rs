pub mod incore_diis;
pub mod util;

pub mod prelude {
    pub use crate::incore_diis::{
        DIISIncore, DIISIncoreFlags, DIISIncoreFlagsBuilder, DIISIncoreFlagsBuilderError,
    };
    pub(crate) use crate::util::logger_init;
    pub use crate::util::{DIISAPI, DIISPopStrategy};
}

pub mod prelude_dev {
    pub use crate::prelude::*;
    pub use derive_builder::Builder;
    pub use std::collections::HashMap;
    pub use tracing_subscriber::fmt::time;
}
