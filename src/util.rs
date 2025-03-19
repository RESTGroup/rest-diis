use crate::prelude_dev::*;
use ndarray::IxDyn;
use rstsr::prelude::*;
use rstsr_openblas::prelude_dev::DeviceCreationAnyAPI;

pub trait DIISAPI<TSR> {
    fn get_head(&self) -> Option<usize>;
    fn pop_head(&mut self, head: Option<usize>);
    fn insert(&mut self, vec: TSR, head: Option<usize>, err: Option<TSR>, iteration: Option<usize>);
    fn extrapolate(&mut self) -> TSR;
    fn update(&mut self, vec: TSR, err: Option<TSR>, iteration: Option<usize>) -> TSR;
}

/// DIIS pop strategy.
#[derive(Debug, Clone, Copy)]
pub enum DIISPopStrategy {
    /// Pop the vector with the largest iteration number.
    Iteration,

    /// Pop the vector with the largest diagonal element of the overlap matrix.
    ErrDiagonal,
}

/// Initialize logger.
///
/// This will only happen when no logger is initialized out of scope.
pub(crate) fn logger_init(verbose: usize) {
    let level = match verbose {
        0 => log::LevelFilter::Error,
        1 => log::LevelFilter::Warn,
        2 => log::LevelFilter::Info,
        3 => log::LevelFilter::Debug,
        4 => log::LevelFilter::Trace,
        _ => log::LevelFilter::Trace,
    };

    // initialize logger if not existed
    let _ = tracing_subscriber::fmt().with_ansi(false).with_timer(time::LocalTime::rfc_3339()).with_max_level(tracing::Level::TRACE).try_init();
    log::set_max_level(level);
}

pub(crate) fn ndarray_to_rstsr<T, B, D>(arr: ndarray::Array<T, D>, device: &B) -> Tensor<T, B>
where
    D: ndarray::Dimension,
    B: DeviceAPI<T, Raw = Vec<T>> + DeviceCreationAnyAPI<T>,
{
    let (arr, _) = arr.into_raw_vec_and_offset();
    rt::asarray((arr, device))
}
