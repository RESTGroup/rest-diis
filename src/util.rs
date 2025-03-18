use crate::prelude_dev::*;

pub trait DIISAPI<TSR> {
    fn get_head(&mut self) -> Option<usize>;
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
    let _ = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_timer(time::LocalTime::rfc_3339())
        .with_max_level(tracing::Level::TRACE)
        .try_init();
    log::set_max_level(level);
}
