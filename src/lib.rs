use derive_builder::Builder;
use std::collections::HashMap;
use tracing_subscriber::fmt::time;

use rstsr::prelude::*;
use rstsr_openblas::DeviceOpenBLAS;

pub type Tsr<T, D = IxD> = Tensor<T, DeviceOpenBLAS, D>;

#[derive(Builder)]
pub struct DIISFlags {
    /// Maximum number of vectors in the DIIS space.
    #[builder(default = "6")]
    pub space: usize,

    /// Minimum number of vectors in the DIIS space for extrapolation.
    #[builder(default = "3")]
    pub min_space: usize,

    /// Verbose level.
    ///
    /// Verbose level is defined as follows (also see [`log` crate](https://docs.rs/log/latest/log/)):
    ///
    /// - 0: Error
    /// - 1: Warn
    /// - 2: Info
    /// - 3: Debug
    /// - 4: Trace
    #[builder(default = "3")]
    pub verbose: usize,

    /// DIIS pop strategy.
    #[builder(default = "DIISPopStrategy::ErrDiagonal")]
    pub pop_strategy: DIISPopStrategy,
}

/// DIIS pop strategy.
#[derive(Debug, Clone, Copy)]
pub enum DIISPopStrategy {
    /// Pop the vector with the largest iteration number.
    Iteration,

    /// Pop the vector with the largest diagonal element of the overlap matrix.
    ErrDiagonal,
}

pub struct DIISIntermediates {
    /// The previous index of the inserted vector.
    prev: Option<usize>,

    /// Error vector overlap matrix for DIIS
    ///
    /// This overlap matrix follows convention that
    /// - the first row and column is auxiliary vector `[0, 1, ..., 1]`;
    /// - the rest of the matrix is the overlap matrix of the error vectors.
    ///
    /// Thus, internal index is 1-based.
    ovlp: Tsr<f64>,

    /// The zero-th vector inserted to the DIIS space.
    ///
    /// This is only used when error is not given, and the error is defined as the vector difference itself.
    /// Since the zero-th iteration does not have vector difference, we need to store the zero-th vector.
    vec_prev: Option<Tsr<f64>>,

    /// Error vectors for DIIS.
    ///
    /// Mapping: idx_internal -> err
    err_map: HashMap<usize, Tsr<f64>>,

    /// Vectors to be extrapolated for DIIS
    ///
    /// Mapping: idx_internal -> vec
    vec_map: HashMap<usize, Tsr<f64>>,

    /// Mapping of internal index.
    ///
    /// Mapping: idx_internal -> iteration
    niter_map: HashMap<usize, usize>,
}

pub struct DIIS {
    pub flags: DIISFlags,
    pub intermediates: DIISIntermediates,
}

/* #region logger */

/// Initialize logger.
///
/// This will only happen when no logger is initialized out of scope.
fn logger_init(verbose: usize) {
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

/* #endregion */

impl DIIS {
    /// Initialize DIIS object.
    pub fn new(flags: DIISFlags, device: &DeviceOpenBLAS) -> Self {
        // initialize logger
        logger_init(flags.verbose);
        log::debug!("Initializing DIIS object.");

        // initialize intermediates
        let mut ovlp = rt::zeros(([flags.space + 1, flags.space + 1], device));
        ovlp.i_mut((0, 1..)).fill(1.0);
        ovlp.i_mut((1.., 0)).fill(1.0);
        let intermediates = DIISIntermediates {
            prev: None,
            ovlp,
            vec_prev: None,
            err_map: HashMap::new(),
            vec_map: HashMap::new(),
            niter_map: HashMap::new(),
        };

        Self { flags, intermediates }
    }

    /// Compute the head by iteration number.
    pub fn get_head_by_iteration(&mut self) -> Option<usize> {
        let cur_space = self.intermediates.err_map.len();
        let head = if cur_space == 0 {
            // No previously inserted vectors.
            None
        } else if cur_space < self.flags.space {
            // The head to be inserted is has not been filled.
            let idx_next = cur_space + 1;
            Some(idx_next)
        } else {
            // We assume that number of vectors should be larger than 1, so unwrap here.
            let key = self.intermediates.niter_map.iter().min_by(|a, b| a.1.cmp(b.1));
            let idx_next = *key.unwrap().0;
            Some(idx_next)
        };
        log::trace!("DIIS head by iteration: {:?}", head);
        head
    }

    /// Compute the head by diagonal of overlap.
    pub fn get_head_by_diagonal(&mut self) -> Option<usize> {
        let cur_space = self.intermediates.err_map.len();
        let head = if cur_space == 0 {
            // No previously inserted vectors.
            None
        } else if cur_space < self.flags.space {
            // The head to be inserted is has not been filled.
            let idx_next = cur_space + 1;
            Some(idx_next)
        } else {
            // We assume that number of vectors should be larger than 1, so unwrap here.
            // Find the index of largest diagonal element of the overlap matrix.
            let ovlp = &self.intermediates.ovlp;
            let diagonal = ovlp.diagonal(None).abs();
            let idx_argmax = diagonal.argmax();
            if idx_argmax == 0 || idx_argmax > self.flags.space {
                // Error of vectors is too small, which is virtually impossible.
                // Evaluate next index by iteration number, but we will not raise error here.
                return self.get_head_by_iteration();
            }
            Some(idx_argmax)
        };
        log::trace!("DIIS head by maximum of diagonal: {:?}", head);
        head
    }

    /// Compute the next index to be inserted.
    pub fn get_head(&mut self) -> Option<usize> {
        // get the head if not internally defined
        // if head is not defined, we will define it by the given strategy.
        match self.flags.pop_strategy {
            DIISPopStrategy::Iteration => self.get_head_by_iteration(),
            DIISPopStrategy::ErrDiagonal => self.get_head_by_diagonal(),
        }
    }

    /// Pop the head index and update the internal overlap of error vectors.
    ///
    /// - `None` will pop the internal evaluated index.
    /// - `Some(idx)` will pop the given index with the given stragety defined in flag `DIISFlags::pop_stragety`.
    ///
    /// Note that we do not assume the index is valid, so we will not raise error if the index is invalid.
    pub fn pop_head(&mut self, head: Option<usize>) {
        // Find the index to be popped.
        let head = head.or(self.get_head());

        if let Some(head) = head {
            log::trace!("DIIS pop head: {:?}", head);

            // Actually pop the vector.
            self.intermediates.err_map.remove(&head);
            self.intermediates.vec_map.remove(&head);
            self.intermediates.niter_map.remove(&head);

            // Update the overlap matrix.
            let ovlp = &mut self.intermediates.ovlp;
            ovlp.i_mut((head, 1..)).fill(0.0);
            ovlp.i_mut((1.., head)).fill(0.0);
        }
    }

    /// Insert a vector to the DIIS space.
    pub fn insert(
        &mut self,
        vec: Tsr<f64>,
        head: Option<usize>,
        err: Option<Tsr<f64>>,
        iteration: Option<usize>,
    ) {
        // 1. unwrap head
        let head = head.or(self.get_head());
        let prev = self.intermediates.prev;

        // specical case: if head is the same to prev, then it means the last extrapolated vector has the maximum error;
        // then the function will infinite loops if remove the maximum error vector; so some code need to avoid this case.
        let head = if head == prev {
            log::warn!(concat!(
                "DIIS error seems not good.\n",
                "The DIIS head is the same to the previous vector.\n",
                "It means that the last extrapolated vector has the maximum error.\n",
                "It is certainly not desired in DIIS, you may want to make a double-check.\n",
                "We will remove the vector with earliest iteration instead of the largest error norm."
            ));
            self.get_head_by_iteration()
        } else {
            head
        };

        // 2. prepare error and vector
        let vec = vec.into_shape(-1);
        let err = match err {
            // a. if error is given, reshape it to 1D
            Some(err) => {
                log::trace!("DIIS error is given by user.");
                err.into_shape(-1)
            },
            None => match &self.intermediates.vec_prev {
                Some(vec_prev) => {
                    log::trace!("DIIS error is computed from the previous vector.");
                    &vec - vec_prev
                },
                None => {
                    log::trace!("This iteration will not be used for DIIS extrapolation.");
                    log::trace!("Save this vec for next error vector evaluation.");
                    self.intermediates.vec_prev = Some(vec);
                    return;
                },
            },
        };

        // 3. prepare iteration
        let iteration = iteration.unwrap_or({
            match prev {
                Some(prev) => self.intermediates.niter_map.get(&prev).unwrap() + 1,
                None => 0,
            }
        });
        log::trace!("DIIS internal iteration counter: {:?}", iteration);

        // 4. pop head if necessary
        if self.intermediates.err_map.len() >= self.flags.space {
            self.pop_head(head);
        }

        // 5. get index that will be inserted
        let cur = head.unwrap_or(1);
        log::trace!("DIIS current index to be inserted: {:?}", cur);

        // 6. insert the vector and update information
        self.intermediates.err_map.insert(cur, err);
        self.intermediates.vec_map.insert(cur, vec);
        self.intermediates.niter_map.insert(cur, iteration);
        self.intermediates.prev = Some(cur);

        // 7. update the overlap matrix
        #[allow(unused_mut)]
        let mut ovlp = &mut self.intermediates.ovlp;
        let err_cur = self.intermediates.err_map.get(&cur).unwrap();
        let num_space = self.intermediates.err_map.len();
        for idx in 1..=num_space {
            let err = self.intermediates.err_map.get(&idx).unwrap();
            let ovlp_val = (err_cur % err).to_scalar();
            ovlp[[cur, idx]] = ovlp_val;
            ovlp[[idx, cur]] = ovlp_val;
        }
    }

    /// Extrapolate the vector from the DIIS space.
    pub fn extrapolate(&mut self) -> Tsr<f64> {
        // 1. get the number of vectors in the DIIS space
        let num_space = self.intermediates.err_map.len();
        if num_space == 0 {
            // no vectors in the DIIS space
            if self.intermediates.vec_prev.is_some() {
                log::trace!("No vectors in the DIIS space.");
                log::trace!("Return the vector user previously requested.");
                return self.intermediates.vec_prev.as_ref().unwrap().to_owned();
            } else {
                // no vectors in the DIIS space and no zero-th vector
                // this is considered as error
                panic!("No vectors in the DIIS space. This may be an internal error.");
            }
        }

        // 1.5 not enough vectors for extrapolation
        if num_space < self.flags.min_space {
            log::trace!("DIIS space is not large enough to be extrapolated.");
            log::trace!("Return the vector user previously requested.");
            let prev = self.intermediates.prev.unwrap();
            return self.intermediates.vec_map.get(&prev).unwrap().to_owned();
        }

        // 2. get the coefficients
        let ovlp = &self.intermediates.ovlp.i((..num_space + 1, ..num_space + 1));
        log::debug!("DIIS ovlp\n{:16.10}", ovlp);

        let (w, v) = rt::linalg::eigh(ovlp);
        log::debug!("DIIS w\n{:16.10}", w);

        if w.i(1..).min() < 1.0e-14 {
            log::warn!("DIIS extrapolation encounters singular overlap matrix.");
        }

        // set the small eigenvalues to inf, then take repciprocals
        let w = w.mapv(|x| if x.abs() < 1.0e-14 { 0.0 } else { 1.0 / x });

        // g: [1, 0, 0, ..., 0]
        let mut g: Tsr<f64> = rt::zeros(([num_space + 1], ovlp.device()));
        g[[0]] = 1.0;

        // DIIS coefficients
        let c = (v.view() * w) % v.t() % g;
        log::debug!("DIIS coeff\n{:10.5}", c);

        // 3. extrapolate the vector
        let mut vec = self.intermediates.vec_map.get(&1).unwrap().zeros_like();
        for idx in 1..=num_space {
            let vec_idx = self.intermediates.vec_map.get(&idx).unwrap();
            let c_idx = c[[idx]];
            vec += vec_idx * c_idx;
        }
        log::trace!("DIIS extrapolated vector\n{:10.5}", vec);

        vec
    }

    /// Update the DIIS space.
    pub fn update(
        &mut self,
        vec: Tsr<f64>,
        err: Option<Tsr<f64>>,
        iteration: Option<usize>,
    ) -> Tsr<f64> {
        let time = std::time::Instant::now();

        self.insert(vec, None, err, iteration);
        let vec = self.extrapolate();
        self.intermediates.vec_prev = Some(vec.to_owned());

        let time_elapsed = time.elapsed();
        log::debug!("DIIS update time elapsed: {:?}", time_elapsed);

        vec
    }
}
