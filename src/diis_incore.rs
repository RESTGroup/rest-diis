use crate::prelude_dev::*;

use num::Complex;
use num::complex::ComplexFloat;
use rstsr::prelude::*;
use rstsr_openblas::DeviceOpenBLAS;

pub type Tsr<T, D = IxD> = Tensor<T, DeviceOpenBLAS, D>;

#[derive(Builder)]
pub struct DIISIncoreFlags {
    /// Maximum number of vectors in the DIIS space.
    #[builder(default = "6")]
    pub space: usize,

    /// Minimum number of vectors in the DIIS space for extrapolation.
    #[builder(default = "2")]
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
    #[builder(default = "2")]
    pub verbose: usize,

    /// DIIS pop strategy.
    #[builder(default = "DIISPopStrategy::ErrDiagonal")]
    pub pop_strategy: DIISPopStrategy,

    /// Chunk size for inner dot.
    #[builder(default = "4194304")]
    pub chunk: usize,
}

pub struct DIISIncoreIntermediates<T> {
    /// The previous index of the inserted vector.
    prev: Option<usize>,

    /// Error vector overlap matrix for DIIS
    ///
    /// This overlap matrix follows convention that
    /// - the first row and column is auxiliary vector `[0, 1, ..., 1]`;
    /// - the rest of the matrix is the overlap matrix of the error vectors.
    ///
    /// Thus, internal index is 1-based.
    ovlp: Tsr<T>,

    /// The zero-th vector inserted to the DIIS space.
    ///
    /// This is only used when error is not given, and the error is defined as the vector difference itself.
    /// Since the zero-th iteration does not have vector difference, we need to store the zero-th vector.
    vec_prev: Option<Tsr<T>>,

    /// Error vectors for DIIS.
    ///
    /// Mapping: idx_internal -> err
    err_map: HashMap<usize, Tsr<T>>,

    /// Vectors to be extrapolated for DIIS
    ///
    /// Mapping: idx_internal -> vec
    vec_map: HashMap<usize, Tsr<T>>,

    /// Mapping of internal index.
    ///
    /// Mapping: idx_internal -> iteration
    niter_map: HashMap<usize, usize>,
}

pub struct DIISIncore<T> {
    pub flags: DIISIncoreFlags,
    pub intermediates: DIISIncoreIntermediates<T>,
}

/* #region logger */

/* #endregion */

#[allow(clippy::useless_conversion)]
#[duplicate::duplicate_item(T; [f32]; [f64]; [Complex::<f32>]; [Complex::<f64>])]
impl DIISIncore<T> {
    /// Initialize DIIS object.
    pub fn new(flags: DIISIncoreFlags, device: &DeviceOpenBLAS) -> Self {
        // initialize logger
        logger_init(flags.verbose);
        log::debug!("Initializing DIIS object.");

        // initialize intermediates
        let mut ovlp = rt::zeros(([flags.space + 1, flags.space + 1], device));
        ovlp.i_mut((0, 1..)).fill(T::from(1.0));
        ovlp.i_mut((1.., 0)).fill(T::from(1.0));
        let intermediates =
            DIISIncoreIntermediates { prev: None, ovlp, vec_prev: None, err_map: HashMap::new(), vec_map: HashMap::new(), niter_map: HashMap::new() };

        Self { flags, intermediates }
    }

    /// Compute the head by iteration number.
    pub fn get_head_by_iteration(&self) -> Option<usize> {
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
    pub fn get_head_by_diagonal(&self) -> Option<usize> {
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
    pub fn get_head(&self) -> Option<usize> {
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
            ovlp.i_mut((head, 1..)).fill(T::from(0.0));
            ovlp.i_mut((1.., head)).fill(T::from(0.0));
        }
    }

    /// Insert a vector to the DIIS space.
    pub fn insert(&mut self, vec: Tsr<T>, head: Option<usize>, err: Option<Tsr<T>>, iteration: Option<usize>) {
        // 1. unwrap head
        let head = head.or(self.get_head());
        let prev = self.intermediates.prev;

        // specical case: if head is the same to prev, then it means the last extrapolated vector has the maximum error;
        // then the function will infinite loops if remove the maximum error vector; so some code need to avoid this case.
        let head = if head == prev && head.is_some() {
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

        // get index that will be inserted
        let head = head.unwrap_or(1);
        log::trace!("DIIS current index to be inserted: {:?}", head);

        // pop head if necessary
        if self.intermediates.err_map.len() >= self.flags.space {
            self.pop_head(Some(head));
        }

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

        // 4. insert the vector and update information
        self.intermediates.err_map.insert(head, err);
        self.intermediates.vec_map.insert(head, vec);
        self.intermediates.niter_map.insert(head, iteration);
        self.intermediates.prev = Some(head);

        // 5. update the overlap matrix
        let ovlp = &mut self.intermediates.ovlp;
        let err_cur = self.intermediates.err_map.get(&head).unwrap();
        let num_space = self.intermediates.err_map.len();
        let err_list = (1..=num_space).into_iter().map(|i| self.intermediates.err_map.get(&i).unwrap()).collect::<Vec<_>>();
        let ovlp_cur = Self::incore_inner_dot(err_cur, &err_list, self.flags.chunk);
        ovlp.i_mut((head, 1..num_space + 1)).assign(&ovlp_cur);
        ovlp.i_mut((1..num_space + 1, head)).assign(&ovlp_cur.conj());
    }

    /// Extrapolate the vector from the DIIS space.
    pub fn extrapolate(&mut self) -> Tsr<T> {
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

        let eps = 30.0 * <T as ComplexFloat>::Real::EPSILON;
        if w.i(1..).min() < eps {
            log::warn!("DIIS extrapolation encounters singular overlap matrix.");
        }

        // set the small eigenvalues to inf, then take repciprocals
        let w = w.mapv(|x| {
            let val = if x.abs() < eps { 0.0 } else { 1.0 / x };
            T::from(val)
        });

        // g: [1, 0, 0, ..., 0]
        let mut g: Tsr<T> = rt::zeros(([num_space + 1], ovlp.device()));
        g[[0]] = T::from(1.0);

        // DIIS coefficients
        let c = (v.view() * w) % v.t() % g;
        log::debug!("DIIS coeff\n{:10.5}", c);

        // 3. extrapolate the vector
        let mut vec = self.intermediates.vec_map.get(&1).unwrap().zeros_like();
        for idx in 1..=num_space {
            let vec_idx = self.intermediates.vec_map.get(&idx).unwrap();
            vec += vec_idx * c[[idx]];
        }
        log::trace!("DIIS extrapolated vector\n{:10.5}", vec);

        vec
    }

    /// Update the DIIS space.
    pub fn update(&mut self, vec: Tsr<T>, err: Option<Tsr<T>>, iteration: Option<usize>) -> Tsr<T> {
        let time = std::time::Instant::now();

        self.insert(vec, None, err, iteration);
        let vec = self.extrapolate();
        self.intermediates.vec_prev = Some(vec.to_owned());

        let time_elapsed = time.elapsed();
        log::debug!("DIIS update time elapsed: {:?}", time_elapsed);

        vec
    }

    /// Perform inner dot for obtaining overlap.
    ///
    /// This performs `a.conj() % b` by chunk.
    /// Note that `a.conj()` will allocate a new memory buffer, so this also costs some L3 cache bandwidth.
    #[allow(clippy::useless_conversion)]
    pub fn incore_inner_dot(a: &Tsr<T>, b_list: &[&Tsr<T>], chunk: usize) -> Tsr<T> {
        let size = a.size();
        let nlist = b_list.len();
        let mut result = rt::zeros(([nlist], a.device()));

        for i in (0..size).step_by(chunk) {
            let a = a.slice(i..i + chunk).conj();
            for (n, b) in b_list.iter().enumerate() {
                let b = b.slice(i..i + chunk);
                result[[n]] += (&a % b).to_scalar();
            }
        }
        result
    }
}

#[duplicate::duplicate_item(T; [f32]; [f64]; [Complex::<f32>]; [Complex::<f64>])]
impl DIISAPI<Tsr<T>> for DIISIncore<T> {
    fn get_head(&self) -> Option<usize> {
        self.get_head()
    }

    fn pop_head(&mut self, head: Option<usize>) {
        self.pop_head(head);
    }

    fn insert(&mut self, vec: Tsr<T>, head: Option<usize>, err: Option<Tsr<T>>, iteration: Option<usize>) {
        self.insert(vec, head, err, iteration);
    }

    fn extrapolate(&mut self) -> Tsr<T> {
        self.extrapolate()
    }

    fn update(&mut self, vec: Tsr<T>, err: Option<Tsr<T>>, iteration: Option<usize>) -> Tsr<T> {
        self.update(vec, err, iteration)
    }
}
