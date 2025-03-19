use std::cell::UnsafeCell;

use crate::prelude_dev::*;

use hdf5_metno::Dataset;
use num::complex::ComplexFloat;
use rstsr::prelude::*;
use rstsr_openblas::DeviceOpenBLAS;

pub type Tsr<T, D = IxD> = Tensor<T, DeviceOpenBLAS, D>;
// use hdf5_metno::

#[derive(Builder)]
pub struct DIISSemiIncoreFlags {
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

    /// Path for scratch space.
    #[builder(default = "None", setter(into, strip_option))]
    pub scratch_path: Option<String>,

    /// Chunk size for inner dot.
    #[builder(default = "131072")]
    pub chunk: usize,
}

pub struct DIISSemiIncoreIntermediates<T> {
    /// Scratch file (for OS system) for DIIS.
    #[allow(dead_code)]
    scratch_file: tempfile::NamedTempFile,

    /// Scratch file (HDF5 handle) file.
    scratch_hdf5: hdf5_metno::File,

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
    vec_prev: Option<Dataset>,

    /// Error vectors for DIIS.
    ///
    /// Mapping: idx_internal -> err
    err_map: HashMap<usize, Dataset>,

    /// Vectors to be extrapolated for DIIS
    ///
    /// Mapping: idx_internal -> vec
    vec_map: HashMap<usize, Dataset>,

    /// Mapping of internal index.
    ///
    /// Mapping: idx_internal -> iteration
    niter_map: HashMap<usize, usize>,
}

pub struct DIISSemiIncore<T> {
    pub flags: DIISSemiIncoreFlags,
    pub intermediates: DIISSemiIncoreIntermediates<T>,
}

#[allow(clippy::useless_conversion)]
impl DIISSemiIncore<f64> {
    pub fn new(flags: DIISSemiIncoreFlags, device: &DeviceOpenBLAS) -> Self {
        // initialize logger
        logger_init(flags.verbose);
        log::debug!("Initializing DIIS object.");

        // generate scratch space
        // if no file is given, a temporary file will be crated.
        let temp_dir = std::env::var("REST_TMPDIR").map(|x| x.into()).unwrap_or(std::env::temp_dir());
        let scratch_file = tempfile::NamedTempFile::new_in(temp_dir).unwrap_or(tempfile::NamedTempFile::new().unwrap());
        log::debug!("DIIS scratch file: {:?}", scratch_file.path());

        let scratch_hdf5 = hdf5_metno::File::open_as(&scratch_file, hdf5_metno::OpenMode::ReadWrite).unwrap();

        // initialize intermediates
        let mut ovlp = rt::zeros(([flags.space + 1, flags.space + 1], device));
        ovlp.i_mut((0, 1..)).fill(f64::from(1.0));
        ovlp.i_mut((1.., 0)).fill(f64::from(1.0));
        let intermediates = DIISSemiIncoreIntermediates {
            scratch_file,
            scratch_hdf5,
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
            ovlp.i_mut((head, 1..)).fill(f64::from(0.0));
            ovlp.i_mut((1.., head)).fill(f64::from(0.0));
        }
    }

    /// Insert a vector to the DIIS space.
    pub fn insert(&mut self, vec: Tsr<f64>, head: Option<usize>, err: Option<Tsr<f64>>, iteration: Option<usize>) {
        // 1. unwrap head and something used in this function
        let head = head.or(self.get_head());
        let prev = self.intermediates.prev;
        let device = self.intermediates.ovlp.device().clone();

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

        // get index that will be inserted
        let head = head.unwrap_or(1);
        log::trace!("DIIS current index to be inserted: {:?}", head);

        // pop head if necessary
        if self.intermediates.err_map.len() >= self.flags.space {
            self.pop_head(Some(head));
        }

        // 2. prepare error and vector
        let scratch_hdf5 = &self.intermediates.scratch_hdf5;
        let hdf5_tag_err = format!("err{:}", head);
        let hdf5_tag_vec = format!("vec{:}", head);
        let size_vec = vec.size();
        let size_err = match err.as_ref() {
            Some(err) => err.size(),
            None => vec.size(),
        };

        // create current vec and err datasets
        println!("DEBUG: hdf5_tag_vec {:?}", scratch_hdf5.dataset(hdf5_tag_vec.as_str()));
        let vec_dataset = match scratch_hdf5.dataset(hdf5_tag_vec.as_str()) {
            Ok(dataset) => dataset,
            Err(_) => scratch_hdf5.new_dataset_builder().empty::<f64>().shape([size_vec]).create(hdf5_tag_vec.as_str()).unwrap(),
        };
        let err_dataset = match scratch_hdf5.dataset(hdf5_tag_err.as_str()) {
            Ok(dataset) => dataset,
            Err(_) => scratch_hdf5.new_dataset_builder().empty::<f64>().shape([size_err]).create(hdf5_tag_err.as_str()).unwrap(),
        };

        // reshape to 1-D
        let vec = vec.into_shape(-1);
        let err = err.map(|err| err.into_shape(-1));

        // write vec and err
        vec_dataset.write(&vec.raw()).unwrap();
        match err {
            // a. if error is given, reshape it to 1D
            Some(err) => {
                log::trace!("DIIS error is given by user.");
                err_dataset.write(&err.into_vec()).unwrap();
            },
            None => match &self.intermediates.vec_prev {
                Some(vec_prev_dataset) => {
                    log::trace!("DIIS error is computed from the previous vector.");
                    // &vec - vec_prev
                    let chunk = self.flags.chunk;
                    for i in (0..size_vec).step_by(self.flags.chunk) {
                        let i_max = std::cmp::min(i + chunk, size_vec);
                        let vec_prev = ndarray_to_rstsr(vec_prev_dataset.read_slice_1d::<f64, _>(i..i_max).unwrap(), &device);
                        let err = &vec.i(i..i_max) - vec_prev;
                        err_dataset.write_slice(err.raw(), i..i_max).unwrap();
                    }
                },
                None => {
                    log::trace!("This iteration will not be used for DIIS extrapolation.");
                    log::trace!("Save this vec for next error vector evaluation.");
                    self.intermediates.vec_prev = Some(scratch_hdf5.new_dataset_builder().with_data(&vec.raw()).create("vec_prev").unwrap());
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

        // 6. insert the vector and update information
        self.intermediates.err_map.insert(head, err_dataset);
        self.intermediates.vec_map.insert(head, vec_dataset);
        self.intermediates.niter_map.insert(head, iteration);
        self.intermediates.prev = Some(head);

        // 7. update the overlap matrix
        let ovlp = &mut self.intermediates.ovlp;
        let err_cur = self.intermediates.err_map.get(&head).unwrap();
        let num_space = self.intermediates.err_map.len();
        let err_list = (1..=num_space).into_iter().map(|i| self.intermediates.err_map.get(&i).unwrap()).collect::<Vec<_>>();
        let ovlp_cur = Self::semi_incore_inner_dot(err_cur, &err_list, ovlp.device(), self.flags.chunk);
        ovlp.i_mut((head, 1..num_space + 1)).assign(&ovlp_cur);
        ovlp.i_mut((1..num_space + 1, head)).assign(&ovlp_cur.conj());
    }

    /// Extrapolate the vector from the DIIS space.
    pub fn extrapolate(&mut self) -> Tsr<f64> {
        let device = self.intermediates.ovlp.device().clone();
        // 1. get the number of vectors in the DIIS space
        let num_space = self.intermediates.err_map.len();
        if num_space == 0 {
            // no vectors in the DIIS space
            if self.intermediates.vec_prev.is_some() {
                log::trace!("No vectors in the DIIS space.");
                log::trace!("Return the vector user previously requested.");
                return ndarray_to_rstsr(self.intermediates.vec_prev.as_ref().unwrap().read_1d::<f64>().unwrap(), &device);
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
            return ndarray_to_rstsr(self.intermediates.vec_map.get(&prev).unwrap().read_1d::<f64>().unwrap(), &device);
        }

        // 2. get the coefficients
        let ovlp = &self.intermediates.ovlp.i((..num_space + 1, ..num_space + 1));
        log::debug!("DIIS ovlp\n{:16.10}", ovlp);

        let (w, v) = rt::linalg::eigh(ovlp);
        log::debug!("DIIS w\n{:16.10}", w);

        let eps = 30.0 * <f64 as ComplexFloat>::Real::EPSILON;
        if w.i(1..).min() < eps {
            log::warn!("DIIS extrapolation encounters singular overlap matrix.");
        }

        // set the small eigenvalues to inf, then take repciprocals
        let w = w.mapv(|x| {
            let val = if x.abs() < eps { 0.0 } else { 1.0 / x };
            f64::from(val)
        });

        // g: [1, 0, 0, ..., 0]
        let mut g: Tsr<f64> = rt::zeros(([num_space + 1], ovlp.device()));
        g[[0]] = f64::from(1.0);

        // DIIS coefficients
        let c = (v.view() * w) % v.t() % g;
        log::debug!("DIIS coeff\n{:10.5}", c);

        // 3. extrapolate the vector
        let size_vec = self.intermediates.vec_map.get(&1).unwrap().size();
        let vec = rt::zeros(([size_vec], ovlp.device()));
        let vec_list = (1..=num_space).into_iter().map(|i| self.intermediates.vec_map.get(&i).unwrap()).collect::<Vec<_>>();
        let vec = Self::semi_incore_update_vec(vec, &vec_list, c.i(1..num_space + 1).into_owned(), self.flags.chunk);
        log::trace!("DIIS extrapolated vector\n{:10.5}", vec);

        vec
    }

    /// Update the DIIS space.
    pub fn update(&mut self, vec: Tsr<f64>, err: Option<Tsr<f64>>, iteration: Option<usize>) -> Tsr<f64> {
        let time = std::time::Instant::now();

        self.insert(vec, None, err, iteration);
        let vec = self.extrapolate();
        match &self.intermediates.vec_prev {
            Some(vec_prev) => {
                vec_prev.write(vec.raw()).unwrap();
            },
            None => {
                self.intermediates.vec_prev =
                    Some(self.intermediates.scratch_hdf5.new_dataset_builder().with_data(&vec.raw()).create("vec_prev").unwrap());
            },
        }

        let time_elapsed = time.elapsed();
        log::debug!("DIIS update time elapsed: {:?}", time_elapsed);

        vec
    }

    /// Perform inner dot for obtaining overlap.
    ///
    /// This performs `a.conj() % b` by chunk.
    pub fn semi_incore_inner_dot(a: &Dataset, b_list: &[&Dataset], device: &DeviceOpenBLAS, chunk: usize) -> Tsr<f64> {
        #[cfg(feature = "sequential_io")]
        {
            let size = a.size();
            let nlist = b_list.len();
            let mut result: Tsr<f64> = rt::zeros(([nlist], device));

            for i in (0..size).step_by(chunk) {
                let i_max = std::cmp::min(i + chunk, size);
                let a = ndarray_to_rstsr(a.read_slice_1d::<f64, _>(i..i_max).unwrap(), device);
                let b_list = b_list.iter().map(|b| ndarray_to_rstsr(b.read_slice_1d::<f64, _>(i..i_max).unwrap(), device)).collect::<Vec<_>>();

                for (n, b) in b_list.iter().enumerate() {
                    result[[n]] += (&a % b).to_scalar();
                }
            }
            result
        }

        #[cfg(not(feature = "sequential_io"))]
        {
            let size = a.size();
            let nlist = b_list.len();
            let result: Tsr<f64> = rt::zeros(([nlist], device));

            std::thread::scope(|s| {
                let mut task = s.spawn(|| {});
                for i in (0..size).step_by(chunk) {
                    let i_max = std::cmp::min(i + chunk, size);
                    let a = ndarray_to_rstsr(a.read_slice_1d::<f64, _>(i..i_max).unwrap(), device);
                    let b_list = b_list.iter().map(|b| ndarray_to_rstsr(b.read_slice_1d::<f64, _>(i..i_max).unwrap(), device)).collect::<Vec<_>>();
                    let result = result.view();

                    task.join().unwrap();
                    task = s.spawn(move || {
                        let mut result = unsafe { result.force_mut() };
                        for (n, b) in b_list.iter().enumerate() {
                            result[[n]] += (&a % b).to_scalar();
                        }
                    });
                }
            });
            result
        }
    }

    pub fn semi_incore_update_vec(a: Tsr<f64>, b_list: &[&Dataset], c: Tsr<f64>, chunk: usize) -> Tsr<f64> {
        #[cfg(feature = "sequential_io")]
        {
            let mut a = a;
            let size = a.size();
            for i in (0..size).step_by(chunk) {
                let i_max = std::cmp::min(i + chunk, size);
                let b_list = b_list.iter().map(|b| ndarray_to_rstsr(b.read_slice_1d::<f64, _>(i..i_max).unwrap(), a.device())).collect::<Vec<_>>();
                for (n, b) in b_list.iter().enumerate() {
                    *&mut a.i_mut(i..i_max) += b * c[[n]];
                }
            }
            a
        }

        #[cfg(not(feature = "sequential_io"))]
        {
            let size = a.size();
            let device = a.device().clone();

            std::thread::scope(|s| {
                let mut task = s.spawn(|| {});
                for i in (0..size).step_by(chunk) {
                    let i_max = std::cmp::min(i + chunk, size);
                    let b_list = b_list.iter().map(|b| ndarray_to_rstsr(b.read_slice_1d::<f64, _>(i..i_max).unwrap(), &device)).collect::<Vec<_>>();
                    let a_sliced = a.i(i..i_max);
                    let c = c.view();

                    task.join().unwrap();
                    task = s.spawn(move || {
                        for (n, b) in b_list.iter().enumerate() {
                            let mut a_sliced = unsafe { a_sliced.force_mut() };
                            a_sliced += b * c[[n]];
                        }
                    });
                }
            });
            a
        }
    }
}

#[duplicate::duplicate_item(T; [f64])]
impl DIISAPI<Tsr<T>> for DIISSemiIncore<T> {
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

#[test]
fn playground() {
    let flags = DIISSemiIncoreFlagsBuilder::default().build().unwrap();
    let device = DeviceOpenBLAS::default();
    let diis = DIISSemiIncore::new(flags, &device);

    let scratch_hdf5 = diis.intermediates.scratch_hdf5;
    let db = scratch_hdf5.new_dataset_builder();
    let dataset = db.with_data(&[1.0, 2.0, 3.0]).create("1").unwrap();
    dataset.write_slice(&[1.0, 2.0], ndarray::s![1..3]).unwrap();
    let data = dataset.read_1d::<f64>().unwrap();
    println!("{:?}", data);
    let dataset = scratch_hdf5.new_dataset_builder().empty::<f64>().shape([3]).create("2").unwrap();
    let data = dataset.read_1d::<f64>().unwrap();
    println!("{:?}", data);
    let dataset = scratch_hdf5.new_dataset_builder().empty::<f64>().shape([3]).create("2").unwrap();
    let data = dataset.read_1d::<f64>().unwrap();
    println!("{:?}", data);
}
