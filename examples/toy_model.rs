use rest_diis::prelude::{DIISAPI, DIISIncore, DIISIncoreFlagsBuilder, DIISSemiIncore, DIISSemiIncoreFlagsBuilder};
use rstsr::prelude::*;
use rstsr_openblas::DeviceOpenBLAS;

pub type Tsr<T> = Tensor<T, DeviceOpenBLAS, IxD>;

/// This problem is to solve the following linear equation:
///
/// `a @ x / diag(a) = b`
///
/// where `a` is symmetric matrix with large diagonal elements.
///
/// The iteration scheme is
///
/// `b' = a @ x / diag(a)`
/// `x (new) -> (b - b') + x`
///
/// For this case, the iterated `x` can be further extrapolated by DIIS.
/// - The naive iteration converges in 19 steps.
/// - The DIIS iteration converges in 5 steps.
fn main() {
    /* #region 1. initialization */

    // initialize matrix `a` and vector `b` of this problem
    #[rustfmt::skip]
    let vec_a = [
        [1.0, 0.1, 0.3, 0.2],
        [0.1, 1.5, 0.2, 0.1],
        [0.3, 0.2, 1.8, 0.2],
        [0.2, 0.1, 0.2, 1.3],
    ];
    let vec_b = vec![0.9, 1.5, 1.1, 0.4];

    let vec_a = vec_a.iter().flatten().copied().collect::<Vec<_>>();
    let device = DeviceOpenBLAS::default();
    let a = rt::asarray((vec_a, [4, 4], &device));
    let b = rt::asarray((vec_b, [4], &device));
    let d = a.diagonal(None);

    // the task to be solved in this problem
    // f(x) = a @ x / diag(a) - b
    let f = |x: &Tsr<f64>| &a % x - &b;

    /* #endregion 1. initialization */

    /* #region 2. solve by naive iteration */

    // the number of iterations and the tolerance
    let mut niter = 0;
    let maxiter = 20;
    let tol = 1e-7;

    // the initial guess
    let mut x = &b / &d;
    let mut x0;

    println!("== Naive iteration Start ==");
    while niter < maxiter && f(&x).l2_norm_all() > tol {
        niter += 1;
        x0 = x;
        let b0 = &a % &x0;
        x = (&b - b0) / &d + &x0;
        println!("  Naive iter {niter:2}, residue {:10.3e}", f(&x).l2_norm_all());
    }

    if niter == maxiter {
        panic!("The iteration did not converge.");
    } else {
        println!("Naive iteration converged in {:} steps.", niter);
        println!("  The solution is: {:10.6}", x);
        println!("  The residual is: {:10.3e}", f(&x).l2_norm_all());
    }

    /* #endregion 2. solve by naive iteration */

    /* #region 3. diis incore */

    // the number of iterations and the tolerance
    let mut niter = 0;
    let maxiter = 20;
    let tol = 1e-7;

    // the initial guess
    let mut x = &b / &d;
    let mut x0;

    // DIIS incore driver
    let diis_flags = DIISIncoreFlagsBuilder::default().build().unwrap();
    let mut diis = DIISIncore::<f64>::new(diis_flags, &device);
    // initial guess to DIIS
    diis.update(x.to_owned(), None, None);

    println!("== DIIS SemiIncore iteration Start ==");
    while niter < maxiter && f(&x).l2_norm_all() > tol {
        niter += 1;
        x0 = x;
        let b0 = &a % &x0;
        x = (&b - b0) / &d + &x0;
        x = diis.update(x, None, None);
        println!("  DIIS iter {niter:2}, residue {:10.3e}", f(&x).l2_norm_all());
    }

    if niter == maxiter {
        panic!("The iteration did not converge.");
    } else {
        println!("DIIS SemiIncore iteration converged in {:} steps.", niter);
        println!("  The solution is: {:10.6}", x);
        println!("  The residual is: {:10.3e}", f(&x).l2_norm_all());
    }

    /* #endregion 3. diis incore */

    /* #region 4. diis semi-incore dyn-object showcase */

    // the number of iterations and the tolerance
    let mut niter = 0;
    let maxiter = 20;
    let tol = 1e-7;

    // the initial guess
    let mut x = &b / &d;
    let mut x0;

    // DIIS incore driver
    let semi_incore = x.size() < 3; // assuming x > 3 requires semi-incore (disk I/O)
    let mut diis: Box<dyn DIISAPI<Tsr<f64>>> = match semi_incore {
        false => Box::new({
            let diis_flags = DIISIncoreFlagsBuilder::default().build().unwrap();
            DIISIncore::<f64>::new(diis_flags, &device)
        }),
        true => Box::new({
            let diis_flags = DIISSemiIncoreFlagsBuilder::default().build().unwrap();
            DIISSemiIncore::<f64>::new(diis_flags, &device)
        }),
    };
    // initial guess to DIIS
    diis.update(x.to_owned(), None, None);

    println!("== DIIS iteration Start ==");
    while niter < maxiter && f(&x).l2_norm_all() > tol {
        niter += 1;
        x0 = x;
        let b0 = &a % &x0;
        x = (&b - b0) / &d + &x0;
        x = diis.update(x, None, None);
        println!("  DIIS iter {niter:2}, residue {:10.3e}", f(&x).l2_norm_all());
    }

    if niter == maxiter {
        panic!("The iteration did not converge.");
    } else {
        println!("DIIS iteration converged in {:} steps.", niter);
        println!("  The solution is: {:10.6}", x);
        println!("  The residual is: {:10.3e}", f(&x).l2_norm_all());
    }

    /* #endregion 4. diis semi-incore dyn-object showcase */
}

#[test]
fn test() {
    main();
}
