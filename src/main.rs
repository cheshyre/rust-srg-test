extern crate cblas;
extern crate lapacke;
extern crate netlib_src;

extern crate ode_solvers;

extern nalgebra

use cblas::*;
// use lapacke::*;
use ode_solvers::dopri5::*;

// Struct definition
struct SRG {
    operator: Vec<f64>,
    lambda: f64,
    rhs: fn(_: f64, _: &Vec<f64>, _: &mut Vec<f64>),
}

// Instance methods
impl SRG {
    fn evolve(&mut self, new_lambda: f64) {
        let mut stepper = Dopri5::new(self.rhs, self.lambda, new_lambda,
                                      (self.lambda - new_lambda)/1.0e6,
                                      self.operator.to_vec(),
                                      1.0e-10, 1.0e-10);
        let res = stepper.integrate();
        match res {
            Ok(stats) => {
                stats.print();
                self.operator = stepper.y_out().to_vec();
                self.lambda = new_lambda;
            },
            Err(_) => {
                println!("Integration failed for some reason.");
            }
        }
    }
}

// Associated methods
impl SRG {
    fn init(operator: Vec<f64>, lambda: f64,
            rhs: fn(_: f64, _: &Vec<f64>, _: &mut Vec<f64>)) -> SRG {
        return SRG {
            operator,
            lambda,
            rhs
        };
    }

    fn rhs(lambda: f64, generator: &Vec<f64>, 
           operator: &Vec<f64>) -> Vec<f64> {
        let n = operator.len() as i32; 
        let mut hg = vec![0.0; n as usize];
        let mut hh = vec![0.0; n as usize];
        let mut dh = vec![0.0; n as usize];

        unsafe {
            dsymm(cblas::Layout::RowMajor, cblas::Side::Left,
                  cblas::Part::Upper, n, n, 1.0, &operator, n, &generator, n,
                  0.0, &mut hg, n);
            dsymm(cblas::Layout::RowMajor, cblas::Side::Left,
                  cblas::Part::Upper, n, n, 1.0, &operator, n, &operator, n,
                  0.0, &mut hh, n);
        }

        let hg = hg;
        let hh = hh;

        unsafe {
            // HHG
            dsymm(cblas::Layout::RowMajor, cblas::Side::Left,
                  cblas::Part::Upper, n, n, 1.0, &operator, n, &hg, n,
                  0.0, &mut dh, n);
            // - 2 HGH
            dsymm(cblas::Layout::RowMajor, cblas::Side::Right,
                  cblas::Part::Upper, n, n, -2.0, &operator, n, &hg, n,
                  1.0, &mut dh, n);
            // GHH
            dsymm(cblas::Layout::RowMajor, cblas::Side::Left,
                  cblas::Part::Upper, n, n, 1.0, &generator, n, &hh, n,
                  1.0, &mut dh, n);
        }

        let dh = dh;

        return dh;
    }
}


fn main() {
    // BLAS test
    
    let n = 2;
    let a = vec![
        1.0, 1.0,
        1.0, 1.0,
    ];
    let b = vec![
        1.0, 2.0,
        2.0, 3.0,
    ];
    let mut c = vec![
        1.0, 0.0,
        0.0, 1.0,
    ];
    unsafe {
        dsymm(cblas::Layout::RowMajor, cblas::Side::Left, cblas::Part::Upper,
              n, n, 1.0, &a, n, &b, n, 1.0, &mut c, n);
    }
    println!("{:?}", c);
    // let (m, n, k) = (2, 4, 3);
    // let a = vec![
    //     1.0, 4.0,
    //     2.0, 5.0,
    //     3.0, 6.0,
    // ];
    // let b = vec![
    //     1.0, 5.0,  9.0,
    //     2.0, 6.0, 10.0,
    //     3.0, 7.0, 11.0,
    //     4.0, 8.0, 12.0,
    // ];
    // let mut c = vec![
    //     2.0, 7.0,
    //     6.0, 2.0,
    //     0.0, 7.0,
    //     4.0, 2.0,
    // ];
    //
    // unsafe {
    //     dgemm(cblas::Layout::ColumnMajor, Transpose::None, Transpose::None,
    //           m, n, k, 1.0, &a, m, &b, k, 1.0, &mut c, m);
    // }
    //
    // println!("c = {:?}", c);
    //
    // assert!(
    //     c == vec![
    //         40.0,  90.0,
    //         50.0, 100.0,
    //         50.0, 120.0,
    //         60.0, 130.0,
    //     ]
    // );

    // LAPACK test
    // let n = 3;
    // let mut a = vec![
    //     3.0, 1.0, 1.0,
    //     1.0, 3.0, 1.0,
    //     1.0, 1.0, 3.0,
    // ];
    // let mut w = vec![0.0; n as usize];
    // let info;
    //
    // unsafe {
    //     info = dsyev(lapacke::Layout::ColumnMajor, b'V', b'U', n, &mut a, n, &mut w);
    // }
    //
    // assert!(info == 0);
    // for (one, another) in w.iter().zip(&[2.0, 2.0, 5.0]) {
    //     assert!((one - another).abs() < 1e-14);
    // }
}

fn gauss_legendre(n: i32, x1: f64, x2: f64) {
}
