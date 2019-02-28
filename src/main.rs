extern crate cblas;
extern crate lapacke;
extern crate netlib_src;

mod dopri5;
mod dop_shared;
mod butcher_tableau;
mod controller;

use cblas::*;
// use lapacke::*;
use dopri5::*;
use std::f64;

// Struct definition
struct SRG {
    operator: Vec<f64>,
    lambda: f64,
    // rhs: fn(_: f64, _: &Vec<f64>, _: &mut Vec<f64>),
    generator: Vec<f64>,
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
                println!("{}", stats);
                // stats.print();
                let res = stepper.y_out().last();
                match res {
                    Some(vec) => self.operator = vec.to_vec(),
                    None      => println!("What happened?"),
                }
                // self.operator = stepper.y_out().last().to_vec();
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
    fn init(operator: &Vec<f64>, lambda: f64,
            generator: &Vec<f64>) -> SRG {
        return SRG {
            operator: operator.to_vec(),
            lambda,
            generator: generator.to_vec(),
        };
    }

    fn rhs(lambda: f64, generator: &Vec<f64>,
           operator: &Vec<f64>) -> Vec<f64> {
        let n = operator.len() as i32;
        let mut hg = vec![0.0; n as usize];
        let mut hh = vec![0.0; n as usize];
        let mut dh = vec![0.0; n as usize];

        let n: i32 = (n as f64).sqrt() as i32;

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

        let dh = dh.iter().map(|x| x * (-4.0)/(lambda.powi(5))).collect();

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

fn V_even(p: f64, q: f64) -> f64 {
    V(p, q) + V(p, -1.0 * q)
}

fn V(p: f64, q: f64) -> f64 {
    let V1 = -2.0;
    let sig1: f64 = 0.8;
    V1 / (2.0 * f64::consts::PI) * (-1.0 * (p - q).powi(2) * sig1.powi(2) / 4.0).exp()
}

fn gauss_legendre(n: i32, x1: f64, x2: f64) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n as usize];
    let mut weights = vec![0.0; n as usize];

    let eps = 3.0e-11;

    let m: i32 = (((n as f64) + 1.0) / 2.0).floor() as i32;

    let xm: f64 = (x2 + x1) / 2.0;
    let xl: f64 = (x2 - x1) / 2.0;

    for i in 1..m + 1 {
        let mut z = (f64::consts::PI * ((i as f64) - 0.25) / ((n as f64) + 0.5)).cos();
        let mut z1 = 1.0;
        let mut pp = 0.0;

        while (z - z1).abs() >= eps {
            let mut p1 = 1.0;
            let mut p2 = 0.0;
            for j in 1..n + 1 {
                let p3 = p2;
                p2 = p1;
                p1 = ((2.0 * (j as f64) - 1.0) * z * p2 - ((j as f64) - 1.0) * p3) / (j as f64);
            }
            pp = (n as f64) * (z * p1 - p2) / (z * z - 1.0);
            z1 = z;
            z = z1 - p1 / pp;
        }
        nodes[(i - 1) as usize] = xm - xl * z;
        nodes[(n - i) as usize] = xm + xl * z;
        weights[(i - 1) as usize] = 2.0 * xl / ((1.0 - z * z) * pp * pp);
        weights[(n - i) as usize] = weights[(i - 1) as usize];
    }

    (nodes, weights)
}
