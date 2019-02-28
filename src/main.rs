extern crate cblas;
extern crate lapacke;
extern crate netlib_src;

mod dopri5;
mod dop_shared;
mod butcher_tableau;
mod controller;

use cblas::*;
use lapacke::*;
use dopri5::*;
use std::f64;

// T_rel generator
#[derive(Clone)]
struct T_rel {
    kinetic_energy: Vec<f64>,
}

impl ODE for T_rel {
    fn rhs(&self, lambda: f64, state: &Vec<f64>, dstate: &mut Vec<f64>) {
        let change = srg_rhs(lambda, state, &self.kinetic_energy);
        for i in 0..dstate.len() {
            dstate[i] = change[i];
        }
    }
}


// Struct definition
struct SRG<O>
where
    O: ODE + Clone,
{
    operator: Vec<f64>,
    lambda: f64,
    // rhs: fn(_: f64, _: &Vec<f64>, _: &mut Vec<f64>),
    generator: O,
}

// Instance methods
impl<O> SRG<O>
where
    O: ODE + Clone,
{
    fn evolve(&mut self, new_lambda: f64) {
        let mut stepper = Dopri5::new(&self.generator, self.lambda, new_lambda,
                                      (self.lambda - new_lambda)/1.0e6,
                                      self.operator.to_vec(),
                                      1.0e-10, 1.0e-10);
        let res = stepper.integrate();
        match res {
            Ok(stats) => {
                // println!("{}", stats);
                stats.print();
                let res = stepper.y_out().last();
                // println!("{:?}", stepper.y_out().len());
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
impl<O> SRG<O>
where
    O: ODE + Clone,
{
    fn init(operator: &Vec<f64>, lambda: f64,
            generator: &O) -> SRG<O> {
        return SRG {
            operator: operator.to_vec(),
            lambda,
            generator: generator.clone(),
        };
    }
}

fn srg_rhs(lambda: f64, generator: &Vec<f64>,
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


fn main() {
    // Constants
    let hbarc: f64 = 1.0;
    let mass_nucleon: f64 = 1.0;
    let reduced_mass = mass_nucleon / 2.0;

    // Set array length
    let len: i32 = 128;

    // Get Gauss-Legendre nodes and weights
    let (nodes, weights) = gauss_legendre(len, 0.0, 10.0);

    // Get potential
    let mut pot = vec![0.0; (len * len) as usize];
    for i in 0..len {
        for j in 0..len {
            let index = i * len + j;
            pot[index as usize] = V_even(nodes[i as usize], nodes[j as usize]) * weights[i as usize].sqrt() * weights[j as usize].sqrt();
        }
    }
    let pot = pot;

    // println!("{:?}", pot);

    // Generate kinetic energy
    let mut kin = vec![0.0; (len * len) as usize];
    for i in 0..len {
        let index = i * len + i;
        kin[index as usize] = hbarc.powi(2) * nodes[i as usize].powi(2) / (2.0 * reduced_mass);
    }
    let kin = kin;

    // println!("{:?}", kin);

    let kin_generator = T_rel {
        kinetic_energy: kin.to_vec(),
    };

    let ham: Vec<f64> = pot.iter().zip(&kin).map(|(a, b)| a + b).collect();
    let mut ham2 = ham.to_vec();
    let mut w = vec![0.0; len as usize];
    let info;

    unsafe {
        info = dsyev(lapacke::Layout::RowMajor, b'V', b'U', len, &mut ham2, len, &mut w);
    }

    let mut min: f64 = 0.0;
    for x in w {
        min = min.min(x);
    }
    println!("Starting ground state energy: {}", min);

    let mut srg_evolver = SRG::init(&ham, 40.0, &kin_generator);
    
    // srg_evolver.evolve(30.0);
    srg_evolver.evolve(10.0);
    let new_ham = srg_evolver.operator.to_vec();
    let mut new_ham2 = new_ham.to_vec();
    let mut w2 = vec![0.0; len as usize];
    let info2;

    unsafe {
        info2 = dsyev(lapacke::Layout::RowMajor, b'V', b'U', len, &mut new_ham2, len, &mut w2);
    }

    let mut min2: f64 = 0.0;
    for x in &w2 {
        min2 = min2.min(*x);
    }
    println!("Final ground state energy: {}", min2);


    let new_pot : Vec<f64> = new_ham.iter().zip(&kin).map(|(a, b)| a - b).collect();
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
