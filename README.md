# Rust SRG Test

This was a brief test of what a Rust implementation of the SRG would look like. The ODE solver is adapted from the `ode_solvers` crate, which did not fully suit my needs. Linear algebra operations are done via calls to CBLAS and LAPACKE routines. Running this yourself will require you to set this up yourself. Good luck, because it sucks.

The ODE solver is slow and has some bugs. In particular it performs worse than other implementations of the same algorithm when encountering stiffness. It is slow in part because I simply replaced every move with a copy which for O(10k) vectors becomes pretty expensive.

### Brief Conclusion

This allowed me to learn a lot about Rust, in particular traits and ownership. I have some ideas for how a proper general Dopri5 implementation would look in terms of interfaces, which I may make a personal project. However, I think the existing tools in Rust for numerical computing do not suit my needs, and reimplementing things from other languages is not the best use of my time, so I will stick to Python (C++ when performance is an issue) for my current work.
