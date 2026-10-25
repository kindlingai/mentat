//! Solves a mentat claim: places named sets of nodes on a probed cluster so
//! that each set's members link the way the set asks, as a mesh, a ring, a
//! line or a star.
//!
//! `solve` takes typed input and returns a typed `View`. The input and the
//! view serialize to the JSON forms in mentat's PROTOCOL.md. The
//! `mentat-claim` binary reads and writes those forms.
//!
//! The answer depends only on the input. Two callers holding the same
//! topology get the same view.

pub mod island;
pub mod shape;
mod solve;
pub mod topology;
pub mod view;

#[cfg(test)]
mod test_alloc;

pub use island::Island;
pub use shape::{Between, Driver, Layout, Link, Ports, Set, Shape};
pub use topology::{islands, Node, Port, Probe, Topology};
pub use view::View;

/// A node's id. Any string works, as long as each node has one.
pub type NodeId = String;

/// Why a claim has no view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The shape breaks a rule of the request format.
    BadShape(String),
    /// No placement on the topology meets the shape.
    Unsolvable(String),
}

impl Error {
    /// The code mentat refuses a claim with.
    pub fn code(&self) -> &'static str {
        match self {
            Error::BadShape(_) => "bad_shape",
            Error::Unsolvable(_) => "unsolvable_claim",
        }
    }

    pub fn message(&self) -> &str {
        match self {
            Error::BadShape(m) | Error::Unsolvable(m) => m,
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for Error {}

/// Places `shape` on `topology`.
///
/// `driver` is the node of the process that will hold rank 0, as vLLM
/// requires. With the shape's default `driver: first`, a star puts its hub
/// there when it can, and an `any` set passes over a star with the node on a
/// leaf and a line with the node inside it. A ring and a line keep their
/// solved order, and the caller rotates a ring or reverses a line to start
/// at the node. With `driver: ignore`, `driver` has no effect.
pub fn solve(topology: &Topology, shape: &Shape, driver: Option<&str>) -> Result<View, Error> {
    shape.check()?;
    let g = solve::Graph::new(topology);
    let req = solve::Request::new(shape, driver);
    solve::solve(&g, &req)
        .map(|s| s.view(&g))
        .map_err(Error::Unsolvable)
}
