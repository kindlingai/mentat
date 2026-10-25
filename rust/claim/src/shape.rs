//! What a claim asks for: its sets, what holds between them, and how the
//! driver's node counts.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::Error;

/// A set has at most this many members. A count is expanded into one bundle
/// per member, so a larger one would only allocate.
pub const MAX_MEMBERS: usize = 4096;

/// What a set needs of the links between its own members.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Link {
    /// An operator-tagged fabric address, probe-confirmed between every pair.
    Rdma,
    /// Anything that replied to a probe.
    #[default]
    Any,
}

impl Link {
    pub fn parse(s: &str) -> Result<Link, String> {
        match s {
            "rdma" | "roce" | "fabric" => Ok(Link::Rdma),
            "ip" | "any" | "" => Ok(Link::Any),
            other => Err(format!("unknown link {other:?}, expected rdma or ip")),
        }
    }
}

/// How a set's members link to each other.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Layout {
    /// Every member links to every other: one island.
    #[default]
    Mesh,
    /// Each member links to the next, and the last to the first.
    Ring,
    /// Each member links to the next.
    Line,
    /// The first member, the hub, links to every other.
    Star,
    /// The first of mesh, ring, star and line that fits, else `Hops`.
    Any,
    /// What `Any` falls back to: every member links to every other, or
    /// reaches it through at most `max_hops` other members. A request
    /// cannot name it.
    Hops,
}

impl Layout {
    /// A layout a request may name. `hops` is refused.
    pub fn parse(s: &str) -> Result<Layout, String> {
        match s {
            "mesh" | "" => Ok(Layout::Mesh),
            "ring" => Ok(Layout::Ring),
            "line" => Ok(Layout::Line),
            "star" => Ok(Layout::Star),
            "any" => Ok(Layout::Any),
            other => Err(format!(
                "unknown layout {other:?}, expected mesh, ring, line, star or any"
            )),
        }
    }

    /// Whether the members come in an order that placement keeps.
    pub fn ordered(self) -> bool {
        matches!(self, Layout::Ring | Layout::Line | Layout::Star)
    }

    pub fn name(self) -> &'static str {
        match self {
            Layout::Mesh => "mesh",
            Layout::Ring => "ring",
            Layout::Line => "line",
            Layout::Star => "star",
            Layout::Any => "any",
            Layout::Hops => "hops",
        }
    }
}

impl Serialize for Layout {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.name())
    }
}

/// Reads every layout a view can name, `hops` included.
impl<'de> Deserialize<'de> for Layout {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Layout, D::Error> {
        let s = String::deserialize(d)?;
        match s.as_str() {
            "hops" => Ok(Layout::Hops),
            other => Layout::parse(other).map_err(serde::de::Error::custom),
        }
    }
}

/// Which ports a ring or line member reaches its two neighbours by.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Ports {
    /// Split where some ring or line allows it, else any.
    #[default]
    Auto,
    /// Different ports for the two neighbours.
    Split,
    /// One port for both neighbours.
    Same,
}

impl Ports {
    pub fn parse(s: &str) -> Result<Ports, String> {
        match s {
            "auto" | "" => Ok(Ports::Auto),
            "split" => Ok(Ports::Split),
            "same" => Ok(Ports::Same),
            other => Err(format!(
                "unknown ports {other:?}, expected auto, split or same"
            )),
        }
    }
}

/// Whether the driver's node shapes the answer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Driver {
    /// Rank 0 on the driver's node when it is a member, as vLLM requires.
    #[default]
    First,
    /// The answer and its order leave the driver out.
    Ignore,
}

impl Driver {
    pub fn parse(s: &str) -> Result<Driver, String> {
        match s {
            "first" | "" => Ok(Driver::First),
            "ignore" => Ok(Driver::Ignore),
            other => Err(format!(
                "unknown driver {other:?}, expected first or ignore"
            )),
        }
    }
}

/// One group of nodes the caller wants placed together.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Set {
    pub name: String,
    /// GPUs per member, one entry per node wanted.
    pub bundles: Vec<u32>,
    pub link: Link,
    pub layout: Layout,
    /// For a ring, a line or an `any` set. Any other layout takes `Auto`.
    pub ports: Ports,
    /// For an `any` set, how many other members a fallback member may reach
    /// another through. Only 1 for now, and None means 1.
    pub max_hops: Option<u32>,
    /// Pins the set to one GPU vendor. None lets the solver pick one, and it
    /// still picks only one: no collective spans vendors.
    pub vendor: Option<String>,
}

/// A requirement between two sets.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Between {
    pub from: String,
    pub to: String,
    pub link: Link,
}

/// A claim's request. `from_json` reads the shape of a `claim` message, and
/// `check` holds a typed shape to the same rules.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Shape {
    pub sets: Vec<Set>,
    pub between: Vec<Between>,
    pub driver: Driver,
}

impl Shape {
    /// Reads the shape of a `claim` message, as PROTOCOL.md describes it.
    pub fn from_json(shape: &Value) -> Result<Shape, Error> {
        read(shape).map_err(Error::BadShape)
    }

    /// Checks a typed shape against the rules of `from_json`.
    pub fn check(&self) -> Result<(), Error> {
        for s in &self.sets {
            check_set(s).map_err(Error::BadShape)?;
        }
        Ok(())
    }
}

fn check_set(s: &Set) -> Result<(), String> {
    let name = &s.name;
    if s.bundles.len() > MAX_MEMBERS {
        return Err(format!(
            "set {name:?} asks for {} nodes, more than {MAX_MEMBERS}",
            s.bundles.len()
        ));
    }
    if s.layout == Layout::Hops {
        return Err(format!(
            "set {name:?}: unknown layout \"hops\", expected mesh, ring, line, star or any"
        ));
    }
    if s.ports != Ports::Auto && !matches!(s.layout, Layout::Ring | Layout::Line | Layout::Any) {
        return Err(format!(
            "set {name:?}: ports applies to a ring, a line or any"
        ));
    }
    // Only one hop for now. A longer path relays through more boxes whose
    // ports carry their own ranks' traffic.
    match s.max_hops {
        None => {}
        Some(v) if s.layout != Layout::Any => {
            return Err(format!(
                "set {name:?}: max_hops applies to an any layout, got {v}"
            ))
        }
        Some(1) => {}
        Some(v) => return Err(format!("set {name:?}: max_hops must be 1 for now, got {v}")),
    }
    Ok(())
}

impl TryFrom<Value> for Shape {
    type Error = Error;

    fn try_from(v: Value) -> Result<Shape, Error> {
        Shape::from_json(&v)
    }
}

impl<'de> Deserialize<'de> for Shape {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Shape, D::Error> {
        let v = Value::deserialize(d)?;
        Shape::from_json(&v).map_err(serde::de::Error::custom)
    }
}

fn read(shape: &Value) -> Result<Shape, String> {
    let sets =
        shape["sets"]
            .as_array()
            .ok_or("shape needs a sets array")?
            .iter()
            .map(|s| {
                let name = s["name"].as_str().ok_or("a set needs a name")?.to_string();
                let bundles: Vec<u32> = match &s["bundles"] {
                    // Whole GPUs. `canonical` spells 1 and 1.0 alike. A fraction
                    // is refused.
                    Value::Array(a) => a
                        .iter()
                        .map(|b| {
                            b.as_f64()
                                .filter(|f| {
                                    f.is_finite()
                                        && *f >= 0.0
                                        && f.fract() == 0.0
                                        && *f <= u32::MAX as f64
                                })
                                .map(|f| f as u32)
                                .ok_or_else(|| {
                                    format!("set {name:?}: every bundle is a whole number of GPUs")
                                })
                        })
                        .collect::<Result<Vec<u32>, String>>()?,
                    // A count with no per-node figure means one GPU each, which
                    // is what a rank usually wants.
                    Value::Number(n) => {
                        let n = n.as_u64().unwrap_or(0);
                        if n > MAX_MEMBERS as u64 {
                            return Err(format!(
                                "set {name:?} asks for {n} nodes, more than {MAX_MEMBERS}"
                            ));
                        }
                        vec![1; n as usize]
                    }
                    _ => return Err(format!("set {name:?} needs bundles")),
                };
                let layout = Layout::parse(s["layout"].as_str().unwrap_or("mesh"))?;
                let ports = Ports::parse(s["ports"].as_str().unwrap_or("auto"))?;
                // `check_set` refuses every value but 1.
                let max_hops =
                    match &s["max_hops"] {
                        Value::Null => None,
                        v => Some(v.as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(
                            || format!("set {name:?}: max_hops must be 1 for now, got {v}"),
                        )?),
                    };
                let set = Set {
                    name,
                    bundles,
                    link: Link::parse(s["link"].as_str().unwrap_or("ip"))?,
                    layout,
                    ports,
                    max_hops,
                    vendor: s["vendor"]
                        .as_str()
                        .filter(|v| !v.is_empty())
                        .map(str::to_string),
                };
                check_set(&set)?;
                Ok(set)
            })
            .collect::<Result<Vec<_>, String>>()?;
    let between = shape["between"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|b| {
                    Ok(Between {
                        from: b["from"].as_str().ok_or("between needs from")?.to_string(),
                        to: b["to"].as_str().ok_or("between needs to")?.to_string(),
                        link: Link::parse(b["link"].as_str().unwrap_or("ip"))?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()
        })
        .transpose()?
        .unwrap_or_default();
    Ok(Shape {
        sets,
        between,
        driver: match &shape["driver"] {
            Value::Null => Driver::First,
            Value::String(s) => Driver::parse(s)?,
            v => return Err(format!("driver must be first or ignore, got {v}")),
        },
    })
}

/// A shape with whole numbers spelled one way.
///
/// A holder check that compares shapes as JSON would otherwise read `[1]`
/// and `[1.0]` as two shapes, and refuse the second holder of one claim.
pub fn canonical(v: &Value) -> Value {
    canon(v, false)
}

/// `inside_set` marks the object level where `bundles` may spell a count.
fn canon(v: &Value, inside_set: bool) -> Value {
    if inside_set {
        if let Some(n) = v.as_u64().filter(|n| *n <= MAX_MEMBERS as u64) {
            // `from_json` reads a count as that many single-GPU bundles, so
            // the two spellings of one request compare equal here too.
            return Value::Array((0..n).map(|_| Value::from(1u64)).collect());
        }
    }
    match v {
        Value::Number(n) => match n.as_f64() {
            Some(f) if f.is_finite() && f >= 0.0 && f.fract() == 0.0 && f <= u64::MAX as f64 => {
                Value::from(f as u64)
            }
            _ => v.clone(),
        },
        Value::Array(a) => Value::Array(a.iter().map(|x| canon(x, false)).collect()),
        Value::Object(o) => Value::Object(
            o.iter()
                .map(|(k, x)| (k.clone(), canon(x, k == "bundles")))
                .collect(),
        ),
        _ => v.clone(),
    }
}
