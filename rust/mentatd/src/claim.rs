//! Matching a requested shape against the cluster's probed topology.
//!
//! Placement today settles one question: put these bundles inside one fabric
//! island. It cannot express "a cabled pair here and a cabled pair there, with
//! only IP between them", which is what pipeline parallel over two
//! tensor-parallel pairs needs. A request here names its sets, reports which of them need a fabric,
//! and reports what has to hold between them.
//!
//! The answer names nodes and, for every link it relied on, the address to
//! dial and the interface it sits on. A caller binding NCCL needs both and
//! only the node knows the second.
//!
//! Every collection walked here is sorted. Two daemons holding the same view
//! must return the same answer, or one name would mean two placements.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Value};

use crate::island::Island;
use crate::state::NodeId;

/// What a set needs of the links between its own members.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Link {
    /// An operator-tagged fabric address, probe-confirmed between every pair.
    Rdma,
    /// Anything that replied to a probe.
    Any,
}

/// How a set's members link to each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Every member links to every other: one island.
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

/// Which ports a ring or line member reaches its two neighbours by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ports {
    /// Split where some ring or line allows it, else any.
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Driver {
    /// Rank 0 on the driver's node when it is a member, as vLLM requires.
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

impl Link {
    pub fn parse(s: &str) -> Result<Link, String> {
        match s {
            "rdma" | "roce" | "fabric" => Ok(Link::Rdma),
            "ip" | "any" | "" => Ok(Link::Any),
            other => Err(format!("unknown link {other:?}, expected rdma or ip")),
        }
    }
}

/// A shape with whole numbers spelled one way.
///
/// The holder check compares shapes as JSON, so `[1]` and `[1.0]` would
/// otherwise name two shapes and refuse the second holder of one claim.
pub fn canonical(v: &Value) -> Value {
    canon(v, false)
}

/// `inside_set` marks the object level where `bundles` may spell a count.
fn canon(v: &Value, inside_set: bool) -> Value {
    if inside_set {
        if let Some(n) = v.as_u64() {
            // `parse` reads a count as that many single-GPU bundles, so the
            // two spellings of one request compare equal here too.
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

/// One group of nodes the caller wants placed together.
#[derive(Clone, Debug)]
pub struct SetReq {
    pub name: String,
    /// GPUs per member, one entry per node wanted.
    pub bundles: Vec<f64>,
    pub link: Link,
    pub layout: Layout,
    pub ports: Ports,
    /// For an `any` set, how many other members a fallback member may
    /// reach another through.
    pub max_hops: usize,
    /// Pins the set to one GPU vendor. Empty lets the daemon pick one, and
    /// it still picks only one: no collective spans vendors.
    pub vendor: String,
}

/// A requirement between two sets.
#[derive(Clone, Debug)]
pub struct BetweenReq {
    pub from: String,
    pub to: String,
    pub link: Link,
}

#[derive(Clone, Debug)]
pub struct Request {
    pub sets: Vec<SetReq>,
    pub between: Vec<BetweenReq>,
    /// The node of the client that sent the claim. A star puts its hub
    /// there when it can, since vLLM puts rank 0 on the driver's node.
    pub driver: Option<NodeId>,
    /// With `Ignore`, the solve leaves `driver` out.
    pub driver_rule: Driver,
}

/// One address a node listens on.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Port {
    pub addr: String,
    /// None for an address from configuration.
    pub iface: Option<String>,
    pub tags: Vec<String>,
}

impl Port {
    fn rdma(&self) -> bool {
        self.tags.iter().any(|t| t == "rdma")
    }
}

/// What the matcher reads. Assembled from the daemon's merged view, and
/// built directly in tests.
#[derive(Clone, Debug, Default)]
pub struct Topology {
    /// Fabric islands, as the island module derived them.
    pub islands: Vec<Island>,
    /// Every address each node listens on.
    pub ports: BTreeMap<NodeId, Vec<Port>>,
    /// Address pairs that replied to a probe, with the round trip observed.
    /// Undirected: a one-way link is not something to place on.
    pub links: BTreeMap<(String, String), u64>,
    /// Linked pairs whose ARP entry named a third box. That box forwards
    /// between the pair. A mesh counts the pair, and a ring or a line,
    /// built from cables, leaves it out.
    pub forwarded: BTreeSet<(String, String)>,
    /// GPUs free on each node right now.
    pub free_gpus: BTreeMap<NodeId, f64>,
    /// Hostname per node, held through for the answer to be readable.
    pub hosts: BTreeMap<NodeId, String>,
    /// The GPU vendor each node offers, a single vendor per node. A set is
    /// placed on a single vendor, because a collective spans only one.
    pub vendors: BTreeMap<NodeId, String>,
}

impl Topology {
    fn rtt(&self, a: &str, b: &str) -> Option<u64> {
        self.links
            .get(&(a.to_string(), b.to_string()))
            .or_else(|| self.links.get(&(b.to_string(), a.to_string())))
            .copied()
    }

    fn is_forwarded(&self, a: &str, b: &str) -> bool {
        self.forwarded.contains(&(a.to_string(), b.to_string()))
            || self.forwarded.contains(&(b.to_string(), a.to_string()))
    }

    /// The best link between two nodes meeting `link`, or None. With
    /// `cable`, a forwarded pair does not count.
    fn path(&self, a: &NodeId, b: &NodeId, link: Link, cable: bool) -> Option<Path> {
        self.paths(a, b, link, cable).into_iter().next()
    }

    /// Every link between two nodes meeting `link`, best first.
    ///
    /// Ranked by round trip, then by address, so a tie resolves the same way
    /// on every daemon.
    fn paths(&self, a: &NodeId, b: &NodeId, link: Link, cable: bool) -> Vec<Path> {
        let (Some(pa), Some(pb)) = (self.ports.get(a), self.ports.get(b)) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for x in pa {
            for y in pb {
                if link == Link::Rdma && !(x.rdma() && y.rdma()) {
                    continue;
                }
                let Some(rtt) = self.rtt(&x.addr, &y.addr) else {
                    continue;
                };
                if cable && self.is_forwarded(&x.addr, &y.addr) {
                    continue;
                }
                out.push(Path {
                    from: a.clone(),
                    to: b.clone(),
                    local: x.clone(),
                    remote: y.clone(),
                    rtt_ms: rtt,
                });
            }
        }
        out.sort_by(|p, q| {
            (p.rtt_ms, &p.local.addr, &p.remote.addr).cmp(&(
                q.rtt_ms,
                &q.local.addr,
                &q.remote.addr,
            ))
        });
        out
    }

    /// Each address to the lowest address of its node that reaches the same
    /// far addresses over cables. A ring treats two such addresses as one
    /// port. A DGX Spark shows each QSFP port as two interfaces on one wire.
    /// Each interface reaches both of the far box's, so the two share an
    /// entry. Two cables to one box reach different far addresses and keep
    /// two entries.
    fn port_groups(&self) -> BTreeMap<String, String> {
        let mut reach: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        for (a, b) in self.links.keys() {
            if !self.is_forwarded(a, b) {
                reach.entry(a).or_default().insert(b);
                reach.entry(b).or_default().insert(a);
            }
        }
        let mut out = BTreeMap::new();
        for ps in self.ports.values() {
            for p in ps {
                let mine = reach.get(p.addr.as_str());
                let first = ps
                    .iter()
                    .map(|q| &q.addr)
                    .filter(|q| reach.get(q.as_str()) == mine)
                    .min()
                    .unwrap_or(&p.addr);
                out.insert(p.addr.clone(), first.clone());
            }
        }
        out
    }
}

/// One usable link between two nodes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Path {
    pub from: NodeId,
    pub to: NodeId,
    pub local: Port,
    pub remote: Port,
    pub rtt_ms: u64,
}

impl Path {
    fn reversed(&self) -> Path {
        Path {
            from: self.to.clone(),
            to: self.from.clone(),
            local: self.remote.clone(),
            remote: self.local.clone(),
            rtt_ms: self.rtt_ms,
        }
    }

    /// Each end names its set, so a reader with three sets can tell which
    /// `between` showed which request without recomputing the solve.
    fn to_json(&self, hosts: &BTreeMap<NodeId, String>, sets: &BTreeMap<NodeId, String>) -> Value {
        let end = |n: &NodeId, p: &Port| {
            json!({
                "set": sets.get(n).cloned().unwrap_or_default(),
                "node": n,
                "host": hosts.get(n).cloned().unwrap_or_else(|| n.clone()),
                "addr": p.addr,
                "iface": p.iface,
            })
        };
        json!({
            "from": end(&self.from, &self.local),
            "to": end(&self.to, &self.remote),
            "rtt_ms": self.rtt_ms,
        })
    }
}

/// A member of a placed set, with the link its rank should bind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Member {
    pub node: NodeId,
    pub bind: Port,
    /// In a ring or a line, the links to the members before and after this
    /// one. A line's ends have one each.
    pub prev: Option<Path>,
    pub next: Option<Path>,
    /// In a star, a leaf's link to the hub.
    pub hub: Option<Path>,
    /// In a star, the hub's link to each leaf, in member order.
    pub leaves: Vec<Path>,
    /// The GPU vendor this member's bundles sit on. One vendor per set,
    /// since no collective spans vendors.
    pub vendor: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Solution {
    pub sets: BTreeMap<String, Vec<Member>>,
    pub between: Vec<Path>,
    /// For each ring, line, star or any set, the links and paths among
    /// its members.
    pub facts: BTreeMap<String, Facts>,
    /// What each `any` set was placed as.
    pub layouts: BTreeMap<String, Layout>,
}

/// What a caller needs to apply its own policy to a ring, line or star:
/// every link between two members, and the fewest-hop paths between
/// members with no link. Nothing here depends on the layout's order or on
/// `ports`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Facts {
    /// Every probed, non-forwarded link between two members that meets the
    /// set's `link`, once each, with `a` the lower node id.
    pub links: Vec<LinkFact>,
    /// For each ordered pair of members with no link, every fewest-hop path
    /// through the set's members, as node ids from the first to the last.
    pub paths: BTreeMap<NodeId, BTreeMap<NodeId, Vec<Vec<NodeId>>>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinkFact {
    pub a: NodeId,
    pub b: NodeId,
    pub a_port: Port,
    pub b_port: Port,
    /// Each end's port group, named by the lowest address in it.
    pub a_group: String,
    pub b_group: String,
    pub rtt_ms: u64,
}

/// A pair of members has at most this many paths listed. Dense sets have
/// few pairs without a link, but the count of tied paths grows with the
/// product of the degrees along them.
const MAX_PATHS: usize = 64;

/// The path walk for one pair stops after this many partial paths. A walk
/// that keeps to fewest-hop steps stays far below it. The head builds paths
/// holding its state lock, so this bounds the cost of a fault there.
const MAX_PATH_STEPS: usize = 100_000;

/// The links and paths among `nodes`.
fn facts(t: &Topology, set: &SetReq, groups: &BTreeMap<String, String>, nodes: &[NodeId]) -> Facts {
    let mut members: Vec<&NodeId> = nodes.iter().collect();
    members.sort();
    let group = |p: &Port| {
        groups
            .get(&p.addr)
            .cloned()
            .unwrap_or_else(|| p.addr.clone())
    };
    let mut links = Vec::new();
    let mut linked: BTreeMap<&NodeId, BTreeSet<&NodeId>> = BTreeMap::new();
    for (i, &a) in members.iter().enumerate() {
        for &b in &members[i + 1..] {
            let mut found = t.paths(a, b, set.link, true);
            found.sort_by(|p, q| {
                (&p.local.addr, &p.remote.addr).cmp(&(&q.local.addr, &q.remote.addr))
            });
            if !found.is_empty() {
                linked.entry(a).or_default().insert(b);
                linked.entry(b).or_default().insert(a);
            }
            links.extend(found.into_iter().map(|p| LinkFact {
                a: a.clone(),
                b: b.clone(),
                a_group: group(&p.local),
                b_group: group(&p.remote),
                a_port: p.local,
                b_port: p.remote,
                rtt_ms: p.rtt_ms,
            }));
        }
    }
    let mut paths: BTreeMap<NodeId, BTreeMap<NodeId, Vec<Vec<NodeId>>>> = BTreeMap::new();
    let empty = BTreeSet::new();
    let next = |n: &NodeId| linked.get(n).unwrap_or(&empty);
    for &from in &members {
        // Hops from `from` to each member it reaches.
        let mut dist: BTreeMap<&NodeId, usize> = BTreeMap::from([(from, 0)]);
        let mut frontier = vec![from];
        while !frontier.is_empty() {
            let mut more = Vec::new();
            for n in frontier {
                for m in next(n) {
                    if !dist.contains_key(m) {
                        dist.insert(m, dist[n] + 1);
                        more.push(*m);
                    }
                }
            }
            frontier = more;
        }
        for &to in &members {
            if to == from || next(from).contains(to) || !dist.contains_key(to) {
                continue;
            }
            // Walk back from `to` along members one hop nearer `from`, so
            // only fewest-hop paths are built, in node id order.
            let mut found: Vec<Vec<NodeId>> = Vec::new();
            let mut stack: Vec<Vec<&NodeId>> = vec![vec![to]];
            let mut steps = 0;
            while let Some(tail) = stack.pop() {
                steps += 1;
                if found.len() >= MAX_PATHS || steps > MAX_PATH_STEPS {
                    break;
                }
                let head = tail[tail.len() - 1];
                if head == from {
                    found.push(tail.iter().rev().map(|n| (*n).clone()).collect());
                    continue;
                }
                let step = dist[head] - 1;
                let back: Vec<&NodeId> = next(head)
                    .iter()
                    .copied()
                    .filter(|m| dist.get(m) == Some(&step))
                    .collect();
                for m in back.into_iter().rev() {
                    let mut longer = tail.clone();
                    longer.push(m);
                    stack.push(longer);
                }
            }
            found.sort();
            paths
                .entry(from.clone())
                .or_default()
                .insert(to.clone(), found);
        }
    }
    Facts { links, paths }
}

impl Solution {
    pub fn to_json(&self, t: &Topology) -> Value {
        let name = |n: &NodeId| t.hosts.get(n).cloned().unwrap_or_else(|| n.clone());
        let link = |p: &Path| {
            json!({
                "node": p.to,
                "host": name(&p.to),
                "addr": p.remote.addr,
                "bind": p.local.addr,
                "iface": p.local.iface,
            })
        };
        let sets: BTreeMap<String, Value> = self
            .sets
            .iter()
            .map(|(k, ms)| {
                let members: Vec<Value> = ms
                    .iter()
                    .map(|m| {
                        let mut row = json!({
                            "node": m.node,
                            "host": name(&m.node),
                            "vendor": m.vendor,
                            "bind": m.bind.addr,
                            "iface": m.bind.iface,
                            "tags": m.bind.tags,
                        });
                        for (key, p) in [("prev", &m.prev), ("next", &m.next), ("hub", &m.hub)] {
                            if let Some(p) = p {
                                row[key] = link(p);
                            }
                        }
                        if !m.leaves.is_empty() {
                            row["leaves"] = m.leaves.iter().map(link).collect();
                        }
                        row
                    })
                    .collect();
                (k.clone(), Value::Array(members))
            })
            .collect();
        // Which set each node landed in, so a `between` end can name it.
        let mut of_set: BTreeMap<NodeId, String> = BTreeMap::new();
        for (k, ms) in &self.sets {
            for m in ms {
                of_set.insert(m.node.clone(), k.clone());
            }
        }
        let end = |n: &NodeId, p: &Port, g: &String| json!({"node": n, "host": name(n), "addr": p.addr, "iface": p.iface, "port": g});
        let links: BTreeMap<&String, Vec<Value>> = self
            .facts
            .iter()
            .map(|(k, f)| {
                let rows = f
                    .links
                    .iter()
                    .map(|l| {
                        json!({
                            "a": end(&l.a, &l.a_port, &l.a_group),
                            "b": end(&l.b, &l.b_port, &l.b_group),
                            "rtt_ms": l.rtt_ms,
                        })
                    })
                    .collect();
                (k, rows)
            })
            .collect();
        let paths: BTreeMap<&String, Value> = self
            .facts
            .iter()
            .map(|(k, f)| (k, json!(f.paths)))
            .collect();
        json!({
            "sets": sets,
            "between": self
                .between
                .iter()
                .map(|p| p.to_json(&t.hosts, &of_set))
                .collect::<Vec<_>>(),
            "links": links,
            "paths": paths,
            "layouts": self
                .layouts
                .iter()
                .map(|(k, l)| (k.clone(), l.name()))
                .collect::<BTreeMap<_, _>>(),
        })
    }
}

/// One placement of a set: its nodes in order, and for a ring or a line
/// the link from each member to the next. A ring's last link closes it.
#[derive(Clone, Debug, PartialEq)]
struct Cand {
    nodes: Vec<NodeId>,
    links: Vec<Path>,
    /// What the set was placed as. An `any` set resolves to one of mesh,
    /// ring, line or hops.
    layout: Layout,
}

/// Candidate node groups for a set, best first.
///
/// An `rdma` set may only sit inside an island, since that is the derived
/// answer to "these nodes are cabled together and every pair was probed". An
/// `any` set uses nodes in id order, which is arbitrary but identical on
/// every daemon.
///
/// A ring or a line under `Ports::Auto` lists the answers that split every
/// member's two links across ports first. Four boxes on a switch and cabled
/// in pairs then ring over the cables and the switch in turn, with each box
/// using both ports.
/// With `first`, a ring or line starts only at that node.
fn candidates(
    t: &Topology,
    set: &SetReq,
    groups: &BTreeMap<String, String>,
    taken: &BTreeSet<NodeId>,
    driver: Option<&NodeId>,
    first: Option<&NodeId>,
) -> Vec<Cand> {
    if set.layout == Layout::Any {
        // Placement leaves a star pending with the driver on a leaf, and a
        // line with the driver inside it, so neither counts then.
        let places = |c: &Cand| {
            let Some(at) = driver.and_then(|d| c.nodes.iter().position(|n| n == d)) else {
                return true;
            };
            match c.layout {
                Layout::Star => at == 0,
                Layout::Line => at == 0 || at + 1 == c.nodes.len(),
                _ => true,
            }
        };
        // A line search stops at MAX_CHAINS answers. All of them can hold
        // the driver inside, so lines that start at the driver come first.
        let mut passes = vec![
            (Layout::Mesh, None),
            (Layout::Ring, None),
            (Layout::Star, None),
        ];
        if driver.is_some() {
            passes.push((Layout::Line, driver));
        }
        passes.push((Layout::Line, None));
        let mut out: Vec<Cand> = Vec::new();
        for (layout, first) in passes {
            let as_layout = SetReq {
                layout,
                ..set.clone()
            };
            for c in candidates(t, &as_layout, groups, taken, driver, first) {
                if places(&c) && !out.contains(&c) {
                    out.push(c);
                }
            }
        }
        out.extend(hops(t, set, taken));
        return out;
    }
    if set.layout.ordered() {
        let nodes: Vec<&NodeId> = t.free_gpus.keys().filter(|n| !taken.contains(*n)).collect();
        if set.layout == Layout::Star {
            let mut out = Vec::new();
            stars(t, set, &nodes, driver, &mut out);
            return out
                .into_iter()
                .map(|nodes| Cand {
                    nodes,
                    links: Vec::new(),
                    layout: Layout::Star,
                })
                .collect();
        }
        let rules: &[Rule] = match set.ports {
            Ports::Auto => &[Rule::Split, Rule::Any],
            Ports::Split => &[Rule::Split],
            Ports::Same => &[Rule::Same],
        };
        let mut out = Vec::new();
        for &rule in rules {
            let mut found = Vec::new();
            let w = Walk {
                t,
                set,
                groups,
                rule,
                first,
                steps: std::cell::Cell::new(0),
            };
            let nodes = if rule == Rule::Split && set.layout == Layout::Ring {
                match split_ring_nodes(t, set, groups, &nodes) {
                    Some(n) => n,
                    None => continue,
                }
            } else {
                nodes.clone()
            };
            w.walk(&nodes, &mut Vec::new(), Vec::new(), &mut found);
            for c in found {
                if !out.contains(&c) {
                    out.push(c);
                }
            }
        }
        return out;
    }
    let want = set.bundles.len();
    let fits = |n: &NodeId, i: usize| {
        !taken.contains(n) && t.free_gpus.get(n).copied().unwrap_or(0.0) >= set.bundles[i]
    };
    match set.link {
        Link::Rdma => {
            let mut out = Vec::new();
            for island in &t.islands {
                let avail: Vec<NodeId> = island
                    .nodes
                    .iter()
                    .filter(|n| !taken.contains(*n))
                    .cloned()
                    .collect();
                if avail.len() < want {
                    continue;
                }
                // The first `want` that fit, in island order. Island members
                // are interchangeable, so the first fit is as good as any. A
                // member whose GPUs are already reserved is skipped.
                let mut chosen: Vec<NodeId> = Vec::new();
                for n in &avail {
                    if chosen.len() == want {
                        break;
                    }
                    if fits(n, chosen.len()) {
                        chosen.push(n.clone());
                    }
                }
                if chosen.len() == want {
                    out.push(chosen);
                }
            }
            out.into_iter()
                .map(|nodes| Cand {
                    nodes,
                    links: Vec::new(),
                    layout: Layout::Mesh,
                })
                .collect()
        }
        Link::Any => {
            let mut chosen: Vec<NodeId> = Vec::new();
            for n in t.free_gpus.keys() {
                if chosen.len() == want {
                    break;
                }
                if fits(n, chosen.len()) {
                    chosen.push(n.clone());
                }
            }
            if chosen.len() == want {
                vec![Cand {
                    nodes: chosen,
                    links: Vec::new(),
                    layout: Layout::Mesh,
                }]
            } else {
                Vec::new()
            }
        }
    }
}

/// The node sets for an `any` set's fallback: `want` free nodes, each
/// linked over a cable to every other or reaching it through at most
/// `max_hops` other members. Sets come in node id order.
fn hops(t: &Topology, set: &SetReq, taken: &BTreeSet<NodeId>) -> Vec<Cand> {
    let want = set.bundles.len();
    let nodes: Vec<&NodeId> = t.free_gpus.keys().filter(|n| !taken.contains(*n)).collect();
    let mut out = Vec::new();
    if want == 0 || want > nodes.len() {
        return out;
    }
    let linked = |a: &NodeId, b: &NodeId| t.path(a, b, set.link, true).is_some();
    // Whether every member reaches every other within max_hops + 1 links,
    // stepping through members only.
    let close = |members: &[&NodeId]| {
        members.iter().all(|&from| {
            let mut seen: BTreeSet<&NodeId> = BTreeSet::from([from]);
            let mut frontier = vec![from];
            for _ in 0..=set.max_hops {
                let mut more = Vec::new();
                for n in frontier {
                    for &m in members {
                        if !seen.contains(m) && linked(n, m) {
                            seen.insert(m);
                            more.push(m);
                        }
                    }
                }
                frontier = more;
            }
            seen.len() == members.len()
        })
    };
    let mut steps = 0;
    let mut pick: Vec<usize> = (0..want).collect();
    loop {
        steps += 1;
        if steps > MAX_STEPS || out.len() >= MAX_CHAINS {
            break;
        }
        let members: Vec<&NodeId> = pick.iter().map(|&i| nodes[i]).collect();
        let fits = members
            .iter()
            .zip(&set.bundles)
            .all(|(n, b)| t.free_gpus.get(*n).copied().unwrap_or(0.0) >= *b);
        if fits && close(&members) {
            out.push(Cand {
                nodes: members.into_iter().cloned().collect(),
                links: Vec::new(),
                layout: Layout::Hops,
            });
        }
        // The next combination in order.
        let Some(i) = (0..want).rev().find(|&i| pick[i] < nodes.len() - want + i) else {
            break;
        };
        pick[i] += 1;
        for j in i + 1..want {
            pick[j] = pick[j - 1] + 1;
        }
    }
    out
}

/// A ring, line or star search stops after this many answers. Each is a
/// candidate the solver may reject for a later set, and a larger cluster
/// has many more.
const MAX_CHAINS: usize = 32;

/// A partial ring or line. `leaves` is the port group at the start of its
/// first link, and `arrives` the port group at the end of its last link.
struct Trail {
    leaves: String,
    arrives: String,
    links: Vec<Path>,
}

/// The nodes that can hold a split ring, or None when no split ring exists.
///
/// A split ring member reaches its two neighbours by two port groups, so a
/// node with fewer than two groups that link to other usable nodes is left
/// out, and leaving it out can strand another. A node with exactly two
/// groups uses both, so a group that reaches only one node names a
/// neighbour the ring must hold. Where those forced neighbours are mutual,
/// they join the nodes into parts, and a ring holds each part whole or not
/// at all. No split ring exists unless some parts add up to its size. Boxes
/// on a switch and cabled in pairs are parts of two, so this refuses an odd
/// split ring there before any search.
fn split_ring_nodes<'a>(
    t: &Topology,
    set: &SetReq,
    groups: &BTreeMap<String, String>,
    nodes: &[&'a NodeId],
) -> Option<Vec<&'a NodeId>> {
    let want = set.bundles.len();
    if want < 2 {
        return Some(nodes.to_vec());
    }
    let least = set.bundles.iter().copied().fold(f64::INFINITY, f64::min);
    let mut keep: Vec<&NodeId> = nodes
        .iter()
        .copied()
        .filter(|n| t.free_gpus.get(*n).copied().unwrap_or(0.0) >= least)
        .collect();
    // Each kept node's port groups, and the kept nodes each one reaches.
    let reach = |keep: &[&NodeId]| -> BTreeMap<NodeId, BTreeMap<String, BTreeSet<NodeId>>> {
        let mut out: BTreeMap<NodeId, BTreeMap<String, BTreeSet<NodeId>>> = BTreeMap::new();
        for &a in keep {
            for &b in keep {
                if a == b {
                    continue;
                }
                for p in t.paths(a, b, set.link, true) {
                    let g = groups
                        .get(&p.local.addr)
                        .cloned()
                        .unwrap_or_else(|| p.local.addr.clone());
                    out.entry(a.clone())
                        .or_default()
                        .entry(g)
                        .or_default()
                        .insert(b.clone());
                }
            }
        }
        out
    };
    let mut by_group = reach(&keep);
    loop {
        let before = keep.len();
        keep.retain(|n| by_group.get(*n).is_some_and(|g| g.len() >= 2));
        if keep.len() == before {
            break;
        }
        by_group = reach(&keep);
    }
    if keep.len() < want {
        return None;
    }
    let mut forced: BTreeMap<&NodeId, BTreeSet<&NodeId>> = BTreeMap::new();
    for &n in &keep {
        let gs = &by_group[n];
        if gs.len() != 2 {
            continue;
        }
        for to in gs.values().filter(|to| to.len() == 1) {
            let x = keep.iter().copied().find(|k| to.contains(*k)).unwrap();
            forced.entry(n).or_default().insert(x);
        }
    }
    let mutual = forced.iter().all(|(n, xs)| {
        xs.iter()
            .all(|x| forced.get(x).is_some_and(|b| b.contains(n)))
    });
    if !mutual {
        return Some(keep);
    }
    // The sizes of the parts forced neighbours join, then whether some of
    // them add up to `want`.
    let mut part: BTreeMap<&NodeId, usize> = BTreeMap::new();
    let mut sizes = Vec::new();
    for &n in &keep {
        if part.contains_key(n) {
            continue;
        }
        let mut todo = vec![n];
        let mut size = 0;
        part.insert(n, sizes.len());
        while let Some(m) = todo.pop() {
            size += 1;
            for &x in forced.get(m).into_iter().flatten() {
                if part.insert(x, sizes.len()).is_none() {
                    todo.push(x);
                }
            }
        }
        sizes.push(size);
    }
    let mut sums = vec![false; want + 1];
    sums[0] = true;
    for size in sizes {
        for total in (size..=want).rev() {
            sums[total] |= sums[total - size];
        }
    }
    sums[want].then_some(keep)
}

/// What a ring or line search asks of each member's two links.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rule {
    /// Different port groups.
    Split,
    /// One port group.
    Same,
    /// The best link for each step, whatever its port.
    Any,
}

/// A ring or line search gives up after visiting this many partial
/// chains. Some requests have no answer and a search space that grows with
/// the factorial of the cluster: a split ring of odd size on boxes cabled in
/// pairs visits every order. The head solves a claim holding its state lock.
const MAX_STEPS: usize = 20_000;

/// A ring or line search under one `Rule`.
struct Walk<'t> {
    t: &'t Topology,
    set: &'t SetReq,
    groups: &'t BTreeMap<String, String>,
    rule: Rule,
    /// The only node a chain may start at, when given.
    first: Option<&'t NodeId>,
    steps: std::cell::Cell<usize>,
}

impl Walk<'_> {
    /// Whether a member may reach its neighbours by these two port groups.
    fn allows(&self, a: &str, b: &str) -> bool {
        match self.rule {
            Rule::Split => a != b,
            Rule::Same => a == b,
            Rule::Any => true,
        }
    }

    fn group(&self, p: &Port) -> String {
        self.groups
            .get(&p.addr)
            .cloned()
            .unwrap_or_else(|| p.addr.clone())
    }

    /// The links worth trying from `a` to `b`: under `Rule::Any` the best
    /// one, else the best per pair of port groups.
    fn options(&self, a: &NodeId, b: &NodeId) -> Vec<Path> {
        let all = self.t.paths(a, b, self.set.link, true);
        if self.rule == Rule::Any {
            return all.into_iter().take(1).collect();
        }
        let mut seen = BTreeSet::new();
        all.into_iter()
            .filter(|p| seen.insert((self.group(&p.local), self.group(&p.remote))))
            .collect()
    }

    /// Each trail extended by a link from `last` to `n`, one per distinct
    /// pair of end groups. An empty `trails` means `last` is the first
    /// member.
    fn extend(&self, last: &NodeId, n: &NodeId, trails: &[Trail]) -> Vec<Trail> {
        let options = self.options(last, n);
        let mut out: Vec<Trail> = Vec::new();
        let mut add = |leaves: String, p: &Path, before: &[Path]| {
            let arrives = self.group(&p.remote);
            if out
                .iter()
                .any(|o| o.leaves == leaves && o.arrives == arrives)
            {
                return;
            }
            let mut links = before.to_vec();
            links.push(p.clone());
            out.push(Trail {
                leaves,
                arrives,
                links,
            });
        };
        if trails.is_empty() {
            for p in &options {
                add(self.group(&p.local), p, &[]);
            }
        }
        for tr in trails {
            for p in &options {
                if !self.allows(&tr.arrives, &self.group(&p.local)) {
                    continue;
                }
                add(tr.leaves.clone(), p, &tr.links);
            }
        }
        out
    }

    /// The links of a full chain: a line's as they are, a ring's with a
    /// link from the last member back to the first.
    fn finish(&self, chain: &[&NodeId], trails: &[Trail]) -> Option<Vec<Path>> {
        let k = chain.len();
        if k == 1 {
            return Some(Vec::new());
        }
        if self.set.layout != Layout::Ring {
            return trails.first().map(|tr| tr.links.clone());
        }
        // Two members on one link are a ring.
        if k == 2 && self.rule == Rule::Any {
            let tr = trails.first()?;
            return Some(vec![tr.links[0].clone(), tr.links[0].reversed()]);
        }
        let options = self.options(chain[k - 1], chain[0]);
        for tr in trails {
            for p in &options {
                if !self.allows(&tr.arrives, &self.group(&p.local))
                    || !self.allows(&self.group(&p.remote), &tr.leaves)
                {
                    continue;
                }
                let mut links = tr.links.clone();
                links.push(p.clone());
                return Some(links);
            }
        }
        None
    }

    /// Collect into `out` the rings or lines that extend `chain`: each
    /// member links to the next over a cable, and a ring's last member links
    /// to its first.
    ///
    /// A ring starts at its lowest node id, so each ring is found once per
    /// direction. Placement rotates it to the driver's node.
    fn walk<'a>(
        &self,
        nodes: &[&'a NodeId],
        chain: &mut Vec<&'a NodeId>,
        trails: Vec<Trail>,
        out: &mut Vec<Cand>,
    ) {
        let set = self.set;
        let want = set.bundles.len();
        if out.len() >= MAX_CHAINS || self.steps.get() >= MAX_STEPS {
            return;
        }
        self.steps.set(self.steps.get() + 1);
        if chain.len() == want {
            if let Some(links) = self.finish(chain, &trails) {
                out.push(Cand {
                    nodes: chain.iter().map(|n| (*n).clone()).collect(),
                    links,
                    layout: self.set.layout,
                });
            }
            return;
        }
        let i = chain.len();
        for &n in nodes {
            if chain.contains(&n)
                || self.t.free_gpus.get(n).copied().unwrap_or(0.0) < set.bundles[i]
            {
                continue;
            }
            if chain.is_empty() && self.first.is_some_and(|f| f != n) {
                continue;
            }
            if set.layout == Layout::Ring && chain.first().is_some_and(|first| n < *first) {
                continue;
            }
            let next = match chain.last() {
                Some(last) => {
                    let next = self.extend(last, n, &trails);
                    if next.is_empty() {
                        continue;
                    }
                    next
                }
                None => Vec::new(),
            };
            chain.push(n);
            self.walk(nodes, chain, next, out);
            chain.pop();
        }
    }
}

/// Collect into `out` the node orders for a star set: a hub, then members
/// each cabled to the hub. Hubs come in id order, with `first` ahead of the
/// rest.
fn stars(
    t: &Topology,
    set: &SetReq,
    nodes: &[&NodeId],
    first: Option<&NodeId>,
    out: &mut Vec<Vec<NodeId>>,
) {
    let mut hubs = nodes.to_vec();
    hubs.sort_by_key(|n| Some(*n) != first);
    for hub in hubs {
        if t.free_gpus.get(hub).copied().unwrap_or(0.0) < set.bundles[0] {
            continue;
        }
        let leaves: Vec<&NodeId> = nodes
            .iter()
            .copied()
            .filter(|n| *n != hub && t.path(hub, n, set.link, true).is_some())
            .collect();
        spokes(t, set, &leaves, &mut vec![hub], out);
    }
}

/// Extend `chain`, a hub and some of its leaves, to a full star.
fn spokes<'a>(
    t: &Topology,
    set: &SetReq,
    leaves: &[&'a NodeId],
    chain: &mut Vec<&'a NodeId>,
    out: &mut Vec<Vec<NodeId>>,
) {
    if out.len() >= MAX_CHAINS {
        return;
    }
    let i = chain.len();
    if i == set.bundles.len() {
        out.push(chain.iter().map(|n| (*n).clone()).collect());
        return;
    }
    for &n in leaves {
        if chain.contains(&n) || t.free_gpus.get(n).copied().unwrap_or(0.0) < set.bundles[i] {
            continue;
        }
        // Leaves asking for the same GPUs are interchangeable, so they come
        // in id order and each choice is found once.
        if (1..i).any(|j| set.bundles[j] == set.bundles[i] && n < chain[j]) {
            continue;
        }
        chain.push(n);
        spokes(t, set, leaves, chain, out);
        chain.pop();
    }
}

/// In a star, a leaf's link to the hub, or the hub's link to each leaf.
/// `layout` is what the set was placed as, which for an `any` set differs
/// from its request.
fn star_links(
    t: &Topology,
    set: &SetReq,
    layout: Layout,
    nodes: &[NodeId],
    i: usize,
) -> Result<(Option<Path>, Vec<Path>), String> {
    if layout != Layout::Star {
        return Ok((None, Vec::new()));
    }
    let link = |j: usize| {
        t.path(&nodes[i], &nodes[j], set.link, true).ok_or_else(|| {
            format!(
                "{} has no cable to {} in set {:?}",
                nodes[i], nodes[j], set.name
            )
        })
    };
    if i == 0 {
        Ok((None, (1..nodes.len()).map(link).collect::<Result<_, _>>()?))
    } else {
        Ok((Some(link(0)?), Vec::new()))
    }
}

/// The address a member binds for its own set's traffic.
fn bind_for(t: &Topology, set: &SetReq, node: &NodeId, peers: &[NodeId]) -> Option<Port> {
    let ports = t.ports.get(node)?;
    match set.link {
        // The fabric address that reaches the rest of the set.
        Link::Rdma => ports
            .iter()
            .find(|p| {
                p.rdma()
                    && peers.iter().filter(|q| *q != node).all(|q| {
                        t.ports
                            .get(q)
                            .map(|qp| {
                                qp.iter()
                                    .any(|y| y.rdma() && t.rtt(&p.addr, &y.addr).is_some())
                            })
                            .unwrap_or(false)
                    })
            })
            .cloned(),
        // The node's own first choice.
        Link::Any => ports.first().cloned(),
    }
}

/// The port of `node` with a cable to the most other members, the first
/// such port on a tie.
fn bind_hops(t: &Topology, set: &SetReq, node: &NodeId, members: &[NodeId]) -> Option<Port> {
    let ports = t.ports.get(node)?;
    let reach = |p: &Port| {
        members
            .iter()
            .filter(|m| *m != node)
            .filter(|m| {
                t.paths(node, m, set.link, true)
                    .iter()
                    .any(|q| q.local.addr == p.addr)
            })
            .count()
    };
    let mut best: Option<(&Port, usize)> = None;
    for p in ports {
        if set.link == Link::Rdma && !p.rdma() {
            continue;
        }
        let n = reach(p);
        if best.is_none_or(|(_, b)| n > b) {
            best = Some((p, n));
        }
    }
    best.map(|(p, _)| p.clone())
}

/// Match a request against the topology.
///
/// Sets are placed in the order given, with `rdma` sets first: they have the
/// fewest places to go, and placing a loose set first can claim a node the
/// constrained one needed. Each set's candidates are tried in turn and the
/// choice is undone if a later set or a `between` requirement fails.
pub fn solve(t: &Topology, req: &Request) -> Result<Solution, String> {
    for s in &req.sets {
        if s.bundles.is_empty() {
            return Err(format!("set {:?} asks for no nodes", s.name));
        }
    }
    let names: BTreeSet<&str> = req.sets.iter().map(|s| s.name.as_str()).collect();
    if names.len() != req.sets.len() {
        return Err("two sets share a name".into());
    }
    for b in &req.between {
        for n in [&b.from, &b.to] {
            if !names.contains(n.as_str()) {
                return Err(format!("between names {n:?}, which is not a set"));
            }
        }
    }

    let mut order: Vec<&SetReq> = req.sets.iter().collect();
    order.sort_by_key(|s| (s.link != Link::Rdma, s.name.clone()));

    let groups = t.port_groups();
    let mut chosen: BTreeMap<String, Cand> = BTreeMap::new();
    let mut taken: BTreeSet<NodeId> = BTreeSet::new();
    if !search(t, req, &groups, &order, 0, &mut chosen, &mut taken) {
        return Err(why_not(t, req));
    }

    let mut sets: BTreeMap<String, Vec<Member>> = BTreeMap::new();
    for s in &req.sets {
        let Cand {
            nodes,
            links,
            layout,
        } = &chosen[&s.name];
        let k = nodes.len();
        let mut members = Vec::new();
        for (i, n) in nodes.iter().enumerate() {
            let (prev, next) = match layout {
                Layout::Ring if k > 1 => (
                    Some(links[(i + k - 1) % k].reversed()),
                    Some(links[i].clone()),
                ),
                Layout::Line => (
                    (i > 0).then(|| links[i - 1].reversed()),
                    links.get(i).cloned(),
                ),
                _ => (None, None),
            };
            let (hub, leaves) = star_links(t, s, *layout, nodes, i)?;
            // A ring or line member binds the port toward its next member,
            // and a line's last member the port toward its previous one. A
            // star's leaf binds the port toward the hub, and the hub the port
            // toward its first leaf.
            let toward = next.as_ref().or(prev.as_ref()).or(hub.as_ref());
            // A hops member reaches some members only through others, so it
            // binds the port linking it to the most of them.
            let bind = match (toward.or(leaves.first()), layout) {
                (Some(p), _) => p.local.clone(),
                (None, Layout::Hops) => bind_hops(t, s, n, nodes)
                    .ok_or_else(|| format!("{n} has no address for set {:?}", s.name))?,
                (None, _) => bind_for(t, s, n, nodes)
                    .ok_or_else(|| format!("{n} has no address for set {:?}", s.name))?,
            };
            members.push(Member {
                node: n.clone(),
                bind,
                prev,
                next,
                hub,
                leaves,
                vendor: if s.vendor.is_empty() {
                    t.vendors.get(n).cloned().unwrap_or_default()
                } else {
                    s.vendor.clone()
                },
            });
        }
        sets.insert(s.name.clone(), members);
    }
    let between =
        paths_between(t, req, &chosen).ok_or("no probed link satisfies a between entry")?;
    let facts = req
        .sets
        .iter()
        .filter(|s| s.layout.ordered() || s.layout == Layout::Any)
        .map(|s| (s.name.clone(), facts(t, s, &groups, &chosen[&s.name].nodes)))
        .collect();
    let layouts = req
        .sets
        .iter()
        .filter(|s| s.layout == Layout::Any)
        .map(|s| (s.name.clone(), chosen[&s.name].layout))
        .collect();
    Ok(Solution {
        sets,
        between,
        facts,
        layouts,
    })
}

fn search(
    t: &Topology,
    req: &Request,
    groups: &BTreeMap<String, String>,
    order: &[&SetReq],
    i: usize,
    chosen: &mut BTreeMap<String, Cand>,
    taken: &mut BTreeSet<NodeId>,
) -> bool {
    if i == order.len() {
        return paths_between(t, req, chosen).is_some();
    }
    let set = order[i];
    let driver = req
        .driver
        .as_ref()
        .filter(|_| req.driver_rule == Driver::First);
    for cand in candidates(t, set, groups, taken, driver, None) {
        for n in &cand.nodes {
            taken.insert(n.clone());
        }
        let nodes = cand.nodes.clone();
        chosen.insert(set.name.clone(), cand);
        if search(t, req, groups, order, i + 1, chosen, taken) {
            return true;
        }
        chosen.remove(&set.name);
        for n in &nodes {
            taken.remove(n);
        }
    }
    false
}

/// One path per `between` requirement, or None if any cannot be met.
///
/// A requirement holds when some member of each set can reach some member of
/// the other. The reported path is the one a caller would use.
fn paths_between(
    t: &Topology,
    req: &Request,
    chosen: &BTreeMap<String, Cand>,
) -> Option<Vec<Path>> {
    let mut out = Vec::new();
    for b in &req.between {
        let (from, to) = (&chosen.get(&b.from)?.nodes, &chosen.get(&b.to)?.nodes);
        let mut best: Option<Path> = None;
        for a in from {
            for c in to {
                if let Some(p) = t.path(a, c, b.link, false) {
                    let better = match &best {
                        None => true,
                        Some(b0) => (p.rtt_ms, &p.from, &p.to) < (b0.rtt_ms, &b0.from, &b0.to),
                    };
                    if better {
                        best = Some(p);
                    }
                }
            }
        }
        out.push(best?);
    }
    Some(out)
}

/// Why nothing fit, in the terms the request was written in.
fn why_not(t: &Topology, req: &Request) -> String {
    let mut parts = Vec::new();
    for s in &req.sets {
        let want = s.bundles.len();
        if s.layout == Layout::Any {
            parts.push(format!(
                "set {:?} wants any layout of {want} nodes over {} links: no island \
                 holds them, no ring, star or line of cables joins them, and no {want} \
                 free nodes each reach every other through at most {} others",
                s.name,
                if s.link == Link::Rdma {
                    "rdma"
                } else {
                    "probed"
                },
                s.max_hops,
            ));
            continue;
        }
        if s.layout.ordered() {
            let how = match (s.layout, s.ports) {
                (Layout::Star, _) => "a hub cabled to each of the others",
                (_, Ports::Split) => {
                    "each cabled to the next and reaching its two neighbours by different ports"
                }
                (_, Ports::Same) => {
                    "each cabled to the next and reaching its two neighbours by one port"
                }
                _ => "each cabled to the next",
            };
            // The search stops at MAX_STEPS. A ring or line it did not
            // reach may still exist.
            let order = if s.layout == Layout::Star {
                "no free node has enough free neighbours"
            } else {
                "no such order of free nodes was found"
            };
            parts.push(format!(
                "set {:?} wants a {} of {want} nodes, {how} over {} links, and {order}",
                s.name,
                s.layout.name(),
                if s.link == Link::Rdma {
                    "rdma"
                } else {
                    "probed"
                },
            ));
            continue;
        }
        match s.link {
            Link::Rdma => {
                let biggest = t.islands.iter().map(|i| i.nodes.len()).max().unwrap_or(0);
                if biggest < want {
                    parts.push(format!(
                        "set {:?} wants {want} nodes sharing a fabric, largest island has {biggest}",
                        s.name
                    ));
                }
            }
            Link::Any => {
                let have = t.free_gpus.len();
                if have < want {
                    parts.push(format!(
                        "set {:?} wants {want} nodes, cluster has {have}",
                        s.name
                    ));
                }
            }
        }
    }
    if parts.is_empty() {
        parts.push("no assignment satisfies every set and link together".into());
    }
    parts.join(". ")
}

// ---------------------------------------------------------------------------
// The claim table
// ---------------------------------------------------------------------------

/// Read a request out of the JSON a client sent.
pub fn parse(shape: &Value) -> Result<Request, String> {
    let sets = shape["sets"]
        .as_array()
        .ok_or("shape needs a sets array")?
        .iter()
        .map(|s| {
            let name = s["name"].as_str().ok_or("a set needs a name")?.to_string();
            let bundles: Vec<f64> = match &s["bundles"] {
                // Whole GPUs. `canonical` spells 1 and 1.0 alike. A fraction
                // is refused.
                Value::Array(a) => a
                    .iter()
                    .map(|b| {
                        b.as_f64()
                            .filter(|f| f.is_finite() && *f >= 0.0 && f.fract() == 0.0)
                            .ok_or_else(|| {
                                format!("set {name:?}: every bundle is a whole number of GPUs")
                            })
                    })
                    .collect::<Result<Vec<f64>, String>>()?,
                // A count with no per-node figure means one GPU each, which
                // is what a rank usually wants.
                Value::Number(n) => vec![1.0; n.as_u64().unwrap_or(0) as usize],
                _ => return Err(format!("set {name:?} needs bundles")),
            };
            let layout = Layout::parse(s["layout"].as_str().unwrap_or("mesh"))?;
            let ports = Ports::parse(s["ports"].as_str().unwrap_or("auto"))?;
            if ports != Ports::Auto && !matches!(layout, Layout::Ring | Layout::Line | Layout::Any)
            {
                return Err(format!(
                    "set {name:?}: ports applies to a ring, a line or any"
                ));
            }
            // Only one hop for now. A longer path relays through more boxes
            // whose ports carry their own ranks' traffic.
            let max_hops = match &s["max_hops"] {
                Value::Null => 1,
                v if layout != Layout::Any => {
                    return Err(format!(
                        "set {name:?}: max_hops applies to an any layout, got {v}"
                    ))
                }
                v if v.as_u64() == Some(1) => 1,
                v => return Err(format!("set {name:?}: max_hops must be 1 for now, got {v}")),
            };
            Ok(SetReq {
                name,
                bundles,
                link: Link::parse(s["link"].as_str().unwrap_or("ip"))?,
                layout,
                ports,
                max_hops,
                vendor: s["vendor"].as_str().unwrap_or_default().to_string(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let between = shape["between"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|b| {
                    Ok(BetweenReq {
                        from: b["from"].as_str().ok_or("between needs from")?.to_string(),
                        to: b["to"].as_str().ok_or("between needs to")?.to_string(),
                        link: Link::parse(b["link"].as_str().unwrap_or("ip"))?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()
        })
        .transpose()?
        .unwrap_or_default();
    Ok(Request {
        sets,
        between,
        driver: None,
        driver_rule: match &shape["driver"] {
            Value::Null => Driver::First,
            Value::String(s) => Driver::parse(s)?,
            v => return Err(format!("driver must be first or ignore, got {v}")),
        },
    })
}

/// The topology as this daemon currently sees it.
///
/// Ports come from the peer table, which holds each node's addresses, the
/// operator's tags and the interface each sits on. Links come from the
/// probes, self to peer and peer to peer alike, so a fabric this daemon is
/// not on still counts.
pub fn topology(st: &crate::state::State) -> Topology {
    let mut t = Topology {
        islands: st.fabrics.islands.clone(),
        ..Default::default()
    };
    let mut record = |node: &NodeId,
                      host: &str,
                      addrs: &[String],
                      tags: &BTreeMap<String, Vec<String>>,
                      ifaces: &BTreeMap<String, String>| {
        let ports: Vec<Port> = addrs
            .iter()
            .map(|a| Port {
                addr: a.clone(),
                iface: ifaces.get(a).cloned(),
                tags: tags.get(a).cloned().unwrap_or_default(),
            })
            .collect();
        if !ports.is_empty() {
            t.ports.insert(node.clone(), ports);
        }
        t.hosts.insert(node.clone(), host.to_string());
    };
    record(
        &st.node_id,
        &st.hostname,
        &crate::announce::local_addrs(),
        &crate::announce::local_addr_tags(),
        &crate::announce::local_addr_ifaces(),
    );
    for p in st.peers.values().filter(|p| p.alive) {
        record(
            &p.node_id,
            &p.node_ip,
            &p.addrs,
            &p.addr_tags,
            &p.addr_ifaces,
        );
        for (local, remotes) in &p.probe_pairs {
            for (remote, r) in remotes {
                if r.ok {
                    t.links.insert((local.clone(), remote.clone()), r.rtt_ms);
                }
                if r.direct == Some(false) {
                    t.forwarded.insert((local.clone(), remote.clone()));
                }
            }
        }
        for (_, q) in p.last_status["peers"].as_object().into_iter().flatten() {
            // A dead peer's last probes describe a box that has since gone.
            if !q["alive"].as_bool().unwrap_or(false) {
                continue;
            }
            for (local, remotes) in q["probes"].as_object().into_iter().flatten() {
                for (remote, r) in remotes.as_object().into_iter().flatten() {
                    if r["ok"].as_bool().unwrap_or(false) {
                        t.links.insert(
                            (local.clone(), remote.clone()),
                            r["rtt_ms"].as_u64().unwrap_or(0),
                        );
                    }
                    if r["direct"] == false {
                        t.forwarded.insert((local.clone(), remote.clone()));
                    }
                }
            }
        }
    }
    // Only a node with an agent has GPUs to give.
    for a in st.agents.values().filter(|a| a.alive) {
        *t.free_gpus.entry(a.node_id.clone()).or_insert(0.0) += st.free_gpus_of(&a.id).len() as f64;
        if let Some(g) = a.machine.gpus.first() {
            t.vendors
                .entry(a.node_id.clone())
                .or_insert_with(|| g.vendor.clone());
        }
        // An agent can register a node no daemon peers for, which leaves it
        // holding GPUs with no address to bind. What it told us on register
        // is an address, so it stands in. It is untagged, which limits
        // that node to plain links.
        t.ports.entry(a.node_id.clone()).or_insert_with(|| {
            vec![Port {
                addr: a.node_ip.clone(),
                iface: None,
                tags: Vec::new(),
            }]
        });
        t.hosts
            .entry(a.node_id.clone())
            .or_insert_with(|| a.node_ip.clone());
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `[1]` and `[1.0]` were two shapes to the holder check, which refused
    /// the second holder of a claim. A fractional bundle is refused.
    #[test]
    fn a_shape_reads_the_same_spelled_either_way() {
        let ints = serde_json::json!({"sets": [{"name": "s", "bundles": [1, 2]}]});
        let floats = serde_json::json!({"sets": [{"name": "s", "bundles": [1.0, 2.0]}]});
        assert_eq!(canonical(&ints), canonical(&floats));
        assert_eq!(canonical(&floats), ints);

        // A count spells the same request `parse` expands it into.
        let count = serde_json::json!({"sets": [{"name": "s", "bundles": 2}]});
        let pair = serde_json::json!({"sets": [{"name": "s", "bundles": [1, 1]}]});
        assert_eq!(canonical(&count), canonical(&pair));

        let half = serde_json::json!({"sets": [{"name": "s", "bundles": [0.5]}]});
        let e = parse(&half).unwrap_err();
        assert!(e.contains("whole number"), "{e}");
    }

    fn port(addr: &str, iface: &str, tags: &[&str]) -> Port {
        Port {
            addr: addr.into(),
            iface: Some(iface.into()),
            tags: tags.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Two cabled pairs on separate fabrics, reaching each other only over
    /// the LAN. The topology this was written for.
    fn two_pairs() -> Topology {
        let mut t = Topology::default();
        let spec = [
            ("nA", "10.100.0.1", "192.168.1.70", "gx10-2353"),
            ("nB", "10.100.0.2", "192.168.1.77", "gx10-5818"),
            ("nC", "10.103.0.36", "192.168.1.36", "gx10-9722"),
            ("nD", "10.103.0.93", "192.168.1.93", "gx10-a7c3"),
        ];
        for (n, fab, lan, host) in spec {
            t.ports.insert(
                n.into(),
                vec![
                    port(fab, "enp1s0f0np0", &["connectx", "rdma"]),
                    port(lan, "eno1", &["lan"]),
                ],
            );
            t.free_gpus.insert(n.into(), 1.0);
            t.hosts.insert(n.into(), host.into());
        }
        // Fabric links inside each pair only.
        t.links
            .insert(("10.100.0.1".into(), "10.100.0.2".into()), 0);
        t.links
            .insert(("10.103.0.36".into(), "10.103.0.93".into()), 0);
        // The LAN reaches everything.
        for a in [
            "192.168.1.70",
            "192.168.1.77",
            "192.168.1.36",
            "192.168.1.93",
        ] {
            for b in [
                "192.168.1.70",
                "192.168.1.77",
                "192.168.1.36",
                "192.168.1.93",
            ] {
                if a != b {
                    t.links.insert((a.into(), b.into()), 1);
                }
            }
        }
        t.islands = vec![
            Island {
                nodes: vec!["nA".into(), "nB".into()],
                addr: [
                    ("nA".into(), "10.100.0.1".into()),
                    ("nB".into(), "10.100.0.2".into()),
                ]
                .into_iter()
                .collect(),
            },
            Island {
                nodes: vec!["nC".into(), "nD".into()],
                addr: [
                    ("nC".into(), "10.103.0.36".into()),
                    ("nD".into(), "10.103.0.93".into()),
                ]
                .into_iter()
                .collect(),
            },
        ];
        t
    }

    fn req(sets: &[(&str, usize, Link)], between: &[(&str, &str, Link)]) -> Request {
        Request {
            sets: sets
                .iter()
                .map(|(n, k, l)| SetReq {
                    name: n.to_string(),
                    bundles: vec![1.0; *k],
                    link: *l,
                    layout: Layout::Mesh,
                    ports: Ports::Auto,
                    max_hops: 1,
                    vendor: String::new(),
                })
                .collect(),
            between: between
                .iter()
                .map(|(f, t2, l)| BetweenReq {
                    from: f.to_string(),
                    to: t2.to_string(),
                    link: *l,
                })
                .collect(),
            driver: None,
            driver_rule: Driver::First,
        }
    }

    /// The request this exists for: two cabled pairs, IP between them.
    #[test]
    fn two_fabric_pairs_with_ip_between_them() {
        let t = two_pairs();
        let r = req(
            &[("tp0", 2, Link::Rdma), ("tp1", 2, Link::Rdma)],
            &[("tp0", "tp1", Link::Any)],
        );
        let s = solve(&t, &r).expect("should place");

        let nodes = |k: &str| {
            s.sets[k]
                .iter()
                .map(|m| m.node.clone())
                .collect::<BTreeSet<_>>()
        };
        assert_eq!(nodes("tp0"), ["nA", "nB"].map(String::from).into());
        assert_eq!(nodes("tp1"), ["nC", "nD"].map(String::from).into());

        // Each set binds its fabric, with the interface to bind it on.
        for m in &s.sets["tp0"] {
            assert!(m.bind.tags.contains(&"rdma".to_string()), "{m:?}");
            assert_eq!(m.bind.iface.as_deref(), Some("enp1s0f0np0"));
        }
        // Between them, the LAN.
        assert_eq!(s.between.len(), 1);
        assert!(s.between[0].local.tags.contains(&"lan".to_string()));
        assert_eq!(s.between[0].local.iface.as_deref(), Some("eno1"));
    }

    /// Sets never share a node, or two ranks would collide on one box.
    #[test]
    fn sets_are_disjoint() {
        let t = two_pairs();
        let s = solve(&t, &req(&[("a", 2, Link::Rdma), ("b", 2, Link::Rdma)], &[])).unwrap();
        let a: BTreeSet<_> = s.sets["a"].iter().map(|m| &m.node).collect();
        let b: BTreeSet<_> = s.sets["b"].iter().map(|m| &m.node).collect();
        assert!(a.is_disjoint(&b));
    }

    /// A fabric requirement between two sets on separate fabrics cannot be
    /// met, and reporting it beats placing them and hanging in NCCL.
    #[test]
    fn a_fabric_between_separate_fabrics_is_refused() {
        let t = two_pairs();
        let r = req(
            &[("tp0", 2, Link::Rdma), ("tp1", 2, Link::Rdma)],
            &[("tp0", "tp1", Link::Rdma)],
        );
        assert!(solve(&t, &r).is_err());
    }

    /// Three nodes on one fabric when the biggest island holds two.
    #[test]
    fn a_set_larger_than_any_island_is_refused() {
        let t = two_pairs();
        let e = solve(&t, &req(&[("big", 3, Link::Rdma)], &[])).unwrap_err();
        assert!(e.contains("largest island has 2"), "{e}");
    }

    /// A node with its GPUs already reserved cannot host a rank, so the pair
    /// goes to the other fabric.
    #[test]
    fn a_full_node_sends_the_set_elsewhere() {
        let mut t = two_pairs();
        t.free_gpus.insert("nB".into(), 0.0);
        let s = solve(&t, &req(&[("tp0", 2, Link::Rdma)], &[])).unwrap();
        let got: BTreeSet<_> = s.sets["tp0"].iter().map(|m| m.node.clone()).collect();
        assert_eq!(got, ["nC", "nD"].map(String::from).into());
    }

    /// A loose set placed first can claim a node the cabled set needed, so
    /// the constrained set goes first whatever order the caller wrote.
    ///
    /// nE is off both fabrics and sorts first, so a loose set placed before
    /// the cabled one would claim nA and break the pair.
    #[test]
    fn a_constrained_set_is_placed_before_a_loose_one() {
        let mut t = two_pairs();
        for n in ["nC", "nD"] {
            t.free_gpus.remove(n);
            t.ports.remove(n);
        }
        t.islands.retain(|i| i.nodes.contains(&"nA".to_string()));
        t.ports
            .insert("n0".into(), vec![port("192.168.1.50", "eno1", &["lan"])]);
        t.free_gpus.insert("n0".into(), 1.0);
        t.hosts.insert("n0".into(), "gx10-spare".into());

        let s = solve(
            &t,
            &req(&[("loose", 1, Link::Any), ("cabled", 2, Link::Rdma)], &[]),
        )
        .expect("the cabled pair must not be broken up by the loose set");
        let cabled: BTreeSet<_> = s.sets["cabled"].iter().map(|m| m.node.clone()).collect();
        assert_eq!(cabled, ["nA", "nB"].map(String::from).into());
        assert_eq!(s.sets["loose"][0].node, "n0");
    }

    /// The same view resolves the same way, or one name would mean two
    /// placements.
    #[test]
    fn the_answer_is_stable() {
        let t = two_pairs();
        let r = req(
            &[("tp0", 2, Link::Rdma), ("tp1", 2, Link::Rdma)],
            &[("tp0", "tp1", Link::Any)],
        );
        let first = solve(&t, &r).unwrap();
        for _ in 0..8 {
            assert_eq!(solve(&t, &r).unwrap(), first);
        }
    }

    #[test]
    fn a_between_naming_no_set_is_refused() {
        let t = two_pairs();
        let r = req(&[("tp0", 2, Link::Rdma)], &[("tp0", "ghost", Link::Any)]);
        assert!(solve(&t, &r).unwrap_err().contains("ghost"));
    }

    /// Four boxes with two fabric ports each, cabled in a ring, with one
    /// subnet per cable. No three boxes share a cable, so the islands
    /// are the four cabled pairs.
    fn four_ring() -> Topology {
        let mut t = Topology::default();
        // (node, port toward the previous box, port toward the next box)
        let spec = [
            ("nA", "10.100.4.2", "10.100.1.1"),
            ("nB", "10.100.1.2", "10.100.2.1"),
            ("nC", "10.100.2.2", "10.100.3.1"),
            ("nD", "10.100.3.2", "10.100.4.1"),
        ];
        for (i, (n, p0, p1)) in spec.iter().enumerate() {
            let lan = format!("192.168.1.{}", 10 + i);
            t.ports.insert(
                n.to_string(),
                vec![
                    port(p0, "enp1s0f0np0", &["rdma"]),
                    port(p1, "enp1s0f1np1", &["rdma"]),
                    port(&lan, "enP7s7", &["lan"]),
                ],
            );
            t.free_gpus.insert(n.to_string(), 1.0);
            t.hosts.insert(n.to_string(), n.to_lowercase());
        }
        for (a, b) in [
            ("10.100.1.1", "10.100.1.2"),
            ("10.100.2.1", "10.100.2.2"),
            ("10.100.3.1", "10.100.3.2"),
            ("10.100.4.1", "10.100.4.2"),
        ] {
            t.links.insert((a.into(), b.into()), 0);
        }
        t
    }

    fn laid_out(name: &str, k: usize, layout: Layout) -> Request {
        let mut r = req(&[(name, k, Link::Rdma)], &[]);
        r.sets[0].layout = layout;
        r
    }

    fn order(s: &Solution, set: &str) -> Vec<String> {
        s.sets[set].iter().map(|m| m.node.clone()).collect()
    }

    /// The case this exists for: TP=4 over a switchless ring.
    #[test]
    fn a_ring_of_four_follows_the_cables() {
        let t = four_ring();
        let s = solve(&t, &laid_out("tp", 4, Layout::Ring)).unwrap();
        assert_eq!(order(&s, "tp"), ["nA", "nB", "nC", "nD"]);
        let a = &s.sets["tp"][0];
        let next = a.next.as_ref().unwrap();
        assert_eq!(
            (next.local.addr.as_str(), next.remote.addr.as_str()),
            ("10.100.1.1", "10.100.1.2")
        );
        let prev = a.prev.as_ref().unwrap();
        assert_eq!(
            (prev.to.as_str(), prev.local.addr.as_str()),
            ("nD", "10.100.4.2")
        );
        // A member binds its port toward the next member.
        assert_eq!(a.bind.addr, "10.100.1.1");
        assert_eq!(a.bind.iface.as_deref(), Some("enp1s0f1np1"));
    }

    #[test]
    fn a_mesh_of_four_is_refused_on_a_ring() {
        let t = four_ring();
        assert!(solve(&t, &laid_out("tp", 4, Layout::Mesh)).is_err());
    }

    /// A line ends at its last member, and each end has one neighbour.
    #[test]
    fn a_line_of_three_sits_on_the_ring() {
        let t = four_ring();
        let s = solve(&t, &laid_out("pp", 3, Layout::Line)).unwrap();
        assert_eq!(order(&s, "pp"), ["nA", "nB", "nC"]);
        let m = &s.sets["pp"];
        assert!(m[0].prev.is_none() && m[0].next.is_some());
        assert!(m[2].next.is_none());
        assert_eq!(m[2].bind.addr, "10.100.2.2");
    }

    /// A triangle needs a cable between two boxes the ring keeps apart.
    #[test]
    fn a_ring_of_three_is_refused_on_a_ring_of_four() {
        let t = four_ring();
        let e = solve(&t, &laid_out("tp", 3, Layout::Ring)).unwrap_err();
        assert!(e.contains("ring of 3"), "{e}");
    }

    /// Two boxes on one cable are a ring, a line and a mesh.
    #[test]
    fn one_cable_is_a_ring_of_two() {
        let t = four_ring();
        let s = solve(&t, &laid_out("tp", 2, Layout::Ring)).unwrap();
        let m = &s.sets["tp"];
        assert_eq!(order(&s, "tp"), ["nA", "nB"]);
        assert_eq!(
            m[0].prev.as_ref().unwrap().to,
            m[0].next.as_ref().unwrap().to
        );
    }

    /// B forwards between A and C. That link counts as reach, and a ring
    /// built on it would send one rank's traffic through another box.
    #[test]
    fn a_forwarded_link_closes_no_ring() {
        let mut t = four_ring();
        t.links
            .insert(("10.100.1.1".into(), "10.100.2.2".into()), 1);
        t.forwarded
            .insert(("10.100.1.1".into(), "10.100.2.2".into()));
        assert!(solve(&t, &laid_out("tp", 3, Layout::Ring)).is_err());
        t.forwarded.clear();
        // A, B, C would take C's links to B and to A on one port. A, C, D
        // uses two ports on every member.
        let s = solve(&t, &laid_out("tp", 3, Layout::Ring)).unwrap();
        assert_eq!(order(&s, "tp"), ["nA", "nC", "nD"]);
    }

    /// Four boxes with their first port on one switch, and n0-n1 and n2-n3
    /// cabled on their second ports. With `halves`, each port shows two
    /// interfaces on one wire, as a DGX Spark's does, each in its own
    /// subnet.
    fn switch_and_pairs(halves: bool) -> Topology {
        let mut t = Topology::default();
        let addrs = |i: usize, wire: &str| -> Vec<String> {
            let mut v = vec![format!("10.{wire}.0.{i}")];
            if halves {
                v.push(format!("10.{wire}.1.{i}"));
            }
            v
        };
        for i in 0..4 {
            let mut ports = Vec::new();
            for (wire, iface) in [("50", "sw"), ("60", "pair")] {
                for (h, a) in addrs(i, wire).iter().enumerate() {
                    ports.push(port(a, &format!("{iface}{h}"), &["rdma"]));
                }
            }
            t.ports.insert(format!("n{i}"), ports);
            t.free_gpus.insert(format!("n{i}"), 1.0);
        }
        let mut link = |a: &str, b: &str| {
            t.links.insert((a.to_string(), b.to_string()), 0);
        };
        for i in 0..4 {
            for j in i + 1..4 {
                for a in addrs(i, "50") {
                    for b in addrs(j, "50") {
                        link(&a, &b);
                    }
                }
            }
        }
        for (i, j) in [(0, 1), (2, 3)] {
            for a in addrs(i, "60") {
                for b in addrs(j, "60") {
                    link(&a, &b);
                }
            }
        }
        t
    }

    /// The port each member reaches its previous and next member by.
    fn sides(s: &Solution, set: &str) -> Vec<(String, String)> {
        let wire = |p: &Option<Path>| {
            let a = &p.as_ref().unwrap().local.addr;
            a.split('.').nth(1).unwrap().to_string()
        };
        s.sets[set]
            .iter()
            .map(|m| (wire(&m.prev), wire(&m.next)))
            .collect()
    }

    /// The case this exists for: each box rings over the switch on one side
    /// and its cable on the other, so every box uses both ports.
    #[test]
    fn a_ring_on_a_switch_and_pairs_uses_both_ports() {
        for halves in [false, true] {
            let t = switch_and_pairs(halves);
            let s = solve(&t, &laid_out("tp", 4, Layout::Ring)).unwrap();
            assert_eq!(order(&s, "tp"), ["n0", "n1", "n2", "n3"], "halves {halves}");
            let want: Vec<(String, String)> = [("50", "60"), ("60", "50")]
                .repeat(2)
                .into_iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect();
            assert_eq!(sides(&s, "tp"), want, "halves {halves}");
        }
    }

    /// Issue 3: four boxes whose two PCI roots share one switch port, each
    /// root on its own subnet. The prober links only same-subnet pairs, so
    /// every neighbour's address sits on the interface named with it, and
    /// each member still splits its two neighbours across the roots.
    #[test]
    fn a_ring_over_two_roots_pairs_each_address_with_its_interface() {
        let mut t = Topology::default();
        let roots = [("100", "enp1s0f0np0"), ("101", "enP2p1s0f0np0")];
        for i in 1..=4 {
            let ports = roots
                .iter()
                .map(|(net, iface)| port(&format!("10.{net}.0.{i}"), iface, &["rdma"]))
                .collect();
            t.ports.insert(format!("n{i}"), ports);
            t.free_gpus.insert(format!("n{i}"), 1.0);
        }
        for (net, _) in roots {
            for i in 1..=4 {
                for j in i + 1..=4 {
                    t.links
                        .insert((format!("10.{net}.0.{i}"), format!("10.{net}.0.{j}")), 0);
                }
            }
        }
        let s = solve(&t, &laid_out("tp", 4, Layout::Ring)).unwrap();
        let net = |a: &str| a.split('.').nth(1).unwrap().to_string();
        let on = |iface: &Option<String>| {
            roots
                .iter()
                .find(|(_, i)| Some(i.to_string()) == *iface)
                .unwrap()
                .0
                .to_string()
        };
        for m in &s.sets["tp"] {
            let (p, q) = (m.prev.as_ref().unwrap(), m.next.as_ref().unwrap());
            for l in [p, q] {
                assert_eq!(net(&l.remote.addr), on(&l.local.iface), "{m:?}");
                assert_eq!(net(&l.local.addr), net(&l.remote.addr), "{m:?}");
            }
            assert_ne!(p.local.iface, q.local.iface, "{m:?}");
        }
    }

    /// Seven boxes: A to F on one switch, each switch port showing two PCI
    /// roots with a subnet per root, and second-port cables A-B and F-G.
    /// The prober links only same-subnet pairs.
    fn switch_with_two_cables() -> Topology {
        let mut t = Topology::default();
        let roots = [("100", "enp1s0f0np0"), ("101", "enP2p1s0f0np0")];
        let names = ["nA", "nB", "nC", "nD", "nE", "nF", "nG"];
        for (i, n) in names.iter().enumerate() {
            let mut ports: Vec<Port> = Vec::new();
            if *n != "nG" {
                for (net, iface) in roots {
                    ports.push(port(&format!("10.{net}.0.{i}"), iface, &["rdma"]));
                }
            }
            let cable = match *n {
                "nA" | "nB" => Some("110"),
                "nF" | "nG" => Some("111"),
                _ => None,
            };
            if let Some(net) = cable {
                ports.push(port(&format!("10.{net}.0.{i}"), "enp1s0f1np1", &["rdma"]));
            }
            t.ports.insert(n.to_string(), ports);
            t.free_gpus.insert(n.to_string(), 1.0);
        }
        for (net, _) in roots {
            for i in 0..6 {
                for j in i + 1..6 {
                    t.links
                        .insert((format!("10.{net}.0.{i}"), format!("10.{net}.0.{j}")), 0);
                }
            }
        }
        t.links
            .insert(("10.110.0.0".into(), "10.110.0.1".into()), 0);
        t.links
            .insert(("10.111.0.5".into(), "10.111.0.6".into()), 0);
        t
    }

    #[test]
    fn a_switch_with_two_cables_places_each_layout() {
        let t = switch_with_two_cables();
        let net = |a: &str| a.split('.').nth(1).unwrap().to_string();

        // A ring of six on the switch splits every box across its roots.
        let mut r = laid_out("tp", 6, Layout::Ring);
        r.sets[0].ports = Ports::Split;
        let s = solve(&t, &r).unwrap();
        assert!(!order(&s, "tp").contains(&"nG".to_string()));
        for m in &s.sets["tp"] {
            let (p, q) = (m.prev.as_ref().unwrap(), m.next.as_ref().unwrap());
            assert_ne!(p.local.iface, q.local.iface, "{m:?}");
            for l in [p, q] {
                assert_eq!(net(&l.local.addr), net(&l.remote.addr), "{m:?}");
            }
        }

        // G has one neighbour, so no ring of seven, and a line of seven
        // ends at G.
        assert!(solve(&t, &laid_out("tp", 7, Layout::Ring)).is_err());
        let s = solve(&t, &laid_out("pp", 7, Layout::Line)).unwrap();
        let o = order(&s, "pp");
        assert!(o[0] == "nG" || o[6] == "nG", "{o:?}");

        // F reaches A to E by the switch and G by its cable: the only hub
        // of a star of seven.
        let mut r = laid_out("tp", 7, Layout::Star);
        r.driver = Some("nC".into());
        assert_eq!(order(&solve(&t, &r).unwrap(), "tp")[0], "nF");

        // A and B ring over both the cable and the switch.
        let mut r = laid_out("tp", 2, Layout::Ring);
        r.sets[0].ports = Ports::Split;
        let s = solve(&t, &r).unwrap();
        assert_eq!(order(&s, "tp"), ["nA", "nB"]);
    }

    /// Four boxes in a switchless ring. Box i's first port is cabled to box
    /// i+1's second, and each port shows two PCI roots with a subnet per
    /// root and cable.
    fn spark_ring() -> Topology {
        let mut t = Topology::default();
        for i in 0..4 {
            let prev = (i + 3) % 4;
            let mut ports = Vec::new();
            for r in 0..2 {
                ports.push(port(
                    &format!("10.{}.{i}.1", 100 + r),
                    &format!("f0r{r}"),
                    &["rdma"],
                ));
                ports.push(port(
                    &format!("10.{}.{prev}.2", 100 + r),
                    &format!("f1r{r}"),
                    &["rdma"],
                ));
            }
            t.ports.insert(format!("n{i}"), ports);
            t.free_gpus.insert(format!("n{i}"), 1.0);
        }
        for i in 0..4 {
            for r in 0..2 {
                t.links.insert(
                    (
                        format!("10.{}.{i}.1", 100 + r),
                        format!("10.{}.{i}.2", 100 + r),
                    ),
                    0,
                );
            }
        }
        t
    }

    fn linked_pairs(f: &Facts) -> BTreeMap<(String, String), usize> {
        let mut out = BTreeMap::new();
        for l in &f.links {
            *out.entry((l.a.clone(), l.b.clone())).or_default() += 1;
        }
        out
    }

    fn path(nodes: &[&str]) -> Vec<String> {
        nodes.iter().map(|n| n.to_string()).collect()
    }

    /// Each cable holds two links on different ports at both ends, one per
    /// root, and the opposite box is two hops away either way round.
    #[test]
    fn a_spark_ring_lists_both_roots_and_both_ways_round() {
        let t = spark_ring();
        let s = solve(&t, &laid_out("tp", 4, Layout::Ring)).unwrap();
        let f = &s.facts["tp"];
        let pairs = linked_pairs(f);
        let want: BTreeMap<(String, String), usize> =
            [("n0", "n1"), ("n0", "n3"), ("n1", "n2"), ("n2", "n3")]
                .iter()
                .map(|(a, b)| ((a.to_string(), b.to_string()), 2))
                .collect();
        assert_eq!(pairs, want);
        for (a, b) in want.keys() {
            let both: Vec<&LinkFact> = f.links.iter().filter(|l| (&l.a, &l.b) == (a, b)).collect();
            assert_ne!(both[0].a_group, both[1].a_group, "{both:?}");
            assert_ne!(both[0].b_group, both[1].b_group, "{both:?}");
        }
        assert_eq!(
            f.paths["n0"]["n2"],
            vec![path(&["n0", "n1", "n2"]), path(&["n0", "n3", "n2"])]
        );
        assert_eq!(f.paths["n1"].keys().collect::<Vec<_>>(), ["n3"]);
        let v = s.to_json(&t);
        assert_eq!(v["links"]["tp"].as_array().unwrap().len(), 8);
        assert_eq!(
            v["links"]["tp"][0]["a"]["port"],
            v["links"]["tp"][0]["a"]["addr"]
        );
        assert_eq!(v["paths"]["tp"]["n0"]["n2"][1], json!(["n0", "n3", "n2"]));
    }

    /// The facts depend on the members and the probes alone, so a star's
    /// driver and a ring's `ports` leave them as they are.
    #[test]
    fn facts_ignore_the_order_and_the_ports_rule() {
        let t = spark_ring();
        let mut r = laid_out("tp", 4, Layout::Ring);
        let auto = solve(&t, &r).unwrap().facts;
        r.sets[0].ports = Ports::Split;
        assert_eq!(solve(&t, &r).unwrap().facts, auto);
        let line = solve(&t, &laid_out("tp", 4, Layout::Line)).unwrap().facts;
        assert_eq!(line["tp"], auto["tp"]);
    }

    /// The basement: A to F on a switch, cables A-B and F-G. With GPUs free
    /// only on G, F, C and D, a line of four ends at G. Every switch box
    /// reaches every other directly, so only G's pairs have paths, all
    /// through F.
    #[test]
    fn a_line_off_a_switch_has_paths_only_through_its_cable() {
        let mut t = switch_with_two_cables();
        for n in ["nA", "nB", "nE"] {
            t.free_gpus.insert(n.into(), 0.0);
        }
        let s = solve(&t, &laid_out("pp", 4, Layout::Line)).unwrap();
        let f = &s.facts["pp"];
        let pairs: Vec<(String, String)> = linked_pairs(f).into_keys().collect();
        let want: Vec<(String, String)> = [("nC", "nD"), ("nC", "nF"), ("nD", "nF"), ("nF", "nG")]
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect();
        assert_eq!(pairs, want);
        let mut got: Vec<(String, String, Vec<Vec<String>>)> = Vec::new();
        for (from, row) in &f.paths {
            for (to, ps) in row {
                got.push((from.clone(), to.clone(), ps.clone()));
            }
        }
        let want: Vec<(String, String, Vec<Vec<String>>)> = [
            ("nC", "nG", ["nC", "nF", "nG"]),
            ("nD", "nG", ["nD", "nF", "nG"]),
            ("nG", "nC", ["nG", "nF", "nC"]),
            ("nG", "nD", ["nG", "nF", "nD"]),
        ]
        .iter()
        .map(|(a, b, p)| (a.to_string(), b.to_string(), vec![path(p)]))
        .collect();
        assert_eq!(got, want);

        // A star of seven: only G's pairs, through the hub.
        let mut r = laid_out("tp", 7, Layout::Star);
        r.driver = Some("nF".into());
        let f = &solve(&switch_with_two_cables(), &r).unwrap().facts["tp"];
        assert!(f.paths.iter().all(|(from, row)| {
            row.iter()
                .all(|(to, ps)| (from == "nG") != (to == "nG") && ps.len() == 1 && ps[0][1] == "nF")
        }));
        assert_eq!(f.paths["nG"].len(), 5);
    }

    /// Random graphs: the links are exactly the probed pairs between
    /// members, and the paths exactly the fewest-hop ones, read from the
    /// tables by brute force.
    #[test]
    fn facts_match_brute_force() {
        let mut rng = Rng(0x5851f42d4c957f2d);
        let mut checked = 0;
        for (n, ports, graphs) in [(5, 2, 300), (5, 3, 200), (6, 2, 100)] {
            for _ in 0..graphs {
                let t = port_graph(n, ports, &mut rng);
                for layout in [Layout::Ring, Layout::Line, Layout::Star] {
                    for k in 3..=n {
                        let Ok(s) = solve(&t, &laid_out("s", k, layout)) else {
                            continue;
                        };
                        let members: BTreeSet<String> =
                            s.sets["s"].iter().map(|m| m.node.clone()).collect();
                        check_facts(&t, &members, &s.facts["s"]);
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 500, "{checked}");
    }

    fn check_facts(t: &Topology, members: &BTreeSet<String>, f: &Facts) {
        let owner: BTreeMap<&String, &String> = t
            .ports
            .iter()
            .flat_map(|(n, ps)| ps.iter().map(move |p| (&p.addr, n)))
            .collect();
        let mut want: BTreeSet<(String, String, String, String)> = BTreeSet::new();
        for (x, y) in t.links.keys() {
            if t.forwarded.contains(&(x.clone(), y.clone()))
                || t.forwarded.contains(&(y.clone(), x.clone()))
            {
                continue;
            }
            let (Some(a), Some(b)) = (owner.get(x), owner.get(y)) else {
                continue;
            };
            if a == b || !members.contains(*a) || !members.contains(*b) {
                continue;
            }
            want.insert(if a < b {
                ((*a).clone(), (*b).clone(), x.clone(), y.clone())
            } else {
                ((*b).clone(), (*a).clone(), y.clone(), x.clone())
            });
        }
        let got: BTreeSet<(String, String, String, String)> = f
            .links
            .iter()
            .map(|l| {
                (
                    l.a.clone(),
                    l.b.clone(),
                    l.a_port.addr.clone(),
                    l.b_port.addr.clone(),
                )
            })
            .collect();
        assert_eq!(got, want, "links on {:?}", t.links);
        assert_eq!(got.len(), f.links.len());

        let adj = |a: &String, b: &String| {
            want.iter()
                .any(|(x, y, _, _)| (x == a && y == b) || (x == b && y == a))
        };
        let all: Vec<String> = members.iter().cloned().collect();
        for from in &all {
            for to in &all {
                if from == to || adj(from, to) {
                    assert!(f.paths.get(from).and_then(|r| r.get(to)).is_none());
                    continue;
                }
                // Every simple path through members, by brute force.
                let mut found: Vec<Vec<String>> = Vec::new();
                fn go(
                    at: &String,
                    to: &String,
                    all: &[String],
                    adj: &dyn Fn(&String, &String) -> bool,
                    cur: &mut Vec<String>,
                    found: &mut Vec<Vec<String>>,
                ) {
                    if at == to {
                        found.push(cur.clone());
                        return;
                    }
                    for m in all {
                        if !cur.contains(m) && adj(at, m) {
                            cur.push(m.clone());
                            go(m, to, all, adj, cur, found);
                            cur.pop();
                        }
                    }
                }
                go(from, to, &all, &adj, &mut vec![from.clone()], &mut found);
                let least = found.iter().map(Vec::len).min();
                let mut shortest: Vec<Vec<String>> = found
                    .into_iter()
                    .filter(|p| Some(p.len()) == least)
                    .collect();
                shortest.sort();
                let got = f
                    .paths
                    .get(from)
                    .and_then(|r| r.get(to))
                    .cloned()
                    .unwrap_or_default();
                assert_eq!(got, shortest, "{from} to {to} on {:?}", t.links);
            }
        }
    }

    /// Every graph on four nodes with each pair absent, cabled or forwarded.
    /// A forwarded pair is not a link, so it shows in neither `links` nor
    /// the hops of a path.
    #[test]
    fn facts_leave_out_forwarded_pairs() {
        let n = 4;
        let m = pairs(n).len() as u32;
        let mut checked = 0;
        for code in 0..3u32.pow(m) {
            let t = graph(n, |k| (code / 3u32.pow(k as u32) % 3) as u8);
            for layout in [Layout::Ring, Layout::Line, Layout::Star] {
                for k in 3..=n {
                    let Ok(s) = solve(&t, &laid_out("s", k, layout)) else {
                        continue;
                    };
                    let members: BTreeSet<String> =
                        s.sets["s"].iter().map(|m| m.node.clone()).collect();
                    check_facts(&t, &members, &s.facts["s"]);
                    checked += 1;
                }
            }
        }
        assert!(checked > 100, "{checked}");
    }

    /// The basement with its switch island, and with `extra` boxes cabled
    /// to F's further ports.
    fn basement(extra: &[&str]) -> Topology {
        let mut t = switch_with_two_cables();
        let sw = ["nA", "nB", "nC", "nD", "nE", "nF"];
        t.islands.push(crate::island::Island {
            nodes: sw.iter().map(|n| n.to_string()).collect(),
            addr: sw
                .iter()
                .enumerate()
                .map(|(i, n)| (n.to_string(), format!("10.100.0.{i}")))
                .collect(),
        });
        for (j, n) in extra.iter().enumerate() {
            let net = 120 + j;
            t.ports
                .get_mut("nF")
                .unwrap()
                .push(port(&format!("10.{net}.0.5"), "x", &["rdma"]));
            t.ports.insert(
                n.to_string(),
                vec![port(&format!("10.{net}.0.9"), "x", &["rdma"])],
            );
            t.free_gpus.insert(n.to_string(), 1.0);
            t.links
                .insert((format!("10.{net}.0.5"), format!("10.{net}.0.9")), 0);
        }
        t
    }

    fn any(k: usize) -> Request {
        laid_out("s", k, Layout::Any)
    }

    /// Mesh first, then ring, star and line, then the one-hop fallback.
    #[test]
    fn any_takes_the_first_layout_that_fits() {
        let t = basement(&[]);
        let s = solve(&t, &any(6)).unwrap();
        assert_eq!(s.layouts["s"], Layout::Mesh);
        assert!(!order(&s, "s").contains(&"nG".to_string()));

        // G reaches only F, so seven is no mesh and no ring. F reaches the
        // other six, so it is a star with F as the hub.
        let s = solve(&t, &any(7)).unwrap();
        assert_eq!(s.layouts["s"], Layout::Star);
        assert_eq!(order(&s, "s")[0], "nF");

        // A driver on C cannot be the hub. A line from G to C puts it on
        // an end.
        let mut r = any(7);
        r.driver = Some("nC".into());
        let s = solve(&t, &r).unwrap();
        assert_eq!(s.layouts["s"], Layout::Line);
        let o = order(&s, "s");
        assert!(o[0] == "nC" || o[6] == "nC", "{o:?}");
        assert!(o[0] == "nG" || o[6] == "nG", "{o:?}");

        // H also off F: still a star on F. With the driver on C, no line
        // holds both G and H, and every box is linked to every other or one
        // box, F, away.
        let t = basement(&["nH"]);
        assert_eq!(solve(&t, &any(8)).unwrap().layouts["s"], Layout::Star);
        let mut r = any(8);
        r.driver = Some("nC".into());
        let s = solve(&t, &r).unwrap();
        assert_eq!(s.layouts["s"], Layout::Hops);
        assert_eq!(order(&s, "s").len(), 8);
        let f = &s.facts["s"];
        assert_eq!(f.paths["nG"]["nH"], vec![path(&["nG", "nF", "nH"])]);
        assert_eq!(f.paths["nH"]["nA"], vec![path(&["nH", "nF", "nA"])]);
        let m = s.sets["s"].iter().find(|m| m.node == "nF").unwrap();
        assert!(m.prev.is_none() && m.next.is_none() && m.leaves.is_empty());
        assert_eq!(s.to_json(&t)["layouts"]["s"], "hops");

        // Ignoring the driver takes the first layout that fits again: the
        // star on F, with C a leaf.
        r.driver_rule = Driver::Ignore;
        let s = solve(&t, &r).unwrap();
        assert_eq!(s.layouts["s"], Layout::Star);
        assert_eq!(order(&s, "s")[0], "nF");

        let e = solve(&t, &any(9)).unwrap_err();
        assert!(e.contains("any layout of 9"), "{e}");
    }

    /// u0 and u1 are each cabled to all four v boxes. Every pair of those
    /// six is linked or one box apart. No box reaches
    /// all five others, and no line passes more than five boxes, so the
    /// six are no star and no line. A seventh, off v0 only, is three links
    /// from the other v boxes.
    #[test]
    fn any_keeps_every_pair_within_one_hop() {
        let mut t = Topology::default();
        let mut add = |n: &str| {
            t.ports.insert(n.into(), Vec::new());
            t.free_gpus.insert(n.into(), 1.0);
        };
        for n in ["u0", "u1", "v0", "v1", "v2", "v3", "w"] {
            add(n);
        }
        let mut cable = |a: &str, b: &str, net: usize| {
            let (x, y) = (format!("10.{net}.0.1"), format!("10.{net}.0.2"));
            t.ports.get_mut(a).unwrap().push(port(&x, "p", &["rdma"]));
            t.ports.get_mut(b).unwrap().push(port(&y, "p", &["rdma"]));
            t.links.insert((x, y), 0);
        };
        let mut net = 0;
        for u in ["u0", "u1"] {
            for v in ["v0", "v1", "v2", "v3"] {
                net += 1;
                cable(u, v, net);
            }
        }
        cable("v0", "w", 99);
        t.free_gpus.insert("w".into(), 0.0);
        let s = solve(&t, &any(6)).unwrap();
        assert_eq!(s.layouts["s"], Layout::Hops);
        assert_eq!(order(&s, "s"), ["u0", "u1", "v0", "v1", "v2", "v3"]);
        assert_eq!(
            s.facts["s"].paths["v0"]["v1"],
            vec![path(&["v0", "u0", "v1"]), path(&["v0", "u1", "v1"])]
        );
        t.free_gpus.insert("w".into(), 1.0);
        let e = solve(&t, &any(7)).unwrap_err();
        assert!(e.contains("through at most 1 others"), "{e}");
    }

    /// Random graphs: `any` answers with a ring where one exists, else a
    /// star, else a line, else a set that brute force confirms keeps every
    /// pair within one hop. It refuses exactly when brute force finds none.
    #[test]
    fn any_matches_brute_force() {
        let mut rng = Rng(0x9e6c63d0676a9a99);
        for (n, ports, graphs) in [(5, 2, 300), (6, 2, 150), (5, 3, 150)] {
            for _ in 0..graphs {
                let t = port_graph(n, ports, &mut rng);
                for k in 2..=n {
                    let ring = solve(&t, &laid_out("s", k, Layout::Ring)).is_ok();
                    let star = solve(&t, &laid_out("s", k, Layout::Star)).is_ok();
                    let line = solve(&t, &laid_out("s", k, Layout::Line)).is_ok();
                    let near = |members: &[String]| {
                        let link = |a: &String, b: &String| {
                            t.ports[a].iter().any(|x| {
                                t.ports[b].iter().any(|y| {
                                    t.links.contains_key(&(x.addr.clone(), y.addr.clone()))
                                        || t.links.contains_key(&(y.addr.clone(), x.addr.clone()))
                                })
                            })
                        };
                        members.iter().all(|a| {
                            members.iter().all(|b| {
                                a == b
                                    || link(a, b)
                                    || members.iter().any(|m| link(a, m) && link(m, b))
                            })
                        })
                    };
                    let mut sets: Vec<Vec<String>> = vec![Vec::new()];
                    for i in 0..n {
                        let more: Vec<Vec<String>> = sets
                            .iter()
                            .filter(|s| s.len() < k)
                            .map(|s| [s.clone(), vec![format!("n{i}")]].concat())
                            .collect();
                        sets.extend(more);
                    }
                    let hops = sets.iter().any(|s| s.len() == k && near(s));
                    let at = format!("any of {k} on {:?}", t.links);
                    match solve(&t, &any(k)) {
                        Ok(s) => {
                            let want = if ring {
                                Layout::Ring
                            } else if star {
                                Layout::Star
                            } else if line {
                                Layout::Line
                            } else {
                                Layout::Hops
                            };
                            assert_eq!(s.layouts["s"], want, "{at}");
                            if want == Layout::Hops {
                                assert!(near(&order(&s, "s")), "{at}");
                            }
                        }
                        Err(_) => assert!(!ring && !star && !line && !hops, "{at}"),
                    }
                    // With a driver, the answer places: a star's hub and a
                    // line's end are the driver's node whenever it is a
                    // member. A ring and a hops set do not depend on it.
                    for d in (0..n).map(|i| format!("n{i}")) {
                        let mut r = any(k);
                        r.driver = Some(d.clone());
                        let Ok(s) = solve(&t, &r) else {
                            assert!(!ring && !hops, "{at}, driver {d}");
                            continue;
                        };
                        let o = order(&s, "s");
                        let at_end = o.first() == Some(&d) || o.last() == Some(&d);
                        match s.layouts["s"] {
                            Layout::Star => assert!(!o.contains(&d) || o[0] == d, "{at}"),
                            Layout::Line => assert!(!o.contains(&d) || at_end, "{at}"),
                            _ => {}
                        }
                        if ring {
                            assert_eq!(s.layouts["s"], Layout::Ring, "{at}, driver {d}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn max_hops_is_one_on_an_any_set() {
        let shape = |layout: &str, extra: Value| {
            let mut set = json!({"name": "s", "bundles": 2, "layout": layout});
            for (k, v) in extra.as_object().unwrap() {
                set[k] = v.clone();
            }
            json!({ "sets": [set] })
        };
        assert_eq!(parse(&shape("any", json!({}))).unwrap().sets[0].max_hops, 1);
        assert_eq!(
            parse(&shape("any", json!({"max_hops": 1}))).unwrap().sets[0].layout,
            Layout::Any
        );
        assert!(parse(&shape("any", json!({"max_hops": 2})))
            .unwrap_err()
            .contains("must be 1"));
        assert!(parse(&shape("ring", json!({"max_hops": 1})))
            .unwrap_err()
            .contains("any layout"));
        assert!(parse(&shape("any", json!({"ports": "split"}))).is_ok());
        assert!(Layout::parse("hops").is_err());
    }

    #[test]
    fn the_driver_rule_is_first_or_ignore() {
        let shape = |driver: Value| {
            let mut v = json!({"sets": [{"name": "s", "bundles": 2}]});
            if !driver.is_null() {
                v["driver"] = driver;
            }
            v
        };
        assert_eq!(
            parse(&shape(Value::Null)).unwrap().driver_rule,
            Driver::First
        );
        assert_eq!(
            parse(&shape(json!("first"))).unwrap().driver_rule,
            Driver::First
        );
        assert_eq!(
            parse(&shape(json!("ignore"))).unwrap().driver_rule,
            Driver::Ignore
        );
        assert!(parse(&shape(json!("last")))
            .unwrap_err()
            .contains("expected first or ignore"));
        assert!(parse(&shape(json!(true)))
            .unwrap_err()
            .contains("first or ignore"));
    }

    /// Two boxes joined by two cables ring over both.
    #[test]
    fn two_cables_make_a_ring_of_two_on_both() {
        let mut t = Topology::default();
        for (n, i) in [("nA", 1), ("nB", 2)] {
            t.ports.insert(
                n.into(),
                vec![
                    port(&format!("10.70.0.{i}"), "p0", &["rdma"]),
                    port(&format!("10.71.0.{i}"), "p1", &["rdma"]),
                ],
            );
            t.free_gpus.insert(n.into(), 1.0);
        }
        t.links.insert(("10.70.0.1".into(), "10.70.0.2".into()), 0);
        t.links.insert(("10.71.0.1".into(), "10.71.0.2".into()), 0);
        let s = solve(&t, &laid_out("tp", 2, Layout::Ring)).unwrap();
        let sides = sides(&s, "tp");
        assert_ne!(sides[0].0, sides[0].1, "{sides:?}");
        assert_ne!(sides[1].0, sides[1].1, "{sides:?}");
    }

    /// The first member sends its first link from one port and must take
    /// the closing link on the other. nA's port 10.80 reaches
    /// both of nB's ports and sorts first, so leaving by it and closing over
    /// it is found first.
    #[test]
    fn a_ring_closes_on_the_first_members_other_port() {
        let mut t = Topology::default();
        t.ports.insert(
            "nA".into(),
            vec![
                port("10.80.0.1", "p0", &["rdma"]),
                port("10.90.0.1", "p1", &["rdma"]),
            ],
        );
        t.ports.insert(
            "nB".into(),
            vec![
                port("10.80.0.2", "p0", &["rdma"]),
                port("10.81.0.2", "p1", &["rdma"]),
            ],
        );
        for n in ["nA", "nB"] {
            t.free_gpus.insert(n.into(), 1.0);
        }
        for (a, b) in [
            ("10.80.0.1", "10.80.0.2"),
            ("10.80.0.1", "10.81.0.2"),
            ("10.90.0.1", "10.80.0.2"),
        ] {
            t.links.insert((a.into(), b.into()), 0);
        }
        let s = solve(&t, &laid_out("tp", 2, Layout::Ring)).unwrap();
        let m = &s.sets["tp"];
        for x in m {
            let (p, q) = (x.prev.as_ref().unwrap(), x.next.as_ref().unwrap());
            assert_ne!(p.local.addr, q.local.addr, "{m:?}");
        }
    }

    /// Boxes with one port each still ring, over the one port.
    #[test]
    fn a_ring_on_one_switch_shares_the_port() {
        let mut t = switch_and_pairs(false);
        t.links.retain(|(a, _), _| a.starts_with("10.50."));
        let s = solve(&t, &laid_out("tp", 4, Layout::Ring)).unwrap();
        assert_eq!(order(&s, "tp"), ["n0", "n1", "n2", "n3"]);
        assert!(sides(&s, "tp").iter().all(|(a, b)| a == "50" && b == "50"));
    }

    /// A mesh holds every ring, so a ring request on a cabled pair works.
    #[test]
    fn a_mesh_island_holds_a_ring() {
        let t = two_pairs();
        let s = solve(&t, &laid_out("tp", 2, Layout::Ring)).unwrap();
        assert_eq!(order(&s, "tp"), ["nA", "nB"]);
    }

    /// On a ring of four, each box is cabled to two others, so a star of
    /// three fits on every box and a star of four on none.
    #[test]
    fn a_star_of_three_puts_its_hub_on_the_driver() {
        let t = four_ring();
        let mut r = laid_out("tp", 3, Layout::Star);
        let s = solve(&t, &r).unwrap();
        assert_eq!(order(&s, "tp"), ["nA", "nB", "nD"]);

        r.driver = Some("nC".into());
        let s = solve(&t, &r).unwrap();
        assert_eq!(order(&s, "tp"), ["nC", "nB", "nD"]);
        let m = &s.sets["tp"];
        let leaves: Vec<(&str, &str, &str)> = m[0]
            .leaves
            .iter()
            .map(|p| (p.to.as_str(), p.local.addr.as_str(), p.remote.addr.as_str()))
            .collect();
        assert_eq!(
            leaves,
            [
                ("nB", "10.100.2.2", "10.100.2.1"),
                ("nD", "10.100.3.1", "10.100.3.2")
            ]
        );
        assert!(m[0].hub.is_none() && m[0].prev.is_none() && m[0].next.is_none());
        // The hub binds the port toward its first leaf, a leaf the port
        // toward the hub.
        assert_eq!(m[0].bind.addr, "10.100.2.2");
        let hub = m[2].hub.as_ref().unwrap();
        assert_eq!(
            (hub.to.as_str(), hub.remote.addr.as_str()),
            ("nC", "10.100.3.1")
        );
        assert_eq!(m[2].bind.addr, "10.100.3.2");
        assert!(m[2].leaves.is_empty());

        let e = solve(&t, &laid_out("tp", 4, Layout::Star)).unwrap_err();
        assert!(e.contains("star of 4"), "{e}");
    }

    /// Six boxes on one switch, and a seventh cabled to the second port of
    /// n2. A switch port reaches every box on the switch, so only n2
    /// reaches all six others, and n2 is the hub of a star of seven.
    fn switch_and_one_off() -> Topology {
        let mut t = Topology::default();
        for i in 0..7 {
            let n = format!("n{i}");
            let mut ports = vec![port(&format!("10.200.0.{i}"), "enp1s0f0np0", &["rdma"])];
            if i == 2 {
                ports.push(port("10.201.0.1", "enp1s0f1np1", &["rdma"]));
            }
            if i == 6 {
                ports = vec![port("10.201.0.2", "enp1s0f0np0", &["rdma"])];
            }
            t.ports.insert(n.clone(), ports);
            t.free_gpus.insert(n.clone(), 1.0);
            t.hosts.insert(n.clone(), n);
        }
        for i in 0..6 {
            for j in i + 1..6 {
                t.links
                    .insert((format!("10.200.0.{i}"), format!("10.200.0.{j}")), 0);
            }
        }
        t.links
            .insert(("10.201.0.1".into(), "10.201.0.2".into()), 0);
        t
    }

    #[test]
    fn a_switch_and_a_cable_make_a_star_of_seven() {
        let t = switch_and_one_off();
        let mut r = laid_out("tp", 7, Layout::Star);
        r.driver = Some("n0".into());
        let s = solve(&t, &r).unwrap();
        assert_eq!(order(&s, "tp"), ["n2", "n0", "n1", "n3", "n4", "n5", "n6"]);
        let hub = &s.sets["tp"][0];
        let ifaces: BTreeSet<&str> = hub
            .leaves
            .iter()
            .filter_map(|p| p.local.iface.as_deref())
            .collect();
        assert_eq!(ifaces, BTreeSet::from(["enp1s0f0np0", "enp1s0f1np1"]));
        let off = s.sets["tp"][6].hub.as_ref().unwrap();
        assert_eq!(off.remote.addr, "10.201.0.1");

        // Any box on the switch is the hub of a star of the six there.
        let mut r = laid_out("tp", 6, Layout::Star);
        r.driver = Some("n4".into());
        let s = solve(&t, &r).unwrap();
        assert_eq!(order(&s, "tp")[0], "n4");
        assert!(!order(&s, "tp").contains(&"n6".to_string()));
    }

    /// A driver whose node cannot be a hub leaves the hub to node order.
    #[test]
    fn a_driver_off_every_star_does_not_block_one() {
        let mut t = four_ring();
        t.free_gpus.insert("nC".into(), 0.0);
        let mut r = laid_out("tp", 3, Layout::Star);
        r.driver = Some("nC".into());
        assert_eq!(order(&solve(&t, &r).unwrap(), "tp"), ["nA", "nB", "nD"]);
    }

    /// A star leaves out a forwarded pair, so B forwarding between A and C
    /// does not make A a hub over C.
    #[test]
    fn a_forwarded_link_is_no_spoke() {
        let mut t = four_ring();
        t.links
            .insert(("10.100.1.1".into(), "10.100.2.2".into()), 1);
        t.forwarded
            .insert(("10.100.1.1".into(), "10.100.2.2".into()));
        assert!(solve(&t, &laid_out("tp", 4, Layout::Star)).is_err());
        t.forwarded.clear();
        let s = solve(&t, &laid_out("tp", 4, Layout::Star)).unwrap();
        assert_eq!(order(&s, "tp"), ["nA", "nB", "nC", "nD"]);
    }

    #[test]
    fn a_star_in_the_view_lists_its_links() {
        let t = four_ring();
        let s = solve(&t, &laid_out("tp", 3, Layout::Star)).unwrap();
        let v = s.to_json(&t);
        let hub = &v["sets"]["tp"][0];
        assert_eq!(hub["leaves"][1]["host"], "nd");
        assert_eq!(hub["leaves"][1]["iface"], "enp1s0f0np0");
        assert!(hub.get("hub").is_none());
        assert_eq!(v["sets"]["tp"][1]["hub"]["addr"], "10.100.1.1");
        assert!(v["sets"]["tp"][1].get("leaves").is_none());
    }

    #[test]
    fn layout_names() {
        assert_eq!(Layout::parse(""), Ok(Layout::Mesh));
        assert_eq!(Layout::parse("ring"), Ok(Layout::Ring));
        assert_eq!(Layout::parse("line"), Ok(Layout::Line));
        assert_eq!(Layout::parse("star"), Ok(Layout::Star));
        assert!(Layout::parse("tree").is_err());
        let r = parse(
            &json!({"sets": [{"name": "tp", "bundles": 4, "link": "rdma",
                                        "layout": "ring"}]}),
        )
        .unwrap();
        assert_eq!(r.sets[0].layout, Layout::Ring);
    }

    /// Every node pair on `n` nodes is absent, a cable or forwarded, as
    /// `state(pair_index)` says: 0, 1 or 2. Each node has one rdma port.
    fn graph(n: usize, state: impl Fn(usize) -> u8) -> Topology {
        let mut t = Topology::default();
        for i in 0..n {
            t.ports.insert(
                format!("n{i}"),
                vec![port(&format!("10.0.0.{i}"), "p", &["rdma"])],
            );
            t.free_gpus.insert(format!("n{i}"), 1.0);
        }
        for (k, (i, j)) in pairs(n).into_iter().enumerate() {
            let key = (format!("10.0.0.{i}"), format!("10.0.0.{j}"));
            match state(k) {
                1 => {
                    t.links.insert(key, 0);
                }
                2 => {
                    t.links.insert(key.clone(), 0);
                    t.forwarded.insert(key);
                }
                _ => {}
            }
        }
        t
    }

    fn pairs(n: usize) -> Vec<(usize, usize)> {
        (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .collect()
    }

    /// Read from the tables directly, so the check shares no code with the
    /// solver.
    fn cable(t: &Topology, a: &str, b: &str) -> bool {
        let (x, y) = (&t.ports[a][0].addr, &t.ports[b][0].addr);
        let there = |k: &dyn Fn(&(String, String)) -> bool| {
            k(&(x.clone(), y.clone())) || k(&(y.clone(), x.clone()))
        };
        there(&|p| t.links.contains_key(p)) && !there(&|p| t.forwarded.contains(p))
    }

    /// Whether `order` is a ring, line or star over cables.
    fn valid(t: &Topology, layout: Layout, order: &[String]) -> bool {
        let k = order.len();
        if layout == Layout::Star {
            return order[1..].iter().all(|n| cable(t, &order[0], n));
        }
        let steps = order.windows(2).all(|w| cable(t, &w[0], &w[1]));
        let closes = layout != Layout::Ring || k < 3 || cable(t, &order[k - 1], &order[0]);
        steps && closes
    }

    /// Every order of `k` distinct nodes out of `n`.
    fn orders(n: usize, k: usize) -> Vec<Vec<String>> {
        fn go(n: usize, k: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<String>>) {
            if cur.len() == k {
                out.push(cur.iter().map(|i| format!("n{i}")).collect());
                return;
            }
            for i in 0..n {
                if !cur.contains(&i) {
                    cur.push(i);
                    go(n, k, cur, out);
                    cur.pop();
                }
            }
        }
        let mut out = Vec::new();
        go(n, k, &mut Vec::new(), &mut out);
        out
    }

    /// Solve every size and layout on `t` and hold the answer to brute
    /// force: an answer is a chain over cables, and there is one exactly
    /// when some order of the nodes is.
    fn check_every_request(t: &Topology, n: usize) {
        for layout in [Layout::Ring, Layout::Line, Layout::Star] {
            for k in 1..=n {
                let exists = orders(n, k).iter().any(|o| valid(t, layout, o));
                let got = solve(t, &laid_out("s", k, layout));
                assert_eq!(got.is_ok(), exists, "{layout:?} of {k} on {:?}", t.links);
                let Ok(sol) = got else { continue };
                let order: Vec<String> = sol.sets["s"].iter().map(|m| m.node.clone()).collect();
                assert!(
                    valid(t, layout, &order),
                    "{layout:?} {order:?} on {:?}",
                    t.links
                );
                if layout == Layout::Ring {
                    assert_eq!(order.iter().min(), order.first(), "{order:?}");
                }
                for (i, m) in sol.sets["s"].iter().enumerate() {
                    let at = |j: usize| order[j].clone();
                    let next = match layout {
                        Layout::Ring if k > 1 => Some(at((i + 1) % k)),
                        Layout::Line if i + 1 < k => Some(at(i + 1)),
                        _ => None,
                    };
                    assert_eq!(m.next.as_ref().map(|p| p.to.clone()), next, "{order:?}");
                    let hub = (layout == Layout::Star && i > 0).then(|| at(0));
                    assert_eq!(m.hub.as_ref().map(|p| p.to.clone()), hub, "{order:?}");
                    let leaves: Vec<String> = m.leaves.iter().map(|p| p.to.clone()).collect();
                    let want = if layout == Layout::Star && i == 0 {
                        order[1..].to_vec()
                    } else {
                        Vec::new()
                    };
                    assert_eq!(leaves, want, "{order:?}");
                    // Both ends of a link name the same cable.
                    if let Some(p) = &m.next {
                        let back = sol.sets["s"][(i + 1) % k].prev.as_ref().unwrap();
                        assert_eq!(*back, p.reversed(), "{order:?}");
                    }
                    let toward = m.next.as_ref().or(m.prev.as_ref()).or(m.hub.as_ref());
                    if let Some(p) = toward.or(m.leaves.first()) {
                        assert_eq!(m.bind, p.local, "{order:?}");
                    }
                }
                assert_eq!(solve(t, &laid_out("s", k, layout)).unwrap(), sol);
                if layout == Layout::Star {
                    check_star_driver(t, n, k);
                }
            }
        }
    }

    /// A star of `k` puts its hub on the driver's node exactly when some
    /// star has its hub there.
    fn check_star_driver(t: &Topology, n: usize, k: usize) {
        for d in (0..n).map(|i| format!("n{i}")) {
            let mut r = laid_out("s", k, Layout::Star);
            r.driver = Some(d.clone());
            let can = orders(n, k)
                .iter()
                .any(|o| o[0] == d && valid(t, Layout::Star, o));
            let sol = solve(t, &r).unwrap();
            let order: Vec<String> = sol.sets["s"].iter().map(|m| m.node.clone()).collect();
            assert!(valid(t, Layout::Star, &order), "{order:?}");
            assert_eq!(order[0] == d, can, "driver {d}, {order:?} on {:?}", t.links);
        }
    }

    /// Every cable graph on four nodes holding one or two GPUs, and every
    /// star asking for one or two GPUs per member. Leaves are tried only in
    /// id order within one GPU count, and this checks no star is lost to
    /// that.
    #[test]
    fn every_star_with_mixed_gpu_counts_matches_brute_force() {
        let n = 4;
        let free = [1.0, 2.0, 1.0, 2.0];
        for bits in 0u32..1 << pairs(n).len() {
            let mut t = graph(n, |k| (bits >> k & 1) as u8);
            for (i, f) in free.iter().enumerate() {
                t.free_gpus.insert(format!("n{i}"), *f);
            }
            for k in 1..=n {
                for mask in 0u32..1 << k {
                    let bundles: Vec<f64> = (0..k).map(|i| 1.0 + (mask >> i & 1) as f64).collect();
                    let fits =
                        |o: &Vec<String>| o.iter().zip(&bundles).all(|(m, b)| t.free_gpus[m] >= *b);
                    let exists = orders(n, k)
                        .iter()
                        .any(|o| fits(o) && valid(&t, Layout::Star, o));
                    let mut r = laid_out("s", k, Layout::Star);
                    r.sets[0].bundles = bundles.clone();
                    let got = solve(&t, &r);
                    assert_eq!(got.is_ok(), exists, "{bundles:?} on {:?}", t.links);
                    if let Ok(sol) = got {
                        let order: Vec<String> =
                            sol.sets["s"].iter().map(|m| m.node.clone()).collect();
                        assert!(fits(&order) && valid(&t, Layout::Star, &order));
                    }
                }
            }
        }
    }

    /// Every graph on five nodes, each pair cabled or not.
    #[test]
    fn every_ring_line_and_star_on_five_nodes_matches_brute_force() {
        let n = 5;
        for bits in 0u32..1 << pairs(n).len() {
            check_every_request(&graph(n, |k| (bits >> k & 1) as u8), n);
        }
    }

    /// Every graph on four nodes, each pair absent, cabled or forwarded. A
    /// forwarded pair must never carry a ring, a line or a star's spoke.
    #[test]
    fn every_ring_line_and_star_on_four_nodes_skips_forwarded_pairs() {
        let n = 4;
        let m = pairs(n).len() as u32;
        for code in 0..3u32.pow(m) {
            check_every_request(&graph(n, |k| (code / 3u32.pow(k as u32) % 3) as u8), n);
        }
    }

    /// A small deterministic generator, so the random graphs are the same on
    /// every run.
    struct Rng(u64);

    impl Rng {
        fn below(&mut self, n: u64) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 % n
        }
    }

    /// `n` nodes with two ports each. Each port is on a switch, cabled to a
    /// port of another node, or unused.
    fn two_port_graph(n: usize, rng: &mut Rng) -> Topology {
        port_graph(n, 2, rng)
    }

    /// `n` nodes with `ports` ports each, as `two_port_graph` builds them.
    fn port_graph(n: usize, ports: usize, rng: &mut Rng) -> Topology {
        let mut t = Topology::default();
        let addr = |i: usize, p: usize| format!("10.0.{p}.{i}");
        for i in 0..n {
            t.ports.insert(
                format!("n{i}"),
                (0..ports)
                    .map(|p| port(&addr(i, p), "p", &["rdma"]))
                    .collect(),
            );
            t.free_gpus.insert(format!("n{i}"), 1.0);
        }
        let mut switch = Vec::new();
        let mut loose = Vec::new();
        for i in 0..n {
            for p in 0..ports {
                match rng.below(3) {
                    0 => switch.push((i, p)),
                    1 => loose.push((i, p)),
                    _ => {}
                }
            }
        }
        for (x, a) in switch.iter().enumerate() {
            for b in &switch[x + 1..] {
                if a.0 != b.0 {
                    t.links.insert((addr(a.0, a.1), addr(b.0, b.1)), 0);
                }
            }
        }
        while loose.len() >= 2 {
            let a = loose.remove(rng.below(loose.len() as u64) as usize);
            let b = loose.remove(rng.below(loose.len() as u64) as usize);
            if a.0 != b.0 {
                t.links.insert((addr(a.0, a.1), addr(b.0, b.1)), 0);
            }
        }
        t
    }

    /// The addresses `a` has a link to, read from the tables directly.
    fn far(t: &Topology, a: &str) -> BTreeSet<String> {
        t.links
            .keys()
            .filter_map(|(x, y)| {
                if x == a {
                    Some(y.clone())
                } else if y == a {
                    Some(x.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Whether some choice of links puts `order` on a ring or a line in
    /// which every member reaches its two neighbours by different ports.
    /// Two ports differ when they reach different far addresses, read from the
    /// tables directly.
    fn splits(t: &Topology, layout: Layout, order: &[String]) -> bool {
        holds(t, layout, order, Ports::Split)
    }

    /// Whether some choice of links puts `order` on a ring or a line in
    /// which every member's two links meet `ports`.
    fn holds(t: &Topology, layout: Layout, order: &[String], ports: Ports) -> bool {
        let reach = |a: &str| far(t, a);
        let links = |a: &str, b: &str| -> Vec<(String, String)> {
            let mut v = Vec::new();
            for x in &t.ports[a] {
                for y in &t.ports[b] {
                    let k = (x.addr.clone(), y.addr.clone());
                    let r = (y.addr.clone(), x.addr.clone());
                    if t.links.contains_key(&k) || t.links.contains_key(&r) {
                        v.push(k);
                    }
                }
            }
            v
        };
        let k = order.len();
        let steps = if layout == Layout::Ring && k > 1 {
            k
        } else {
            k - 1
        };
        let options: Vec<Vec<(String, String)>> = (0..steps)
            .map(|i| links(&order[i], &order[(i + 1) % k]))
            .collect();
        type Check<'c> = &'c dyn Fn(&[(String, String)]) -> bool;
        fn pick(
            options: &[Vec<(String, String)>],
            got: &mut Vec<(String, String)>,
            ok: Check,
        ) -> bool {
            if got.len() == options.len() {
                return ok(got);
            }
            for o in &options[got.len()] {
                got.push(o.clone());
                if pick(options, got, ok) {
                    return true;
                }
                got.pop();
            }
            false
        }
        let ok = |got: &[(String, String)]| {
            (0..k).all(|i| {
                let into = if i > 0 {
                    Some(&got[i - 1].1)
                } else if layout == Layout::Ring && k > 1 {
                    Some(&got[k - 1].1)
                } else {
                    None
                };
                let out = got.get(i).map(|l| &l.0);
                match (into, out) {
                    (Some(a), Some(b)) => (reach(a) == reach(b)) == (ports == Ports::Same),
                    _ => true,
                }
            })
        };
        pick(&options, &mut Vec::new(), &ok)
    }

    /// Random two-port graphs: whenever some ring or line has every member
    /// on different ports to its two neighbours, the answer is one.
    #[test]
    fn a_ring_or_line_uses_both_ports_whenever_one_can() {
        let mut rng = Rng(0x9e3779b97f4a7c15);
        let n = 4;
        for _ in 0..400 {
            let t = two_port_graph(n, &mut rng);
            for layout in [Layout::Ring, Layout::Line] {
                for k in 2..=n {
                    let want = orders(n, k).iter().any(|o| splits(&t, layout, o));
                    let Ok(s) = solve(&t, &laid_out("s", k, layout)) else {
                        assert!(!want, "{layout:?} of {k} on {:?}", t.links);
                        continue;
                    };
                    let got = s.sets["s"].iter().all(|m| match (&m.prev, &m.next) {
                        (Some(p), Some(q)) => far(&t, &p.local.addr) != far(&t, &q.local.addr),
                        _ => true,
                    });
                    assert_eq!(got, want, "{layout:?} of {k} on {:?}", t.links);
                }
            }
        }
    }

    /// Under `split` or `same`, a ring or line exists exactly when one meets
    /// the rule, and the answer meets it.
    #[test]
    fn a_ports_rule_is_met_or_refused() {
        let mut rng = Rng(0x2545f4914f6cdd1d);
        for (n, ports, graphs) in [(4, 2, 300), (5, 2, 150), (4, 3, 600), (5, 3, 100)] {
            check_ports_rule(n, ports, graphs, &mut rng);
        }
    }

    fn check_ports_rule(n: usize, ports: usize, graphs: usize, rng: &mut Rng) {
        for _ in 0..graphs {
            let t = port_graph(n, ports, rng);
            for ports in [Ports::Split, Ports::Same] {
                for layout in [Layout::Ring, Layout::Line] {
                    for k in 2..=n {
                        let want = orders(n, k).iter().any(|o| holds(&t, layout, o, ports));
                        let mut r = laid_out("s", k, layout);
                        r.sets[0].ports = ports;
                        let got = solve(&t, &r);
                        let at = format!("{ports:?} {layout:?} of {k} on {:?}", t.links);
                        assert_eq!(got.is_ok(), want, "{at}");
                        let Ok(s) = got else { continue };
                        for m in &s.sets["s"] {
                            if let (Some(p), Some(q)) = (&m.prev, &m.next) {
                                let same = far(&t, &p.local.addr) == far(&t, &q.local.addr);
                                assert_eq!(same, ports == Ports::Same, "{at}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn same_keeps_each_ring_member_on_one_port() {
        let t = switch_and_pairs(true);
        let mut r = laid_out("tp", 4, Layout::Ring);
        r.sets[0].ports = Ports::Same;
        let s = solve(&t, &r).unwrap();
        assert!(sides(&s, "tp").iter().all(|(a, b)| a == "50" && b == "50"));

        r.sets[0].ports = Ports::Split;
        let s = solve(&t, &r).unwrap();
        assert!(sides(&s, "tp").iter().all(|(a, b)| a != b));
        // A ring of three leaves one box's cable partner out.
        let mut r = laid_out("tp", 3, Layout::Ring);
        r.sets[0].ports = Ports::Split;
        let e = solve(&t, &r).unwrap_err();
        assert!(e.contains("different ports"), "{e}");
    }

    /// Eight pairs on a switch have no split ring of fifteen, since a
    /// split ring alternates cables and the switch. The split search used to
    /// visit every order, for over a minute, before `auto` fell back.
    #[test]
    fn an_odd_ring_on_pairs_falls_back_quickly() {
        let t = pairs_on_a_switch(8);
        let start = std::time::Instant::now();
        assert!(solve(&t, &laid_out("tp", 15, Layout::Ring)).is_ok());
        assert!(solve(&t, &laid_out("tp", 16, Layout::Ring)).is_ok());
        assert!(start.elapsed() < std::time::Duration::from_secs(10));
    }

    /// The pairs are parts of two, so the pre-check refuses an odd split
    /// ring before any search, and an even one still places.
    #[test]
    fn an_odd_split_ring_on_pairs_is_refused_without_a_search() {
        let t = pairs_on_a_switch(8);
        let groups = t.port_groups();
        let nodes: Vec<&NodeId> = t.free_gpus.keys().collect();
        let split = |k: usize| {
            let mut r = laid_out("tp", k, Layout::Ring);
            r.sets[0].ports = Ports::Split;
            r
        };
        assert!(split_ring_nodes(&t, &split(15).sets[0], &groups, &nodes).is_none());
        let start = std::time::Instant::now();
        assert!(solve(&t, &split(15)).is_err());
        assert!(start.elapsed() < std::time::Duration::from_secs(1));
        assert_eq!(
            split_ring_nodes(&t, &split(6).sets[0], &groups, &nodes).map(|n| n.len()),
            Some(16)
        );
        let s = solve(&t, &split(6)).unwrap();
        assert!(sides(&s, "tp").iter().all(|(a, b)| a != b));
    }

    /// A box that loses its cable partner keeps one port group, so a split
    /// ring leaves it out.
    #[test]
    fn a_box_whose_partner_is_busy_sits_out_a_split_ring() {
        let mut t = pairs_on_a_switch(3);
        t.free_gpus.insert("n01".into(), 0.0);
        let mut r = laid_out("tp", 4, Layout::Ring);
        r.sets[0].ports = Ports::Split;
        let s = solve(&t, &r).unwrap();
        assert_eq!(order(&s, "tp"), ["n02", "n03", "n04", "n05"]);
        r.sets[0].bundles = vec![1.0; 5];
        assert!(solve(&t, &r).is_err());
    }

    /// `pairs` cabled pairs, every box also on one switch.
    fn pairs_on_a_switch(pairs: usize) -> Topology {
        let n = pairs * 2;
        let mut t = Topology::default();
        for i in 0..n {
            t.ports.insert(
                format!("n{i:02}"),
                vec![
                    port(&format!("10.50.0.{i}"), "sw", &["rdma"]),
                    port(&format!("10.60.0.{i}"), "pair", &["rdma"]),
                ],
            );
            t.free_gpus.insert(format!("n{i:02}"), 1.0);
        }
        for i in 0..n {
            for j in i + 1..n {
                t.links
                    .insert((format!("10.50.0.{i}"), format!("10.50.0.{j}")), 0);
            }
        }
        for i in (0..n).step_by(2) {
            t.links
                .insert((format!("10.60.0.{i}"), format!("10.60.0.{}", i + 1)), 0);
        }
        t
    }

    #[test]
    fn ports_names() {
        let shape = |layout: &str, ports: &str| json!({"sets": [{"name": "tp", "bundles": 2, "layout": layout, "ports": ports}]});
        assert_eq!(
            parse(&shape("ring", "split")).unwrap().sets[0].ports,
            Ports::Split
        );
        assert_eq!(
            parse(&shape("line", "same")).unwrap().sets[0].ports,
            Ports::Same
        );
        assert_eq!(
            parse(&shape("mesh", "auto")).unwrap().sets[0].ports,
            Ports::Auto
        );
        assert!(parse(&shape("star", "split"))
            .unwrap_err()
            .contains("ports applies to"));
        assert!(parse(&shape("ring", "both")).is_err());
    }

    #[test]
    fn link_names() {
        assert_eq!(Link::parse("roce"), Ok(Link::Rdma));
        assert_eq!(Link::parse("rdma"), Ok(Link::Rdma));
        assert_eq!(Link::parse("ip"), Ok(Link::Any));
        assert!(Link::parse("carrier pigeon").is_err());
    }
}
