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
}

impl Layout {
    pub fn parse(s: &str) -> Result<Layout, String> {
        match s {
            "mesh" | "" => Ok(Layout::Mesh),
            "ring" => Ok(Layout::Ring),
            "line" => Ok(Layout::Line),
            other => Err(format!(
                "unknown layout {other:?}, expected mesh, ring or line"
            )),
        }
    }

    /// Whether the members come in an order that placement keeps.
    pub fn ordered(self) -> bool {
        self != Layout::Mesh
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
    ///
    /// Ranked by round trip, then by address, so a tie resolves the same way
    /// on every daemon. Ports are the node's own order otherwise.
    fn path(&self, a: &NodeId, b: &NodeId, link: Link, cable: bool) -> Option<Path> {
        let (pa, pb) = (self.ports.get(a)?, self.ports.get(b)?);
        let mut best: Option<Path> = None;
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
                let cand = Path {
                    from: a.clone(),
                    to: b.clone(),
                    local: x.clone(),
                    remote: y.clone(),
                    rtt_ms: rtt,
                };
                let better = match &best {
                    None => true,
                    Some(b0) => {
                        (cand.rtt_ms, &cand.local.addr, &cand.remote.addr)
                            < (b0.rtt_ms, &b0.local.addr, &b0.remote.addr)
                    }
                };
                if better {
                    best = Some(cand);
                }
            }
        }
        best
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
    /// The GPU vendor this member's bundles sit on. One vendor per set,
    /// since no collective spans vendors.
    pub vendor: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Solution {
    pub sets: BTreeMap<String, Vec<Member>>,
    pub between: Vec<Path>,
}

impl Solution {
    pub fn to_json(&self, t: &Topology) -> Value {
        let name = |n: &NodeId| t.hosts.get(n).cloned().unwrap_or_else(|| n.clone());
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
                        for (key, p) in [("prev", &m.prev), ("next", &m.next)] {
                            if let Some(p) = p {
                                row[key] = json!({
                                    "node": p.to,
                                    "host": name(&p.to),
                                    "addr": p.remote.addr,
                                    "bind": p.local.addr,
                                    "iface": p.local.iface,
                                });
                            }
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
        json!({
            "sets": sets,
            "between": self
                .between
                .iter()
                .map(|p| p.to_json(&t.hosts, &of_set))
                .collect::<Vec<_>>(),
        })
    }
}

/// Candidate node groups for a set, best first.
///
/// An `rdma` set may only sit inside an island, since that is the derived
/// answer to "these nodes are cabled together and every pair was probed". An
/// `any` set uses nodes in id order, which is arbitrary but identical on
/// every daemon.
fn candidates(t: &Topology, set: &SetReq, taken: &BTreeSet<NodeId>) -> Vec<Vec<NodeId>> {
    if set.layout.ordered() {
        let nodes: Vec<&NodeId> = t.free_gpus.keys().filter(|n| !taken.contains(*n)).collect();
        let mut out = Vec::new();
        walk(t, set, &nodes, &mut Vec::new(), &mut out);
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
            out
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
                vec![chosen]
            } else {
                Vec::new()
            }
        }
    }
}

/// A ring or line search stops after this many answers. Each is a
/// candidate the solver may reject for a later set, and a larger cluster
/// has many more.
const MAX_CHAINS: usize = 32;

/// Collect into `out` the node orders for a ring or line set that extend
/// `chain`: each member links to the next over a cable, and a ring's last
/// member links to its first.
///
/// A ring starts at its lowest node id, so each ring is found once per
/// direction. Placement rotates it to the driver's node.
fn walk<'a>(
    t: &Topology,
    set: &SetReq,
    nodes: &[&'a NodeId],
    chain: &mut Vec<&'a NodeId>,
    out: &mut Vec<Vec<NodeId>>,
) {
    let want = set.bundles.len();
    if out.len() >= MAX_CHAINS {
        return;
    }
    if chain.len() == want {
        let closed = set.layout != Layout::Ring
            || want < 3
            || t.path(chain[want - 1], chain[0], set.link, true).is_some();
        if closed {
            out.push(chain.iter().map(|n| (*n).clone()).collect());
        }
        return;
    }
    let i = chain.len();
    for &n in nodes {
        if chain.contains(&n) || t.free_gpus.get(n).copied().unwrap_or(0.0) < set.bundles[i] {
            continue;
        }
        if set.layout == Layout::Ring && chain.first().is_some_and(|first| n < *first) {
            continue;
        }
        if let Some(last) = chain.last() {
            if t.path(last, n, set.link, true).is_none() {
                continue;
            }
        }
        chain.push(n);
        walk(t, set, nodes, chain, out);
        chain.pop();
    }
}

/// The links from member `i` to the members before and after it.
fn neighbours(
    t: &Topology,
    set: &SetReq,
    nodes: &[NodeId],
    i: usize,
) -> Result<(Option<Path>, Option<Path>), String> {
    let k = nodes.len();
    let (prev, next) = match set.layout {
        Layout::Mesh => (None, None),
        Layout::Ring if k > 1 => (Some((i + k - 1) % k), Some((i + 1) % k)),
        Layout::Ring => (None, None),
        Layout::Line => ((i > 0).then(|| i - 1), (i + 1 < k).then_some(i + 1)),
    };
    let link = |j: Option<usize>| {
        j.map(|j| {
            t.path(&nodes[i], &nodes[j], set.link, true).ok_or_else(|| {
                format!(
                    "{} has no cable to {} in set {:?}",
                    nodes[i], nodes[j], set.name
                )
            })
        })
        .transpose()
    };
    Ok((link(prev)?, link(next)?))
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

    let mut chosen: BTreeMap<String, Vec<NodeId>> = BTreeMap::new();
    let mut taken: BTreeSet<NodeId> = BTreeSet::new();
    if !search(t, req, &order, 0, &mut chosen, &mut taken) {
        return Err(why_not(t, req));
    }

    let mut sets: BTreeMap<String, Vec<Member>> = BTreeMap::new();
    for s in &req.sets {
        let nodes = &chosen[&s.name];
        let mut members = Vec::new();
        for (i, n) in nodes.iter().enumerate() {
            let (prev, next) = neighbours(t, s, nodes, i)?;
            // A ring or line member binds the port toward its next member,
            // and a line's last member the port toward its previous one.
            let bind = match next.as_ref().or(prev.as_ref()) {
                Some(p) => p.local.clone(),
                None => bind_for(t, s, n, nodes)
                    .ok_or_else(|| format!("{n} has no address for set {:?}", s.name))?,
            };
            members.push(Member {
                node: n.clone(),
                bind,
                prev,
                next,
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
    Ok(Solution { sets, between })
}

fn search(
    t: &Topology,
    req: &Request,
    order: &[&SetReq],
    i: usize,
    chosen: &mut BTreeMap<String, Vec<NodeId>>,
    taken: &mut BTreeSet<NodeId>,
) -> bool {
    if i == order.len() {
        return paths_between(t, req, chosen).is_some();
    }
    let set = order[i];
    for cand in candidates(t, set, taken) {
        for n in &cand {
            taken.insert(n.clone());
        }
        chosen.insert(set.name.clone(), cand.clone());
        if search(t, req, order, i + 1, chosen, taken) {
            return true;
        }
        chosen.remove(&set.name);
        for n in &cand {
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
    chosen: &BTreeMap<String, Vec<NodeId>>,
) -> Option<Vec<Path>> {
    let mut out = Vec::new();
    for b in &req.between {
        let (from, to) = (chosen.get(&b.from)?, chosen.get(&b.to)?);
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
        if s.layout.ordered() {
            parts.push(format!(
                "set {:?} wants a {} of {want} nodes, each cabled to the next over {} links, \
                 and no such order of free nodes exists",
                s.name,
                if s.layout == Layout::Ring {
                    "ring"
                } else {
                    "line"
                },
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
            Ok(SetReq {
                name,
                bundles,
                link: Link::parse(s["link"].as_str().unwrap_or("ip"))?,
                layout: Layout::parse(s["layout"].as_str().unwrap_or("mesh"))?,
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
    Ok(Request { sets, between })
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
        let s = solve(&t, &laid_out("tp", 3, Layout::Ring)).unwrap();
        assert_eq!(order(&s, "tp"), ["nA", "nB", "nC"]);
    }

    /// A mesh holds every ring, so a ring request on a cabled pair works.
    #[test]
    fn a_mesh_island_holds_a_ring() {
        let t = two_pairs();
        let s = solve(&t, &laid_out("tp", 2, Layout::Ring)).unwrap();
        assert_eq!(order(&s, "tp"), ["nA", "nB"]);
    }

    #[test]
    fn layout_names() {
        assert_eq!(Layout::parse(""), Ok(Layout::Mesh));
        assert_eq!(Layout::parse("ring"), Ok(Layout::Ring));
        assert_eq!(Layout::parse("line"), Ok(Layout::Line));
        assert!(Layout::parse("star").is_err());
        let r = parse(
            &json!({"sets": [{"name": "tp", "bundles": 4, "link": "rdma",
                                        "layout": "ring"}]}),
        )
        .unwrap();
        assert_eq!(r.sets[0].layout, Layout::Ring);
    }

    #[test]
    fn link_names() {
        assert_eq!(Link::parse("roce"), Ok(Link::Rdma));
        assert_eq!(Link::parse("rdma"), Ok(Link::Rdma));
        assert_eq!(Link::parse("ip"), Ok(Link::Any));
        assert!(Link::parse("carrier pigeon").is_err());
    }
}
