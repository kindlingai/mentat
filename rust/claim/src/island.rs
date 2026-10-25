//! Fabric islands: which nodes reach each other over an RDMA-tagged link,
//! derived from probes.
//!
//! Two boxes are on one fabric when the operator tagged an address on each
//! `rdma` AND a probe between those two addresses succeeded. Each half does
//! its own work. The tag picks out the links meant to hold NCCL, which
//! matters because the LAN reaches every box and a bare probe would call
//! that a fabric. The probe confirms the cabling, which matters because a
//! cluster can number two fabrics out of one subnet, so a cabled pair and
//! two unconnected boxes look alike to address arithmetic.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use serde::{Deserialize, Serialize};

use crate::NodeId;

/// A set of nodes that are mutually reachable over RDMA-tagged addresses.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Island {
    /// Members, sorted, so two islands compare by value.
    pub nodes: Vec<NodeId>,
    /// The address each member listens on inside this island. Every member
    /// has one, and every pair of them replied to a probe. This is what a rank
    /// binds NCCL to.
    #[serde(rename = "addrs")]
    pub addr: BTreeMap<NodeId, String>,
}

/// Everything the derivation reads, gathered from a daemon's merged view.
#[derive(Default, Debug)]
pub struct FabricView {
    /// Node -> its RDMA-tagged addresses, in the node's own rank order.
    pub rdma: BTreeMap<NodeId, Vec<String>>,
    /// Address pairs that replied to a probe. Direction is not kept: a
    /// one-way link is not a thing this fabric can have, and both daemons
    /// probe anyway.
    pub ok_pairs: BTreeSet<(String, String)>,
}

impl FabricView {
    fn linked(&self, a: &str, b: &str) -> bool {
        self.ok_pairs.contains(&(a.to_string(), b.to_string()))
            || self.ok_pairs.contains(&(b.to_string(), a.to_string()))
    }

    /// Every address that could hold fabric traffic, as (node, address),
    /// sorted so every daemon walks them in the same order.
    fn ports(&self) -> Vec<(&NodeId, &String)> {
        let mut out: Vec<(&NodeId, &String)> = self
            .rdma
            .iter()
            .flat_map(|(n, addrs)| addrs.iter().map(move |a| (n, a)))
            .collect();
        out.sort();
        out
    }

    /// Whether two ports may hold traffic between their nodes. Two ports of
    /// one node never do: NCCL between ranks crosses boxes, and the prober
    /// never probes a node against itself.
    fn joins(&self, x: (&NodeId, &String), y: (&NodeId, &String)) -> bool {
        x.0 != y.0 && self.linked(x.1, y.1)
    }
}

/// The islands a view implies.
///
/// The graph is over ports -- one node's one address -- rather than over
/// nodes, because a rank binds one address and every other rank has to reach
/// that one. A node with two fabric ports on separate links joins an island
/// through whichever port reaches all of it, or through neither.
///
/// An island is a set every member reaches. A connected component is weaker
/// than that: placement puts a whole placement group inside one island, so a
/// member reaching only some of the others would leave ranks unable to talk.
/// Components are therefore pruned to mutually-connected sets, least-
/// connected port first. On the topology this exists for -- disjoint cabled
/// pairs -- components already are whole and nothing is pruned. The pruning
/// is what keeps a half-cabled mistake from being announced as a fabric.
///
/// Islands of one are dropped. A lone node is not on a fabric, and placement
/// treats a node as its own island anyway.
///
/// Returns the islands and the ports pruned to find them.
pub fn islands(v: &FabricView) -> (Vec<Island>, Vec<Pruned>) {
    let mut pruned = Vec::new();
    let ports = v.ports();
    let mut out: Vec<Island> = Vec::new();
    let mut placed: HashSet<(&NodeId, &String)> = HashSet::new();

    for seed in &ports {
        if placed.contains(seed) {
            continue;
        }
        // Connected component by breadth-first walk.
        let mut comp: Vec<(&NodeId, &String)> = vec![*seed];
        let mut queue = vec![*seed];
        while let Some(p) = queue.pop() {
            for q in &ports {
                if !comp.contains(q) && v.joins(p, *q) {
                    comp.push(*q);
                    queue.push(*q);
                }
            }
        }
        for p in &comp {
            placed.insert(*p);
        }
        // A rank binds one address, so a node joins through one port: the
        // one that reaches most of the component, then the lowest address.
        // Two ports of one node never join each other. A DGX Spark shows each
        // QSFP port as two interfaces on one wire, and counting the node's
        // own other interface would prune a whole cabled pair.
        let deg: BTreeMap<(&NodeId, &String), usize> = comp
            .iter()
            .map(|p| (*p, comp.iter().filter(|q| v.joins(*p, **q)).count()))
            .collect();
        comp.sort_by_key(|p| (p.0, std::cmp::Reverse(deg[p]), p.1));
        comp.dedup_by(|later, first| later.0 == first.0);
        // Prune to a mutually-connected set: drop the least-connected port
        // until every survivor reaches every other. Ties break on (node,
        // address) so every daemon prunes identically.
        loop {
            let worst = comp
                .iter()
                .map(|p| {
                    let deg = comp.iter().filter(|q| v.joins(*p, **q)).count();
                    (deg, *p)
                })
                .min();
            let Some((deg, port)) = worst else { break };
            if deg + 1 >= comp.len() {
                break;
            }
            pruned.push(Pruned {
                node: port.0.clone(),
                addr: port.1.clone(),
                reaches: deg,
                of: comp.len() - 1,
            });
            comp.retain(|q| q != &port);
        }
        if comp.len() < 2 {
            continue;
        }
        comp.sort();
        out.push(Island {
            nodes: comp.iter().map(|(n, _)| (*n).clone()).collect(),
            addr: comp
                .iter()
                .map(|(n, a)| ((*n).clone(), (*a).clone()))
                .collect(),
        });
    }
    out.sort_by(|a, b| a.nodes.cmp(&b.nodes));
    (out, pruned)
}

/// A port the pruning left out: it reached `reaches` of the `of` other ports
/// in its component.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pruned {
    pub node: NodeId,
    pub addr: String,
    pub reaches: usize,
    pub of: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(rdma: &[(&str, &[&str])], ok: &[(&str, &str)]) -> FabricView {
        FabricView {
            rdma: rdma
                .iter()
                .map(|(n, a)| (n.to_string(), a.iter().map(|s| s.to_string()).collect()))
                .collect(),
            ok_pairs: ok
                .iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect(),
        }
    }

    #[test]
    fn two_cabled_pairs_are_two_islands() {
        let v = view(
            &[
                ("n1", &["10.100.0.1"]),
                ("n2", &["10.100.0.2"]),
                ("n3", &["10.100.0.3"]),
                ("n4", &["10.100.0.4"]),
            ],
            &[("10.100.0.1", "10.100.0.2"), ("10.100.0.3", "10.100.0.4")],
        );
        let got = islands(&v).0;
        assert_eq!(
            got.iter().map(|i| i.nodes.clone()).collect::<Vec<_>>(),
            vec![vec!["n1", "n2"], vec!["n3", "n4"]],
        );
        assert_eq!(got[0].addr["n1"], "10.100.0.1");
        assert_eq!(got[1].addr["n4"], "10.100.0.4");
    }

    /// Same addresses, no probes: subnet arithmetic would call this one
    /// fabric of four. The probes call it nothing.
    #[test]
    fn a_shared_subnet_is_not_a_fabric() {
        let v = view(&[("n1", &["10.100.0.1"]), ("n2", &["10.100.0.2"])], &[]);
        assert!(islands(&v).0.is_empty());
    }

    /// An untagged link that probes fine is the LAN. Serving rides it;
    /// NCCL does not, so it does not form an island.
    #[test]
    fn an_untagged_link_makes_no_island() {
        let v = view(&[], &[("192.168.1.11", "192.168.1.12")]);
        assert!(islands(&v).0.is_empty());
    }

    /// A half-cabled mistake: n3 reaches n1 but not n2. Announcing all
    /// three would place a TP=3 group whose ranks cannot all talk, so n3 is
    /// pruned out.
    #[test]
    fn a_partly_connected_set_is_pruned_to_one_that_is_whole() {
        let v = view(
            &[
                ("n1", &["10.0.0.1"]),
                ("n2", &["10.0.0.2"]),
                ("n3", &["10.0.0.3"]),
            ],
            &[("10.0.0.1", "10.0.0.2"), ("10.0.0.1", "10.0.0.3")],
        );
        let (got, pruned) = islands(&v);
        assert_eq!(got.len(), 1, "{got:?}");
        // The loop logs this port once, when pruning starts.
        assert_eq!(pruned.len(), 1, "{pruned:?}");
        assert_eq!((pruned[0].reaches, pruned[0].of), (1, 2), "{pruned:?}");
        // Two survivors of equal standing exist ({n1,n2} and {n1,n3}). The
        // tie-break decides which, and decides it the same way on every
        // daemon. What is asserted here is what matters: the island is
        // whole, and smaller than the component it came from.
        assert_eq!(got[0].nodes.len(), 2, "{got:?}");
        assert!(got[0].nodes.contains(&"n1".to_string()), "{got:?}");
        let (x, y) = (&got[0].nodes[0], &got[0].nodes[1]);
        assert!(v.linked(&got[0].addr[x], &got[0].addr[y]), "{got:?}");
    }

    /// The trap a node graph misses: n1 has two fabric ports, and each
    /// reaches a different peer. Every node pair looks linked, so nothing
    /// prunes, but no single address of n1 reaches both -- and a rank binds
    /// one address. n1 is left out rather than handed an address that only
    /// half its island listens on.
    #[test]
    fn a_node_whose_ports_split_across_links_is_left_out() {
        let v = view(
            &[
                ("n1", &["10.0.0.1", "10.0.1.1"]),
                ("n2", &["10.0.0.2"]),
                ("n3", &["10.0.1.3"]),
            ],
            &[
                ("10.0.0.1", "10.0.0.2"),
                ("10.0.1.1", "10.0.1.3"),
                ("10.0.0.2", "10.0.1.3"),
            ],
        );
        let got = islands(&v).0;
        assert_eq!(got.len(), 1, "{got:?}");
        assert_eq!(got[0].nodes, vec!["n2", "n3"], "{got:?}");
    }

    /// A DGX Spark shows each QSFP port as two interfaces on one wire, each
    /// numbered in its own subnet. Each interface reaches both of the far
    /// box's, so the component has four ports. Counting a node's own other
    /// interface against it pruned the cabled pair.
    #[test]
    fn two_interfaces_on_one_cable_are_one_pair() {
        let v = view(
            &[
                ("nA", &["10.100.1.1", "10.100.5.1"]),
                ("nB", &["10.100.1.2", "10.100.5.2"]),
            ],
            &[
                ("10.100.1.1", "10.100.1.2"),
                ("10.100.1.1", "10.100.5.2"),
                ("10.100.5.1", "10.100.1.2"),
                ("10.100.5.1", "10.100.5.2"),
            ],
        );
        let (got, pruned) = islands(&v);
        assert!(pruned.is_empty(), "{pruned:?}");
        assert_eq!(got.len(), 1, "{got:?}");
        assert_eq!(got[0].nodes, vec!["nA", "nB"], "{got:?}");
        assert_eq!(got[0].addr["nA"], "10.100.1.1");
        assert_eq!(got[0].addr["nB"], "10.100.1.2");
    }

    /// Every member of an island holds an address, and every pair of those
    /// addresses replied to a probe. Placement hands each rank its entry, so a
    /// gap there is a rank with nothing to bind.
    #[test]
    fn every_member_has_an_address_that_reaches_every_other() {
        let v = view(
            &[
                ("n1", &["10.100.0.1"]),
                ("n2", &["10.100.0.2"]),
                ("n3", &["10.100.0.3"]),
            ],
            &[
                ("10.100.0.1", "10.100.0.2"),
                ("10.100.0.2", "10.100.0.3"),
                ("10.100.0.1", "10.100.0.3"),
            ],
        );
        let got = islands(&v).0;
        assert_eq!(got[0].nodes.len(), 3);
        for x in &got[0].nodes {
            for y in &got[0].nodes {
                if x != y {
                    assert!(v.linked(&got[0].addr[x], &got[0].addr[y]), "{x} {y}");
                }
            }
        }
    }

    /// A node with two fabric ports picks the one that held a probe, in
    /// its own rank order, so every daemon derives the same address.
    #[test]
    fn the_island_address_is_the_one_that_answered() {
        let v = view(
            &[("n1", &["10.0.0.1", "10.0.1.1"]), ("n2", &["10.0.1.2"])],
            &[("10.0.1.1", "10.0.1.2")],
        );
        let got = islands(&v).0;
        assert_eq!(got[0].addr["n1"], "10.0.1.1");
    }
}
