//! Fabric islands: which nodes can actually reach each other over an
//! RDMA-tagged link, derived from the mesh's own probes.
//!
//! Two boxes are on one fabric when the operator tagged an address on each
//! `rdma` AND a probe between those two addresses succeeded. Each half does
//! its own work. The tag picks out the links meant to hold NCCL, which
//! matters because the LAN reaches every box and a bare probe would call
//! that a fabric. The probe confirms the cabling, which matters because the
//! cluster numbers both of its fabrics out of the same subnet, so a cabled
//! pair and two unconnected boxes look alike to address arithmetic.
//!
//! Each daemon computes this for itself, from its own probes plus the
//! probes its peers publish in their status pushes. Soft consistency is
//! fine: one daemon decides a given placement group, the one its driver
//! rendezvoused with, so two daemons disagreeing for a few seconds cannot
//! split a placement.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::config::cfg;
use crate::state::{NodeId, SharedRef};
use mentat_common::logfmt::log;

pub use mentat_claim::island::{islands, FabricView, Island, Pruned};

/// What placement reads: the islands, and which nodes claim a fabric at all.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Fabrics {
    pub islands: Vec<Island>,
    /// Nodes holding at least one `rdma`-tagged address, whether or not a
    /// probe confirmed it. A group none of whose nodes appear here has not
    /// opted in and is placed without the fabric constraint, so tagging one
    /// pair cannot strand a deployment on the other.
    pub tagged: BTreeSet<NodeId>,
}

/// The islands plus the nodes that claim a fabric, which is what placement
/// needs: one to place inside, the other to know whether to constrain. Also
/// the ports pruned on the way.
fn fabrics(v: &FabricView) -> (Fabrics, Vec<Pruned>) {
    let (islands, pruned) = islands(v);
    let f = Fabrics {
        islands,
        tagged: v.rdma.keys().cloned().collect(),
    };
    (f, pruned)
}

/// Read the daemon's merged view: its own tags and probes, its peers', and
/// what its peers publish about their own peers.
///
/// The third source is what makes a four-node cluster work. This daemon
/// probes its own links only, so without its peers' published tables it
/// could never know whether two other boxes share a fabric.
fn gather(shared: &SharedRef) -> FabricView {
    let mut v = FabricView::default();
    let st = shared.st.lock().unwrap();
    note(
        &mut v,
        &st.node_id,
        crate::announce::local_addrs(),
        crate::announce::local_addr_tags(),
    );
    for p in st.peers.values().filter(|p| p.alive) {
        note(&mut v, &p.node_id, p.addrs.clone(), p.addr_tags.clone());
        for (local, remotes) in &p.probe_pairs {
            for (remote, r) in remotes {
                if r.ok {
                    v.ok_pairs.insert((local.clone(), remote.clone()));
                }
            }
        }
        // What this peer knows about the rest of the mesh.
        merge_published(&mut v, &p.last_status["peers"]);
    }
    v
}

/// Record a node's RDMA-tagged addresses. An untagged node is left out: the
/// tag marks a link as meant to hold NCCL.
fn note(v: &mut FabricView, node: &str, addrs: Vec<String>, tags: BTreeMap<String, Vec<String>>) {
    let tagged: Vec<String> = addrs
        .into_iter()
        .filter(|a| {
            tags.get(a)
                .map(|t| t.iter().any(|x| x == "rdma"))
                .unwrap_or(false)
        })
        .collect();
    if !tagged.is_empty() {
        v.rdma.insert(node.to_string(), tagged);
    }
}

/// Fold one peer's published peer table into the view.
///
/// A node's id is the key it is filed under. The entry itself omits the id,
/// so reading one out of the object skipped every entry, and this daemon
/// derived only the fabric it sits on: the third source contributed nothing
/// and a four-node cluster saw two halves.
fn merge_published(v: &mut FabricView, peers: &Value) {
    for (id, q) in peers.as_object().into_iter().flatten() {
        // A dead peer keeps its row so an operator can see a box that is
        // down, and its last probes are whatever was true before it went.
        // Placing a group on it would put ranks on a node nobody can reach.
        if !q["alive"].as_bool().unwrap_or(false) {
            continue;
        }
        note(v, id, str_list(&q["addrs"]), tag_map(&q["addr_tags"]));
        for (local, remotes) in q["probes"].as_object().into_iter().flatten() {
            for (remote, r) in remotes.as_object().into_iter().flatten() {
                if r["ok"].as_bool().unwrap_or(false) {
                    v.ok_pairs.insert((local.clone(), remote.clone()));
                }
            }
        }
    }
}

fn str_list(v: &Value) -> Vec<String> {
    v.as_array()
        .into_iter()
        .flatten()
        .filter_map(|a| a.as_str())
        .map(str::to_string)
        .collect()
}

fn tag_map(v: &Value) -> BTreeMap<String, Vec<String>> {
    v.as_object()
        .into_iter()
        .flatten()
        .map(|(k, t)| (k.clone(), str_list(t)))
        .collect()
}

/// Recompute islands and commit a change only once it has held still for
/// MENTAT_ISLAND_HOLD_DOWN_MS.
///
/// The hold-down is the election's argument applied to cables. A QSFP link
/// that flaps would otherwise move the island boundary between two
/// consecutive placements, and a placement group is the one thing here that
/// cannot be revised after the fact.
pub fn start(shared: SharedRef) {
    std::thread::spawn(move || run(shared));
}

fn run(shared: SharedRef) {
    let hold_down = Duration::from_millis(cfg().island_hold_down_ms);
    let tick = Duration::from_millis((cfg().island_hold_down_ms / 5).clamp(100, 1000));
    let mut candidate: Option<(Fabrics, Instant)> = None;
    // Tagged addresses no probe has ever confirmed. Named once each: the
    // tag records that the operator meant to cable this, so silence about it is
    // how a wrong tag survives a deployment.
    let mut unverified: HashSet<String> = HashSet::new();
    // Pruned ports, logged when pruning starts. The derivation runs every
    // tick, and a ring prunes the same ports each time.
    let mut pruned_now: HashSet<(NodeId, String)> = HashSet::new();
    loop {
        std::thread::sleep(tick);
        let view = gather(&shared);
        let (fresh, pruned) = fabrics(&view);

        let was = std::mem::take(&mut pruned_now);
        for p in pruned {
            let key = (p.node.clone(), p.addr.clone());
            if !was.contains(&key) {
                log(
                    "island_pruned",
                    &[
                        ("node", p.node),
                        ("addr", p.addr),
                        ("reaches", p.reaches.to_string()),
                        ("of", p.of.to_string()),
                        (
                            "why",
                            "a placement group must fit a set every member reaches".to_string(),
                        ),
                    ],
                );
            }
            pruned_now.insert(key);
        }

        let confirmed: HashSet<&String> = view.ok_pairs.iter().flat_map(|(a, b)| [a, b]).collect();
        for (node, addrs) in &view.rdma {
            for a in addrs {
                if confirmed.contains(a) {
                    unverified.remove(a);
                } else if unverified.insert(a.clone()) {
                    log(
                        "fabric_addr_unverified",
                        &[
                            ("node", node.clone()),
                            ("addr", a.clone()),
                            (
                                "why",
                                "tagged rdma, and no probe over it has succeeded. \
                                 Either the cable is out or the tag is on the wrong \
                                 interface"
                                    .to_string(),
                            ),
                        ],
                    );
                }
            }
        }

        {
            let st = shared.st.lock().unwrap();
            if st.fabrics == fresh {
                candidate = None;
                continue;
            }
        }
        let since = match &candidate {
            Some((c, t)) if *c == fresh => *t,
            _ => {
                candidate = Some((fresh, Instant::now()));
                continue;
            }
        };
        if since.elapsed() < hold_down {
            continue;
        }
        let mut st = shared.st.lock().unwrap();
        st.fabrics = fresh.clone();
        let islands: Vec<Value> = fresh
            .islands
            .iter()
            .map(|i| json!({ "nodes": i.nodes, "addrs": i.addr }))
            .collect();
        let tagged = fresh.tagged.len();
        st.emit_patch(
            "islands_changed",
            vec![crate::state::Patch::set(&["islands"], json!(islands))],
            &format!("{tagged} tagged nodes"),
        );
        candidate = None;
        // A placement group that was waiting for a fabric may fit now.
        crate::daemon::try_place(&mut st, &shared.cv);
        shared.cv.notify_all();
    }
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

    /// The cluster this exists for: four boxes, two cabled pairs, every
    /// fabric address out of the same subnet. Only the probes separate them.
    /// The reported gap: on a four-node cluster of two cabled pairs, each
    /// daemon derived only the fabric it sits on. A node's id is the key it
    /// is filed under, and reading one out of the object instead skipped
    /// every published peer, so this source contributed nothing.
    ///
    /// Shaped like a real /status peers map, which omits the id field.
    #[test]
    fn a_published_peer_table_is_read_by_its_keys() {
        let peers = json!({
            "6d656e7461743a31302e3130302e302e31": {
                "node_ip": "10.100.0.1",
                "alive": true,
                "addrs": ["192.168.1.70", "10.100.0.1"],
                "addr_tags": {"10.100.0.1": ["connectx", "rdma"], "192.168.1.70": ["lan"]},
                "probes": {"10.100.0.1": {"10.100.0.2": {"ok": true}}}
            },
            "6d656e7461743a31302e3130302e302e32": {
                "node_ip": "10.100.0.2",
                "alive": true,
                "addrs": ["192.168.1.77", "10.100.0.2"],
                "addr_tags": {"10.100.0.2": ["connectx", "rdma"], "192.168.1.77": ["lan"]},
                "probes": {"10.100.0.2": {"10.100.0.1": {"ok": true}}}
            }
        });
        let mut v = FabricView::default();
        merge_published(&mut v, &peers);

        assert_eq!(v.rdma.len(), 2, "both nodes contribute a tagged address");
        assert!(v
            .ok_pairs
            .contains(&("10.100.0.1".into(), "10.100.0.2".into())));

        let got = islands(&v).0;
        assert_eq!(got.len(), 1, "a fabric this daemon is not on: {got:?}");
        assert_eq!(
            got[0].addr.values().cloned().collect::<BTreeSet<_>>(),
            ["10.100.0.1", "10.100.0.2"].map(String::from).into()
        );
    }

    /// The reported bug: a box that went down kept contributing. A dead
    /// peer keeps its row so an operator can see it, and the probes on that
    /// row are whatever was true before it went. Placing on it would put
    /// ranks on a node nobody can reach.
    #[test]
    fn a_dead_published_peer_forms_no_island() {
        let peers = json!({
            "n1": {
                "alive": false,
                "addrs": ["10.100.0.1"],
                "addr_tags": {"10.100.0.1": ["rdma"]},
                "probes": {"10.100.0.1": {"10.100.0.2": {"ok": true}}}
            },
            "n2": {
                "alive": false,
                "addrs": ["10.100.0.2"],
                "addr_tags": {"10.100.0.2": ["rdma"]},
                "probes": {"10.100.0.2": {"10.100.0.1": {"ok": true}}}
            }
        });
        let mut v = FabricView::default();
        merge_published(&mut v, &peers);
        assert!(v.rdma.is_empty(), "a dead node offered a fabric address");
        assert!(islands(&v).0.is_empty());
    }

    /// One of a cabled pair going down breaks the pair.
    #[test]
    fn half_a_dead_pair_leaves_no_island() {
        let peers = json!({
            "n1": {"alive": true, "addrs": ["10.100.0.1"],
                   "addr_tags": {"10.100.0.1": ["rdma"]},
                   "probes": {"10.100.0.1": {"10.100.0.2": {"ok": true}}}},
            "n2": {"alive": false, "addrs": ["10.100.0.2"],
                   "addr_tags": {"10.100.0.2": ["rdma"]}, "probes": {}}
        });
        let mut v = FabricView::default();
        merge_published(&mut v, &peers);
        assert_eq!(v.rdma.len(), 1, "only the live half offers an address");
        assert!(islands(&v).0.is_empty(), "an island of one is not a fabric");
    }

    /// A LAN address every box shares is not a fabric, tagged or not.
    #[test]
    fn an_untagged_published_peer_contributes_nothing() {
        let peers = json!({
            "n1": {"alive": true, "addrs": ["192.168.1.70"],
                   "addr_tags": {"192.168.1.70": ["lan"]},
                   "probes": {"192.168.1.70": {"192.168.1.77": {"ok": true}}}},
            "n2": {"alive": true, "addrs": ["192.168.1.77"],
                   "addr_tags": {"192.168.1.77": ["lan"]}, "probes": {}}
        });
        let mut v = FabricView::default();
        merge_published(&mut v, &peers);
        assert!(v.rdma.is_empty(), "nothing tagged rdma");
        assert!(islands(&v).0.is_empty());
    }

    /// Tagged but unprobed nodes still count as opted in, so a group on them
    /// is constrained rather than silently placed across the LAN.
    #[test]
    fn tagging_a_node_opts_it_in_before_any_probe_succeeds() {
        let (f, _) = fabrics(&view(&[("n1", &["10.0.0.1"]), ("n2", &["10.0.0.2"])], &[]));
        assert!(f.islands.is_empty());
        assert_eq!(f.tagged.len(), 2);
    }
}
