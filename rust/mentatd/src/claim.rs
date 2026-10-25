//! The claim solver's input, read from this daemon's state. The solver lives
//! in the `mentat-claim` crate.

use std::collections::BTreeMap;

pub use mentat_claim::shape::canonical;
pub use mentat_claim::{Driver, Layout, Shape};
use mentat_claim::{Node, Port, Probe, Topology};

use crate::state::{NodeId, State};

/// The topology as this daemon currently sees it, with GPUs offered only on
/// the nodes where `group` has a live agent. Placement puts a group only on
/// its own agents.
///
/// Ports come from the peer table: each node's addresses, the operator's
/// tags and each address's interface. Probes come from self to
/// peer and peer to peer alike, so a fabric this daemon is not on still
/// counts.
pub fn topology(st: &State, group: &str) -> Topology {
    let mut t = Topology {
        islands: Some(st.fabrics.islands.clone()),
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
        let n = t.nodes.entry(node.clone()).or_default();
        if !ports.is_empty() {
            n.ports = ports;
        }
        n.host = host.to_string();
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
            p.last_status["hostname"]
                .as_str()
                .filter(|h| !h.is_empty())
                .unwrap_or(&p.node_ip),
            &p.addrs,
            &p.addr_tags,
            &p.addr_ifaces,
        );
        for (local, remotes) in &p.probe_pairs {
            for (remote, r) in remotes {
                t.probes.push(Probe {
                    from: local.clone(),
                    to: remote.clone(),
                    ok: r.ok,
                    rtt_ms: r.rtt_ms,
                    direct: r.direct,
                });
            }
        }
        for (_, q) in p.last_status["peers"].as_object().into_iter().flatten() {
            // A dead peer's last probes describe a box that has since gone.
            if !q["alive"].as_bool().unwrap_or(false) {
                continue;
            }
            for (local, remotes) in q["probes"].as_object().into_iter().flatten() {
                for (remote, r) in remotes.as_object().into_iter().flatten() {
                    t.probes.push(Probe {
                        from: local.clone(),
                        to: remote.clone(),
                        ok: r["ok"].as_bool().unwrap_or(false),
                        rtt_ms: r["rtt_ms"].as_u64().unwrap_or(0),
                        direct: r["direct"].as_bool(),
                    });
                }
            }
        }
    }
    let alive = || st.agents.values().filter(|a| a.alive);
    let mut free: BTreeMap<&NodeId, u32> = BTreeMap::new();
    for a in alive() {
        *free.entry(&a.node_id).or_default() += st.free_gpus_of(&a.id).len() as u32;
    }
    for a in alive() {
        // An agent can register a node no daemon peers for, which leaves it
        // holding GPUs with no address to bind. What it told us on register
        // is an address, so it stands in. It is untagged, which limits
        // that node to plain links.
        let n = t.nodes.entry(a.node_id.clone()).or_insert_with(|| Node {
            host: a.node_ip.clone(),
            ..Default::default()
        });
        if n.ports.is_empty() {
            n.ports = vec![Port {
                addr: a.node_ip.clone(),
                iface: None,
                tags: Vec::new(),
            }];
        }
        if n.vendor.is_none() {
            n.vendor = a.machine.gpus.first().map(|g| g.vendor.clone());
        }
        if a.group == group {
            n.free_gpus = free.get(&a.node_id).copied();
        }
    }
    t
}
