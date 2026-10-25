//! What a claim is solved against: the nodes, their ports and the probes
//! between those ports.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::island::{FabricView, Island};
use crate::NodeId;

/// The cluster a claim is solved against.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Topology {
    pub nodes: BTreeMap<NodeId, Node>,
    /// Every probe between two ports. A pair counts as linked in both
    /// directions when a probe either way answered.
    #[serde(default)]
    pub probes: Vec<Probe>,
    /// The fabric islands an `rdma` mesh is placed inside. None derives them
    /// from the probes and the `rdma` tags, as `islands` does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub islands: Option<Vec<Island>>,
}

/// One node of the cluster.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Node {
    /// The name a view shows. Empty shows the node id.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub host: String,
    #[serde(default)]
    pub ports: Vec<Port>,
    /// GPUs free on the node. None keeps the node out of every set. Its
    /// ports and probes still count toward port groups.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub free_gpus: Option<u32>,
    /// The GPU vendor the node offers. A set is placed on one vendor,
    /// because a collective spans only one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vendor: Option<String>,
}

/// One address a node listens on.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Port {
    pub addr: String,
    /// The interface the address sits on. None for an address from
    /// configuration.
    #[serde(default)]
    pub iface: Option<String>,
    /// Operator tags. An `rdma` set needs ports tagged `rdma`.
    #[serde(default)]
    pub tags: Vec<String>,
}

impl Port {
    pub(crate) fn rdma(&self) -> bool {
        self.tags.iter().any(|t| t == "rdma")
    }
}

/// One probe from a local address to a remote one.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Probe {
    pub from: String,
    pub to: String,
    /// Whether the probe answered. A probe that did not still marks a
    /// forwarded pair.
    #[serde(default = "answered")]
    pub ok: bool,
    #[serde(default)]
    pub rtt_ms: u64,
    /// False when the pair's ARP entry named a third box. That box forwards
    /// between the pair. A mesh uses such a pair, and a ring, a line or a
    /// star skips it. None when unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub direct: Option<bool>,
}

fn answered() -> bool {
    true
}

/// Derives islands from a topology's `rdma` tags and answered probes.
pub fn islands(t: &Topology) -> Vec<Island> {
    let mut v = FabricView::default();
    for (n, node) in &t.nodes {
        let tagged: Vec<String> = node
            .ports
            .iter()
            .filter(|p| p.rdma())
            .map(|p| p.addr.clone())
            .collect();
        if !tagged.is_empty() {
            v.rdma.insert(n.clone(), tagged);
        }
    }
    for p in t.probes.iter().filter(|p| p.ok) {
        v.ok_pairs.insert((p.from.clone(), p.to.clone()));
    }
    crate::island::islands(&v).0
}
