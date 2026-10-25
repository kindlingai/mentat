//! A solved claim, the `view` of a `claim` reply.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::shape::Layout;
use crate::NodeId;

/// Where each set landed, and the links the answer rests on.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct View {
    /// Each set's members. A ring, a line or a star lists them in rank order.
    pub sets: BTreeMap<String, Vec<Member>>,
    /// One link per `between` entry, in request order.
    pub between: Vec<BetweenLink>,
    /// For each ring, line, star or `any` set, every probed link between two
    /// members that meets the set's `link`, with no forwarding box between.
    pub links: BTreeMap<String, Vec<LinkFact>>,
    /// For each ring, line, star or `any` set, every fewest-hop path through
    /// its members between two members with no link. Keyed by the first
    /// member, then the last.
    pub paths: BTreeMap<String, Paths>,
    /// What each `any` set was placed as.
    pub layouts: BTreeMap<String, Layout>,
}

/// For each first member, then each last member, the paths between them as
/// node ids from the first to the last.
pub type Paths = BTreeMap<NodeId, BTreeMap<NodeId, Vec<Vec<NodeId>>>>;

/// One member of a set.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Member {
    pub node: NodeId,
    pub host: String,
    pub vendor: String,
    /// The address for the member's rank to bind.
    pub bind: String,
    pub iface: Option<String>,
    pub tags: Vec<String>,
    /// In a ring or a line, the link to the member before this one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prev: Option<Neighbor>,
    /// In a ring or a line, the link to the member after this one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next: Option<Neighbor>,
    /// In a star, a leaf's link to the hub.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hub: Option<Neighbor>,
    /// In a star, the hub's link to each leaf, in member order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub leaves: Vec<Neighbor>,
}

/// A link from a member to another member.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Neighbor {
    pub node: NodeId,
    pub host: String,
    /// The far member's address.
    pub addr: String,
    /// The near member's address for this link.
    pub bind: String,
    /// The near member's interface for this link.
    pub iface: Option<String>,
}

/// The link that meets one `between` entry.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BetweenLink {
    pub from: BetweenEnd,
    pub to: BetweenEnd,
    pub rtt_ms: u64,
}

/// One end of a `between` link.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BetweenEnd {
    /// The set holding this end's node.
    pub set: String,
    pub node: NodeId,
    pub host: String,
    pub addr: String,
    pub iface: Option<String>,
}

/// One link between two members, with `a` the lower node id.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkFact {
    pub a: LinkEnd,
    pub b: LinkEnd,
    pub rtt_ms: u64,
}

/// One end of a link between two members.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkEnd {
    pub node: NodeId,
    pub host: String,
    pub addr: String,
    pub iface: Option<String>,
    /// The end's port group, named by its lowest address. Addresses on a
    /// shared wire, such as a DGX Spark's two interfaces per QSFP port,
    /// share a group.
    pub port: String,
}
