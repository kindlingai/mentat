//! The cluster graph, read from the daemon snapshots the router already
//! polls: the nodes, the links the daemons probed, and where each group
//! runs. The router's own MCP tools and the status page both read it.
//!
//! The head's snapshot holds every group. Each daemon's snapshot holds the
//! probes it ran itself, so the links come from all of them.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Map, Value};

use crate::Shared;

/// The fresh snapshots, one per watched daemon.
fn snapshots(shared: &Shared) -> Vec<Value> {
    let stale = shared.cfg.poll_interval * 3;
    shared
        .daemons
        .lock()
        .unwrap()
        .values()
        .filter(|v| v.seen.is_some_and(|s| s.elapsed() <= stale))
        .filter_map(|v| v.status.clone())
        .collect()
}

/// The snapshot of a daemon that names itself head, preferring the one
/// holding the most groups if two do.
fn head_of(snaps: &[Value]) -> Option<&Value> {
    snaps
        .iter()
        .filter(|s| s["node_id"].is_string() && s["node_id"] == s["head_node_id"])
        .max_by_key(|s| s["groups"].as_object().map(|g| g.len()).unwrap_or(0))
}

/// Node id to a name a person reads: the daemon's hostname where its own
/// snapshot is here, else the address its peers know it by. A hostname two
/// nodes share gets the node's address beside it.
fn names(snaps: &[Value]) -> BTreeMap<String, String> {
    let mut ip: BTreeMap<String, String> = BTreeMap::new();
    let mut host: BTreeMap<String, String> = BTreeMap::new();
    for s in snaps {
        for (id, p) in s["peers"].as_object().into_iter().flatten() {
            if let Some(a) = p["node_ip"].as_str() {
                ip.entry(id.clone()).or_insert_with(|| a.to_string());
            }
        }
        if let Some(id) = s["node_id"].as_str() {
            if let Some(a) = s["node_ip"].as_str() {
                ip.insert(id.to_string(), a.to_string());
            }
            if let Some(h) = s["hostname"].as_str() {
                host.insert(id.to_string(), h.to_string());
            }
        }
    }
    let mut uses: BTreeMap<&str, usize> = BTreeMap::new();
    for h in host.values() {
        *uses.entry(h.as_str()).or_default() += 1;
    }
    ip.keys()
        .chain(host.keys())
        .map(|id| {
            let label = match (host.get(id), ip.get(id)) {
                (Some(h), Some(a)) if uses[h.as_str()] > 1 => format!("{h} ({a})"),
                (Some(h), _) => h.clone(),
                (None, Some(a)) => a.clone(),
                (None, None) => id.clone(),
            };
            (id.clone(), label)
        })
        .collect()
}

fn name_of(names: &BTreeMap<String, String>, id: &str) -> String {
    names.get(id).cloned().unwrap_or_else(|| id.to_string())
}

/// MAC to the name of the node that owns it, from every published
/// `addr_macs`.
fn mac_owners(snaps: &[Value], names: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for s in snaps {
        let me = s["node_id"].as_str().unwrap_or_default();
        let rows = std::iter::once((me, &s["addr_macs"])).chain(
            s["peers"]
                .as_object()
                .into_iter()
                .flatten()
                .map(|(id, p)| (id.as_str(), &p["addr_macs"])),
        );
        for (id, macs) in rows {
            for mac in macs
                .as_object()
                .into_iter()
                .flatten()
                .filter_map(|(_, m)| m.as_str())
            {
                out.insert(mac.to_string(), name_of(names, id));
            }
        }
    }
    out
}

/// Each group's nodes, by name, with the node that announces its OpenAI
/// endpoint first and marked. A dead agent is left out.
pub fn group_nodes(shared: &Shared) -> BTreeMap<String, Vec<String>> {
    let snaps = snapshots(shared);
    let names = names(&snaps);
    let Some(head) = head_of(&snaps) else {
        return BTreeMap::new();
    };
    head["groups"]
        .as_object()
        .into_iter()
        .flatten()
        .map(|(g, v)| {
            let mut api = BTreeSet::new();
            let mut rest = BTreeSet::new();
            for a in v["agents"]
                .as_object()
                .into_iter()
                .flatten()
                .map(|(_, a)| a)
            {
                if !a["alive"].as_bool().unwrap_or(false) {
                    continue;
                }
                let n = name_of(&names, a["node_id"].as_str().unwrap_or_default());
                if a["services"]["openai"].is_object() {
                    api.insert(n);
                } else {
                    rest.insert(n);
                }
            }
            let mut out: Vec<String> = api.iter().map(|n| format!("{n} (api)")).collect();
            out.extend(rest.into_iter().filter(|n| !api.contains(n)));
            (g.clone(), out)
        })
        .collect()
}

/// `mentat_nodes`: every node, its addresses, and the agents on it.
pub fn nodes(shared: &Shared) -> Value {
    let snaps = snapshots(shared);
    let names = names(&snaps);
    let head = head_of(&snaps);
    let reported: BTreeSet<String> = snaps
        .iter()
        .filter_map(|s| s["head_node_id"].as_str())
        .filter(|h| !h.is_empty())
        .map(|h| name_of(&names, h))
        .collect();

    let mut agents: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for (g, v) in head
        .map(|h| &h["groups"])
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
    {
        for (id, a) in v["agents"].as_object().into_iter().flatten() {
            let gpus = a["machine"]["gpus"]
                .as_array()
                .map(|g| g.len())
                .unwrap_or(0);
            agents
                .entry(a["node_id"].as_str().unwrap_or_default().to_string())
                .or_default()
                .push(json!({
                    "agent": id,
                    "group": g,
                    "container": a["container"],
                    "alive": a["alive"],
                    "gpus": gpus,
                    "gpus_free": a["gpus_free"].as_array().map(|f| f.len()).unwrap_or(0),
                    "services": a["services"].as_object().map(|s| s.keys().cloned().collect::<Vec<_>>()),
                    "meta": a["meta"],
                }));
        }
    }

    let addrs = |s: &Value| -> Vec<Value> {
        s["addrs"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(|a| {
                json!({
                    "addr": a,
                    "iface": s["addr_ifaces"][a],
                    "tags": s["addr_tags"][a],
                    "mac": s["addr_macs"][a],
                })
            })
            .collect()
    };

    let mut rows: BTreeMap<String, Value> = BTreeMap::new();
    for s in &snaps {
        let Some(id) = s["node_id"].as_str() else {
            continue;
        };
        rows.insert(
            id.to_string(),
            json!({
                "node": name_of(&names, id),
                "node_ip": s["node_ip"],
                "watched": true,
                "alive": true,
                "addrs": addrs(s),
            }),
        );
    }
    // A node the router does not watch comes from its peers' tables.
    for s in &snaps {
        for (id, p) in s["peers"].as_object().into_iter().flatten() {
            rows.entry(id.clone()).or_insert_with(|| {
                json!({
                    "node": name_of(&names, id),
                    "node_ip": p["node_ip"],
                    "watched": false,
                    "alive": p["alive"],
                    "addrs": addrs(p),
                })
            });
        }
    }
    let head_id = head.and_then(|h| h["node_id"].as_str()).unwrap_or_default();
    let nodes: Vec<Value> = rows
        .into_iter()
        .map(|(id, mut row)| {
            row["head"] = json!(id == head_id);
            row["agents"] = json!(agents.remove(&id).unwrap_or_default());
            row
        })
        .collect();
    json!({
        "head": head.map(|_| name_of(&names, head_id)),
        // More than one entry is two daemons disagreeing about the head.
        "heads_reported": reported,
        "nodes": nodes,
    })
}

/// `mentat_links`: the address pairs the daemons probed, and the islands.
/// A pair that failed is listed only with `failed`.
pub fn links(shared: &Shared, failed: bool) -> Value {
    let snaps = snapshots(shared);
    let names = names(&snaps);
    let macs = mac_owners(&snaps, &names);
    let tagged = |s: &Value, a: &str| {
        s["addr_tags"][a]
            .as_array()
            .is_some_and(|t| t.iter().any(|t| t == "rdma"))
    };
    let mut out = Vec::new();
    for s in &snaps {
        let me = name_of(&names, s["node_id"].as_str().unwrap_or_default());
        for (peer_id, p) in s["peers"].as_object().into_iter().flatten() {
            for (local, row) in p["probes"].as_object().into_iter().flatten() {
                for (remote, r) in row.as_object().into_iter().flatten() {
                    let ok = r["ok"].as_bool().unwrap_or(false);
                    if !ok && !failed {
                        continue;
                    }
                    let mut link = Map::new();
                    link.insert("from".into(), json!(me));
                    link.insert("from_addr".into(), json!(local));
                    link.insert("to".into(), json!(name_of(&names, peer_id)));
                    link.insert("to_addr".into(), json!(remote));
                    link.insert("ok".into(), json!(ok));
                    link.insert("rdma".into(), json!(tagged(s, local) && tagged(p, remote)));
                    if ok {
                        link.insert("rtt_ms".into(), r["rtt_ms"].clone());
                        link.insert("direct".into(), r["direct"].clone());
                        if r["direct"] == false {
                            let mac = r["neighbour_mac"].as_str().unwrap_or_default();
                            link.insert(
                                "via".into(),
                                json!(macs.get(mac).cloned().unwrap_or(mac.into())),
                            );
                        }
                    } else {
                        link.insert("error".into(), r["error"].clone());
                    }
                    out.push(Value::Object(link));
                }
            }
        }
    }
    let islands: Vec<Value> = head_of(&snaps)
        .map(|h| &h["islands"])
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|i| {
            json!(i["addrs"]
                .as_object()
                .into_iter()
                .flatten()
                .map(|(id, a)| (name_of(&names, id), a.clone()))
                .collect::<BTreeMap<String, Value>>())
        })
        .collect();
    json!({"links": out, "islands": islands})
}

/// `mentat_group`: one group's agents, actors, placement groups and
/// claims, with node names in place of node ids.
pub fn group(shared: &Shared, name: &str) -> Result<Value, String> {
    let snaps = snapshots(shared);
    let names = names(&snaps);
    let head = head_of(&snaps).ok_or("no watched daemon names itself head")?;
    let groups = head["groups"].as_object().cloned().unwrap_or_default();
    let Some(g) = groups.get(name) else {
        let have: Vec<&String> = groups.keys().collect();
        return Err(format!("no group {name:?}. Groups: {have:?}"));
    };
    let node = |v: &Value| json!(name_of(&names, v.as_str().unwrap_or_default()));
    let rows = |key: &str, f: &dyn Fn(&Value) -> Value| -> Map<String, Value> {
        g[key]
            .as_object()
            .into_iter()
            .flatten()
            .map(|(id, v)| (id.clone(), f(v)))
            .collect()
    };
    Ok(json!({
        "group": name,
        "gpus_total": g["gpus_total"],
        "gpus_used": g["gpus_used"],
        "agents": rows("agents", &|a| {
            let mut a = a.clone();
            a["node"] = node(&a["node_id"]);
            a
        }),
        "actors": rows("actors", &|a| {
            let mut a = a.clone();
            a["node"] = node(&a["node_id"]);
            a
        }),
        "placement_groups": g["placement_groups"],
        "claims": g["claims"],
    }))
}
