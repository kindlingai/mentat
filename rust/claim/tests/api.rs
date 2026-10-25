use serde_json::json;

use mentat_claim::{solve, Error, Layout, Set, Shape, Topology};

/// Four boxes cabled in a ring on `10.0.0.x`, all four on a LAN.
fn ring_of_four() -> Topology {
    let mut nodes = serde_json::Map::new();
    let mut probes = Vec::new();
    for i in 0..4 {
        nodes.insert(
            format!("n{i}"),
            json!({
                "host": format!("box{i}"),
                "ports": [
                    {"addr": format!("10.0.{i}.1"), "iface": "fab0", "tags": ["rdma"]},
                    {"addr": format!("10.0.{}.2", (i + 3) % 4), "iface": "fab1", "tags": ["rdma"]},
                    {"addr": format!("192.168.0.{i}"), "iface": "eth0"},
                ],
                "free_gpus": 1,
                "vendor": "nvidia",
            }),
        );
        // Cable i joins n_i's `10.0.i.1` to the next box's `10.0.i.2`.
        probes
            .push(json!({"from": format!("10.0.{i}.1"), "to": format!("10.0.{i}.2"), "rtt_ms": 1}));
        for k in 0..4 {
            if k != i {
                probes.push(json!({"from": format!("192.168.0.{i}"), "to": format!("192.168.0.{k}"), "rtt_ms": 2}));
            }
        }
    }
    serde_json::from_value(json!({"nodes": nodes, "probes": probes})).unwrap()
}

fn shape(v: serde_json::Value) -> Shape {
    Shape::from_json(&v).unwrap()
}

#[test]
fn a_ring_follows_the_cables() {
    let t = ring_of_four();
    let s =
        shape(json!({"sets": [{"name": "tp", "bundles": 4, "link": "rdma", "layout": "ring"}]}));
    let v = solve(&t, &s, None).unwrap();
    let ring = &v.sets["tp"];
    assert_eq!(ring.len(), 4);
    for (i, m) in ring.iter().enumerate() {
        let next = &ring[(i + 1) % 4];
        assert_eq!(m.next.as_ref().unwrap().node, next.node);
        assert!(m.next.as_ref().unwrap().addr.starts_with("10.0."));
        assert_eq!(m.host, format!("box{}", &m.node[1..]));
        assert_eq!(m.vendor, "nvidia");
    }
    assert_eq!(v.links["tp"].len(), 4);
}

/// The view serializes to the field names in PROTOCOL.md, and reads back.
#[test]
fn a_view_round_trips_as_json() {
    let t = ring_of_four();
    let s = shape(json!({"sets": [{"name": "tp", "bundles": 4, "layout": "any", "link": "rdma"}]}));
    let v = solve(&t, &s, Some("n1")).unwrap();
    let j = serde_json::to_value(&v).unwrap();
    assert_eq!(j["layouts"]["tp"], "ring");
    let m = &j["sets"]["tp"][0];
    for key in [
        "node", "host", "vendor", "bind", "iface", "tags", "prev", "next",
    ] {
        assert!(m.get(key).is_some(), "member has no {key}: {m}");
    }
    assert!(m.get("hub").is_none() && m.get("leaves").is_none(), "{m}");
    let back: mentat_claim::View = serde_json::from_value(j).unwrap();
    assert_eq!(back, v);
}

#[test]
fn a_star_puts_its_hub_on_the_driver() {
    let t = ring_of_four();
    let s = shape(json!({"sets": [{"name": "s", "bundles": 3, "layout": "star"}]}));
    for d in ["n0", "n2"] {
        let v = solve(&t, &s, Some(d)).unwrap();
        assert_eq!(v.sets["s"][0].node, d);
        assert_eq!(v.sets["s"][0].leaves.len(), 2);
    }
}

/// A node without `free_gpus` stays out of every set. Its ports still count.
#[test]
fn a_node_without_gpus_is_not_placed() {
    let mut t = ring_of_four();
    t.nodes.get_mut("n3").unwrap().free_gpus = None;
    let mesh = shape(json!({"sets": [{"name": "s", "bundles": 3}]}));
    let v = solve(&t, &mesh, None).unwrap();
    assert!(v.sets["s"].iter().all(|m| m.node != "n3"));
    let ring =
        shape(json!({"sets": [{"name": "s", "bundles": 4, "link": "rdma", "layout": "ring"}]}));
    let e = solve(&t, &ring, None).unwrap_err();
    assert_eq!(e.code(), "unsolvable_claim");
}

#[test]
fn a_typed_shape_is_held_to_the_json_rules() {
    let t = ring_of_four();
    let set = |layout, n| Set {
        name: "s".into(),
        bundles: vec![1; n],
        layout,
        ..Default::default()
    };
    let hops = Shape {
        sets: vec![set(Layout::Hops, 2)],
        ..Default::default()
    };
    assert!(matches!(solve(&t, &hops, None), Err(Error::BadShape(_))));
    let huge = Shape {
        sets: vec![set(Layout::Mesh, mentat_claim::shape::MAX_MEMBERS + 1)],
        ..Default::default()
    };
    assert!(matches!(solve(&t, &huge, None), Err(Error::BadShape(_))));
    let e = Shape::from_json(&json!({"sets": [{"name": "s", "bundles": 100000}]})).unwrap_err();
    assert_eq!(e.code(), "bad_shape");
}

/// Each cable of the ring is an island of two, since no three boxes share
/// a fabric. An `rdma` mesh of two fits on one cable, and a mesh of three fails. Given
/// islands replace the derived ones.
#[test]
fn islands_come_from_the_probes() {
    let mut t = ring_of_four();
    let islands = mentat_claim::islands(&t);
    let pairs: Vec<Vec<String>> = islands.iter().map(|i| i.nodes.clone()).collect();
    assert_eq!(
        pairs,
        [["n0", "n1"], ["n0", "n3"], ["n1", "n2"], ["n2", "n3"]]
    );
    let mesh = |n| shape(json!({"sets": [{"name": "s", "bundles": n, "link": "rdma"}]}));
    let v = solve(&t, &mesh(2), None).unwrap();
    let placed: Vec<&str> = v.sets["s"].iter().map(|m| m.node.as_str()).collect();
    assert_eq!(placed, ["n0", "n1"]);
    assert!(solve(&t, &mesh(3), None).is_err());

    t.islands = Some(vec![islands[3].clone()]);
    let v = solve(&t, &mesh(2), None).unwrap();
    let placed: Vec<&str> = v.sets["s"].iter().map(|m| m.node.as_str()).collect();
    assert_eq!(placed, ["n2", "n3"]);
}
