use std::io::Write;
use std::process::{Command, Stdio};

use serde_json::{json, Value};

/// Runs the binary with `input` on stdin, and returns its exit code and the
/// JSON it printed.
fn run(cmd: &str, input: &str) -> (i32, Value) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_mentat-claim"))
        .arg(cmd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    let v = serde_json::from_slice(&out.stdout).unwrap();
    (out.status.code().unwrap(), v)
}

fn pair() -> Value {
    json!({
        "nodes": {
            "a": {"host": "box-a", "ports": [{"addr": "10.0.0.1", "tags": ["rdma"]}], "free_gpus": 2},
            "b": {"ports": [{"addr": "10.0.0.2", "tags": ["rdma"]}], "free_gpus": 2},
        },
        "probes": [{"from": "10.0.0.1", "to": "10.0.0.2", "rtt_ms": 1}],
    })
}

#[test]
fn solve_prints_the_view() {
    let req = json!({
        "shape": {"sets": [{"name": "tp", "bundles": [2, 2], "link": "rdma"}]},
        "topology": pair(),
        "driver": "b",
    });
    let (code, v) = run("solve", &req.to_string());
    assert_eq!(code, 0, "{v}");
    let members = v["sets"]["tp"].as_array().unwrap();
    assert_eq!(members.len(), 2);
    assert_eq!(members[0]["host"], "box-a");
    // A node with no host shows its id.
    assert_eq!(members[1]["host"], "b");
    assert_eq!(members[1]["bind"], "10.0.0.2");
}

#[test]
fn islands_prints_the_islands() {
    let (code, v) = run("islands", &json!({"topology": pair()}).to_string());
    assert_eq!(code, 0, "{v}");
    assert_eq!(
        v,
        json!([{"nodes": ["a", "b"], "addrs": {"a": "10.0.0.1", "b": "10.0.0.2"}}])
    );
}

#[test]
fn a_failure_prints_its_code() {
    let cases = [
        ("not json", 2, "bad_request"),
        (
            r#"{"topology": {"nodes": 1}, "shape": {"sets": []}}"#,
            2,
            "bad_topology",
        ),
        (
            r#"{"topology": {"nodes": {}}, "shape": {}}"#,
            2,
            "bad_shape",
        ),
        (
            r#"{"topology": {"nodes": {}}, "shape": {"sets": []}, "driver": 3}"#,
            2,
            "bad_request",
        ),
        (
            r#"{"topology": {"nodes": {}}, "shape": {"sets": [{"name": "s", "bundles": 1}]}}"#,
            1,
            "unsolvable_claim",
        ),
    ];
    for (input, want, err) in cases {
        let (code, v) = run("solve", input);
        assert_eq!(
            (code, v["code"].as_str()),
            (want, Some(err)),
            "{input}: {v}"
        );
        assert!(v["error"].as_str().is_some_and(|e| !e.is_empty()), "{v}");
    }
}

/// `ranks` turns a solved ring to start at the driver, and refuses a line
/// with the driver inside it.
#[test]
fn ranks_starts_at_the_driver() {
    let view = |nodes: &[&str]| json!({"sets": {"s": nodes.iter().map(|n| json!({"node": n})).collect::<Vec<_>>()}});
    let req = |layout: &str, driver: &str| {
        json!({
            "shape": {"sets": [{"name": "s", "bundles": 3, "layout": layout}]},
            "view": view(&["a", "b", "c"]),
            "set": "s",
            "driver": driver,
        })
        .to_string()
    };
    let (code, v) = run("ranks", &req("ring", "b"));
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["layout"], "ring");
    let order: Vec<&str> = v["members"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["node"].as_str().unwrap())
        .collect();
    assert_eq!(order, ["b", "c", "a"]);

    let (code, v) = run("ranks", &req("line", "b"));
    assert_eq!(
        (code, v["code"].as_str()),
        (1, Some("driver_position")),
        "{v}"
    );

    let (code, v) = run("ranks", &req("mesh", "b"));
    assert_eq!((code, v), (0, Value::Null));
}
