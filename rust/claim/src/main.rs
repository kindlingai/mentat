//! `mentat-claim`: the claim solver over JSON on stdin and stdout.

use std::io::{Read, Write};
use std::process::ExitCode;

use serde::de::DeserializeOwned;
use serde_json::{json, Value};

use mentat_claim::{Error, Shape, Topology, View};

const USAGE: &str = "\
usage: mentat-claim solve    < {\"shape\": ..., \"topology\": ..., \"driver\": ...}
       mentat-claim ranks    < {\"shape\": ..., \"view\": ..., \"set\": ..., \"driver\": ...}
       mentat-claim islands  < {\"topology\": ...}

solve prints the claim's view. ranks prints {\"layout\", \"members\"} for a
ring, line or star set of that view, with rank 0 on the driver's node, or
null for a set with no order. islands prints the fabric islands from the
topology's probes and rdma tags. driver is optional.

On failure, prints {\"code\", \"error\"} and exits 2 for a bad request or
shape, or 1 when no placement or rank order fits.";

/// A failure: its code, its message and the exit status.
type Fail = (String, String, u8);

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = match args.as_slice() {
        [c] if ["solve", "ranks", "islands"].contains(&c.as_str()) => c.as_str(),
        [f] if f == "--version" || f == "-V" => {
            println!("mentat-claim {}", env!("CARGO_PKG_VERSION"));
            return ExitCode::SUCCESS;
        }
        [f] if f == "--help" || f == "-h" => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    let out = match run(cmd) {
        Ok(v) => v,
        Err((code, error, status)) => {
            println!("{}", json!({"code": code, "error": error}));
            return ExitCode::from(status);
        }
    };
    if writeln!(std::io::stdout().lock(), "{out}").is_err() {
        return ExitCode::from(2);
    }
    ExitCode::SUCCESS
}

fn run(cmd: &str) -> Result<Value, Fail> {
    let bad = |code: &str, error: String| (code.to_string(), error, 2);
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| bad("bad_request", format!("reading stdin: {e}")))?;
    let req: Value = serde_json::from_str(&input)
        .map_err(|e| bad("bad_request", format!("stdin is not JSON: {e}")))?;
    if cmd == "islands" {
        let topology: Topology = field(&req, "topology", "bad_topology")?;
        return Ok(json!(mentat_claim::islands(&topology)));
    }
    let shape = Shape::from_json(&req["shape"]).map_err(refused)?;
    let driver = match &req["driver"] {
        Value::Null => None,
        Value::String(d) => Some(d.as_str()),
        v => {
            return Err(bad(
                "bad_request",
                format!("driver must be a node id, got {v}"),
            ))
        }
    };
    if cmd == "ranks" {
        let view: View = field(&req, "view", "bad_request")?;
        let set = req["set"]
            .as_str()
            .ok_or_else(|| bad("bad_request", "ranks needs a set name".into()))?;
        return Ok(json!(
            mentat_claim::ranks(&shape, &view, set, driver).map_err(refused)?
        ));
    }
    let topology: Topology = field(&req, "topology", "bad_topology")?;
    Ok(json!(
        mentat_claim::solve(&topology, &shape, driver).map_err(refused)?
    ))
}

fn field<T: DeserializeOwned>(req: &Value, name: &str, code: &str) -> Result<T, Fail> {
    serde_json::from_value(req[name].clone())
        .map_err(|e| (code.to_string(), format!("{name}: {e}"), 2))
}

fn refused(e: Error) -> Fail {
    let status = if matches!(e, Error::BadShape(_)) {
        2
    } else {
        1
    };
    (e.code().to_string(), e.message().to_string(), status)
}
