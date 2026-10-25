//! `mentat-claim`: the claim solver over JSON on stdin and stdout.

use std::io::{Read, Write};
use std::process::ExitCode;

use serde_json::{json, Value};

use mentat_claim::{Error, Shape, Topology};

const USAGE: &str = "\
usage: mentat-claim solve    < {\"shape\": ..., \"topology\": ..., \"driver\": ...}
       mentat-claim islands  < {\"topology\": ...}

solve prints the claim's view. islands prints the fabric islands the
topology's probes and rdma tags imply. driver is optional.

On failure, prints {\"code\", \"error\"} and exits 2 for a bad request or
shape, or 1 when no placement fits.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = match args.as_slice() {
        [c] if c == "solve" || c == "islands" => c.as_str(),
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
    let mut input = String::new();
    if let Err(e) = std::io::stdin().read_to_string(&mut input) {
        return fail("bad_request", &format!("reading stdin: {e}"), 2);
    }
    let req: Value = match serde_json::from_str(&input) {
        Ok(v) => v,
        Err(e) => return fail("bad_request", &format!("stdin is not JSON: {e}"), 2),
    };
    let topology: Topology = match serde_json::from_value(req["topology"].clone()) {
        Ok(t) => t,
        Err(e) => return fail("bad_topology", &format!("topology: {e}"), 2),
    };
    let out = if cmd == "islands" {
        json!(mentat_claim::islands(&topology))
    } else {
        let shape = match Shape::from_json(&req["shape"]) {
            Ok(s) => s,
            Err(e) => return fail(e.code(), e.message(), 2),
        };
        let driver = match &req["driver"] {
            Value::Null => None,
            Value::String(d) => Some(d.as_str()),
            v => {
                return fail(
                    "bad_request",
                    &format!("driver must be a node id, got {v}"),
                    2,
                )
            }
        };
        match mentat_claim::solve(&topology, &shape, driver) {
            Ok(v) => json!(v),
            Err(e @ Error::BadShape(_)) => return fail(e.code(), e.message(), 2),
            Err(e) => return fail(e.code(), e.message(), 1),
        }
    };
    let mut stdout = std::io::stdout().lock();
    if writeln!(stdout, "{out}").is_err() {
        return ExitCode::from(2);
    }
    ExitCode::SUCCESS
}

fn fail(code: &str, error: &str, status: u8) -> ExitCode {
    println!("{}", json!({"code": code, "error": error}));
    ExitCode::from(status)
}
