# mentat-claim

The claim solver from [mentat](https://github.com/kindlingai/mentat), as a
library and a command-line tool. The solver places named sets of nodes on a
probed cluster, so that each set's members link as a mesh, a ring, a line or
a star. For each member, it reports the address and interface to bind.

mentatd solves claims with this crate. Use the crate directly when the
cluster's nodes and probes come from somewhere other than mentat.

## Library

```rust
use mentat_claim::{solve, Shape, Topology};

let topology: Topology = serde_json::from_str(topology_json)?;
let shape = Shape::from_json(&serde_json::json!({
    "sets": [{"name": "tp", "bundles": 4, "link": "rdma", "layout": "ring"}]
}))?;
let view = solve(&topology, &shape, Some("node-a"))?;
for m in &view.sets["tp"] {
    println!("{} binds {} on {:?}", m.host, m.bind, m.iface);
}
```

`Shape`, `Topology` and `View` are plain structs. Build them in code, or
read and write them as JSON with serde. `Shape::from_json` reads the shape
of a mentat `claim` message, and `solve` holds a shape built in code to the
same rules.

The answer depends only on the input, so two callers with the same topology
get the same view.

## Command line

```bash
mentat-claim solve < request.json
```

The request is one JSON object:

```json
{
  "shape": {"sets": [{"name": "tp", "bundles": 2, "link": "rdma"}]},
  "topology": {
    "nodes": {
      "a": {"host": "box-a", "free_gpus": 1,
            "ports": [{"addr": "10.0.0.1", "iface": "fab0", "tags": ["rdma"]}]},
      "b": {"host": "box-b", "free_gpus": 1,
            "ports": [{"addr": "10.0.0.2", "iface": "fab0", "tags": ["rdma"]}]}
    },
    "probes": [{"from": "10.0.0.1", "to": "10.0.0.2", "rtt_ms": 1}]
  },
  "driver": "a"
}
```

`solve` prints the view. `mentat-claim islands` reads the same object and
prints the fabric islands from the probes and `rdma` tags.

On failure, the binary prints `{"code": ..., "error": ...}`. It exits 2 for
a bad request, topology or shape, and 1 when no placement fits.

## Input

A node:

- `ports`: each address the node listens on, with its `iface` and its
  operator `tags`. An `rdma` set needs ports tagged `rdma`.
- `free_gpus`: GPUs free on the node. A node without it stays out of every
  set. Its ports and probes still count.
- `host`: the name the view shows. The node id stands in when it is empty.
- `vendor`: the node's GPU vendor. A set is placed on one vendor.

A probe goes `from` a local address `to` a remote one. `ok` defaults to
true. `direct: false` marks a pair a third box forwards between: a mesh
counts it, and a ring, a line or a star does not.

`islands` is optional. Without it, the islands come from the probes and the
`rdma` tags.

The shape format and the view are described in mentat's
[PROTOCOL.md](https://github.com/kindlingai/mentat/blob/main/PROTOCOL.md),
under `claim`.

## Rank order

With the default `"driver": "first"`, `driver` names the node that will hold
rank 0, as vLLM requires. A star puts its hub there when it can. The view
lists a ring or a line in its solved order. To start one at the driver,
rotate the ring, or reverse the line and swap each member's `prev` and
`next`. With `"driver": "ignore"`, `driver` has no effect.
