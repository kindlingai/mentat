# Changelog

The protocol stays at 0.99 in every release below.

## 0.25.0 (2026-10-25)

### Added

- The claim solver is its own crate, `mentat-claim`, with typed `Shape`,
  `Topology` and `View` structs and a `mentat-claim` binary that reads a
  shape and a topology as JSON. mentatd solves claims with it. The
  artifacts image, the release tarballs and crates.io carry it.
- `mentat_claim::ranks` and `mentat-claim ranks` put a ring, line or star
  set in rank order for a driver, the order mentatd places in.

### Fixed

- A shape asking for more than 4096 members in one set is refused. A count
  such as `"bundles": 1000000000` used to allocate one bundle per member
  before the solve.

## 0.24.0 (2026-10-25)

### Changed

- A claim solves over the nodes where the claiming client's group has a
  live agent. It used to solve over every live agent, and could choose a
  node where the group had no agent, which placement could never use.
- A client with no session holds its claims while any of its links is
  open, and lets go when the last one closes. Its claims used to stay until
  the head changed, so a later claim under the name got the old view.

### Fixed

- A claim view's `host` is each node's hostname. For every node but the
  head's own, it used to be the node's address.

## 0.23.0 (2026-10-25)

### Added

- A claim's view lists the facts its placement rests on, with no policy
  applied. `links[set]` holds every probed, non-forwarded link between
  members, with the port group at each end and the round trip.
  `paths[set]` holds every fewest-hop node path for each pair of members
  with no link.
- The `any` layout takes the first of mesh, ring, star and line that
  fits. Failing all four, it takes a `hops` answer: members that each
  reach every other directly or through `max_hops` other members.
  `max_hops` is 1, and other values are refused for now. The view's
  `layouts[set]` names the answer, and placement and `MENTAT_FABRIC_*`
  follow it.
- A shape may carry `"driver": "ignore"` for a client other than vLLM. The
  answer and its order then do not depend on which node claims, and no
  driver position leaves a group pending. The default, `first`, keeps
  rank 0 on the driver's node, which vLLM needs.
- Test builds cap the test binary's memory at `MENTAT_TEST_MEM_CAP` MiB,
  2048 by default.

### Fixed

- vLLM's rank now matches `MENTAT_FABRIC_RANK`. vLLM's RayExecutorV2 put
  the driver's node first and sorted the other nodes by IP, not by the
  claim's order. For a claimed group the shim sets
  `VLLM_RAY_BUNDLE_INDICES=0,1,...`, so vLLM's rank `i` is bundle `i`. A
  value the user set stays.

## 0.22.0 (2026-10-24)

### Added

- A local process can publish JSON records about its node with
  `PUT /records/<name>` on the HTTP side port, and drop one with `DELETE`
  ([#5](https://github.com/kindlingai/mentat/pull/5)). Each peer row in
  `/status` carries that node's records, and the event stream reports
  changes as `record_changed` and `peer_records`. Only a loopback client
  may write. A node's records total at most 16 KiB, and each name changes
  at most once per status push.

### Changed

- Methods other than `GET` on the existing HTTP paths get 405 instead of
  a closed connection.
- A request with a repeated `Content-Length` or `Sec-WebSocket-Key` is
  refused.

## 0.21.0 (2026-10-21)

### Fixed

- A probe bound to a fabric interface goes only to peer addresses in its
  own subnet ([#3](https://github.com/kindlingai/mentat/issues/3)). On a
  box whose two PCI roots share one switch with a subnet per root, a ring
  no longer pairs one root's address with the other root's interface. The
  two roots count as two ports, so a ring splits each box's neighbours
  across them. Every daemon needs the update, since the head reads its
  peers' probes.
- Releases check each architecture's image on a native runner before
  publishing. The 0.20.0 release run had stopped before publishing.

## 0.20.0 (2026-10-21)

### Added

- The `star` layout: a hub, listed first, with a direct link to every
  other member. The hub goes on the driver's node when that node can be
  one. A switch port counts as a link to every box on the switch.
- A ring or line set takes `ports`. `auto`, the default, reaches each
  member's two neighbours by different ports where some ring or line
  allows it. `split` and `same` require their rule. `MENTAT_CLAIM_PORTS`
  sets it on the shim's default shape.
- Each rank of a ring, line or star claim gets `MENTAT_FABRIC_RANK` and
  `MENTAT_FABRIC_SIZE`. A hub gets `MENTAT_FABRIC_LEAVES`, and a leaf gets
  `MENTAT_FABRIC_HUB` and `MENTAT_FABRIC_HUB_IFACE`.
- The router serves the cluster graph as MCP tools: `mentat_nodes`,
  `mentat_links` and `mentat_group`. Its status page and `/stats.json`
  list the nodes serving each model.
- `daemon_up` logs the daemon's version
  ([#2](https://github.com/kindlingai/mentat/pull/2)).
- Releases build on native amd64 and arm64 runners and publish to
  `ghcr.io/kindlingai/<image>` and `docker.io/mmastrac/<image>`
  ([#1](https://github.com/kindlingai/mentat/pull/1)). A GitHub release
  carries the static binaries, the shim wheel and `SHA256SUMS`.
  `latest` moves only through `latest.yml`.

### Fixed

- A ring or line member's next link and the following member's previous
  link now name the same cable. Before, they could name different cables
  when two tied.
- A ring or line search stops after 20,000 steps. A split ring that no
  set of cabled parts can form is refused at once.
