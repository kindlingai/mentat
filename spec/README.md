# spec

`Election.tla` models head election across daemons: which head each daemon
follows, and how a daemon that boots or crashes changes that. Its
`StableHead` property says no daemon leaves the settled head while that head
is up, and `Converges` says the live daemons come to follow one live head.
`BugNoBootWait` restores the defect where a restarted daemon elected itself
before hearing a peer, and one with a lower id took the head. The model
leaves out partitions and assumes a daemon in the mesh reaches every live
daemon within the hold-down.

`HeadDaemon.tla` is a TLA+ model of the head daemon's placement state for
one group: driver sessions, the session reap and its grace, placement
groups, actors, claims, and a daemon restart that adopts the actors still
running.

Two constants switch parts of the model on. `ModelClaims` lets a driver
claim devices and place inside the claim. `ModelOutage` leaves a restarted
daemon down until it boots, and unaware of each actor until the actor's
agent re-registers, while the driver's monitor keeps polling. Each part
grows the state space, so `check-spec` checks each in a run of its own.

```
scripts/check-spec
```

checks the model with TLC. `scripts/tlc` fetches a pinned `tla2tools.jar`
into `~/.cache/mentat` and needs Java 11 or later. To run TLC by hand:

```
scripts/tlc -config spec/HeadDaemon.cfg spec/HeadDaemon.tla
```

## Properties

| Property | Kind | Statement |
| --- | --- | --- |
| `NoDoubleAlloc` | Invariant | A device has at most one live holder, counting a group and its own actor as one |
| `NoStall` | Liveness | A pending group that its expected devices fit is placed |
| `GraceHoldsActors` | Action | During a driver's reap grace, only the reap of its latest disconnect kills its actors |
| `SessionKeepsActors` | Action | A driver that holds the session keeps its actors |
| `ClaimFence` | Invariant | A group placed inside a claim sits on the claim's devices |
| `ClaimHeld` | Invariant | Every holder of a claim holds the session or is inside its grace |
| `RankZeroOnDriver` | Invariant | A claimed group puts rank 0 on the driver's node when the claim holds it |
| `NoFalseDeath` | Invariant | The driver's monitor never reads a live rank as dead |
| `AdoptedBecomeKnown` | Liveness | A live adopted actor's refs reach the daemon |

## Bug constants

Each `Bug` constant restores a defect. `check-spec` sets each one alone and
requires the property it broke to fail, so the spec is shown to find that
defect. The claim constants did not ship. Each plants the defect its
property guards against. The others shipped.

| Constant | Defect | Fails | Shipped |
| --- | --- | --- | --- |
| `BugAccountPgsOnly` | Free devices subtract placement groups alone | `NoDoubleAlloc` | yes |
| `BugNoPlaceOnExit` | An actor's death places nothing | `NoStall` | yes |
| `BugNoOrphanReap` | A new session leaves a gone driver's actors running | `NoStall` | yes |
| `BugKillDuringGrace` | A new session kills actors inside a reap grace | `GraceHoldsActors` | yes |
| `BugReapReconnected` | A deferred reap kills a driver that reopened its session | `SessionKeepsActors` | yes |
| `BugStaleTimer` | The timer of a superseded disconnect reaps | `GraceHoldsActors` | yes |
| `BugNoRedial` | The shim fails a call when a redial is refused | `NoFalseDeath` | yes |
| `BugUnknownRefReady` | A new head answers a ref it does not know yet as finished | `NoFalseDeath` | yes |
| `BugClaimSpill` | Placement inside a claim uses any free device | `ClaimFence` | no |
| `BugClaimKept` | A reap leaves the driver among the claim's holders | `ClaimHeld` | no |
| `BugNoRotate` | An ordered claim keeps its solved order | `RankZeroOnDriver` | no |

## Scope

The model covers one group on the head. It leaves out the mesh, election,
the event stream and byte-level formats. A device stands for a node, so a
claim is a set of devices. The model keeps only the first member of a
claim's order. The exhaustive test in `claim.rs` covers which ring or line
the solver picks. It checks every graph on up to five nodes against brute
force.

TLC checks every behaviour up to 3 devices, 3 placement groups, 2 drivers,
1 daemon restart and 1 superseded reap timer per driver. With claims or the
outage on it checks up to 2 devices and 2 placement groups. A pass means
the model holds at that size.
