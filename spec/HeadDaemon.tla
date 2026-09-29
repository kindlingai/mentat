----------------------------- MODULE HeadDaemon -----------------------------
(***************************************************************************)
(* The head daemon's placement state for one group: driver sessions, the   *)
(* session reap and its grace, placement groups, actors, claims, and a     *)
(* daemon restart that adopts the actors still running.                    *)
(*                                                                         *)
(* ModelClaims and ModelOutage switch on the claim table and the restart's *)
(* outage. Each one grows the state space, so scripts/check-spec checks    *)
(* them in their own runs.                                                 *)
(*                                                                         *)
(* Each Bug constant restores a defect. scripts/check-spec sets each one   *)
(* alone and requires the property it broke to fail, so the spec is shown  *)
(* to find that defect. spec/README.md says which defects shipped.         *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS
    Clients,        \* driver client ids in the group
    Gpus,           \* the group's devices, one per node
    MaxPgs,         \* placement groups created in one behaviour
    MaxRestarts,    \* daemon restarts in one behaviour
    NoClient,       \* the owner of an unused slot
    NoGpu,          \* rank 0's device before placement
    GraceDeferred,  \* MENTAT_SESSION_REAP_GRACE_MS is above 0
    ModelClaims,    \* drivers claim devices and place inside the claim
    ModelOutage,    \* a restart leaves the daemon down, then boots it
    DriverGpu,      \* the device on the driver's node

    \* Free devices subtract placement groups alone, so a dying or adopted
    \* actor's devices go to a second rank.
    BugAccountPgsOnly,
    \* An actor's death places nothing, so a group waiting on its devices
    \* sits PENDING until the pending timeout.
    BugNoPlaceOnExit,
    \* A new session leaves a gone driver's actors running. An adopted actor
    \* then holds its devices with no reap to free them.
    BugNoOrphanReap,
    \* A new session kills the actors of a driver inside its reap grace.
    BugKillDuringGrace,
    \* A deferred reap kills a driver that reopened its session.
    BugReapReconnected,
    \* The timer of a superseded disconnect reaps, so a driver that
    \* disconnects twice gets the grace of its first disconnect.
    BugStaleTimer,
    \* Placement inside a claim uses any free device.
    BugClaimSpill,
    \* A reap leaves the driver among the claim's holders.
    BugClaimKept,
    \* An ordered claim keeps its solved order, so rank 0 misses the
    \* driver's node.
    BugNoRotate,
    \* The shim fails a call when a redial is refused, and vLLM's monitor
    \* reads the exception as a dead worker.
    BugNoRedial,
    \* A daemon that just became head answers a ref it does not know yet as
    \* finished.
    BugUnknownRefReady

VARIABLES
    session,      \* [Clients -> BOOLEAN]: the client holds the driver session
    reapPending,  \* clients whose reap waits out the grace
    stale,        \* [Clients -> 0..1]: timers of superseded disconnects
    pg,           \* [PgIds -> record]: placement group slots
    actor,        \* [PgIds -> record]: the actor placed in each group
    nextPg,       \* the next unused slot
    restarts,
    claim,        \* the group's one claim
    down,         \* the daemon is restarting
    known,        \* [PgIds -> BOOLEAN]: the daemon knows the actor's refs
    falseDeath    \* the driver's monitor read a live rank as dead

vars == <<session, reapPending, stale, pg, actor, nextPg, restarts,
          claim, down, known, falseDeath>>

PgIds == 1..MaxPgs

\* An actor process that holds its devices. A killed one holds them until
\* the agent reports its exit.
Live == {"running", "killing"}

NoClaim == [held |-> FALSE, holders |-> {}, gpus |-> {}]

EmptyPg == [st |-> "none", owner |-> NoClient, need |-> 0, gpus |-> {},
            inClaim |-> FALSE, fence |-> {}, first |-> NoGpu]

TypeOK ==
    /\ session \in [Clients -> BOOLEAN]
    /\ reapPending \subseteq Clients
    /\ stale \in [Clients -> 0..1]
    /\ pg \in [PgIds -> [st : {"none", "pending", "created", "removed"},
                         owner : Clients \cup {NoClient},
                         need : 0..Cardinality(Gpus),
                         gpus : SUBSET Gpus,
                         inClaim : BOOLEAN,
                         fence : SUBSET Gpus,
                         first : Gpus \cup {NoGpu}]]
    /\ actor \in [PgIds -> [st : {"none", "running", "killing", "dead"},
                            owner : Clients \cup {NoClient},
                            gpus : SUBSET Gpus]]
    /\ nextPg \in 1..(MaxPgs + 1)
    /\ restarts \in 0..MaxRestarts
    /\ claim \in [held : BOOLEAN, holders : SUBSET Clients, gpus : SUBSET Gpus]
    /\ down \in BOOLEAN
    /\ known \in [PgIds -> BOOLEAN]
    /\ falseDeath \in BOOLEAN

Init ==
    /\ session = [c \in Clients |-> FALSE]
    /\ reapPending = {}
    /\ stale = [c \in Clients |-> 0]
    /\ pg = [i \in PgIds |-> EmptyPg]
    /\ actor = [i \in PgIds |-> [st |-> "none", owner |-> NoClient, gpus |-> {}]]
    /\ nextPg = 1
    /\ restarts = 0
    /\ claim = NoClaim
    /\ down = FALSE
    /\ known = [i \in PgIds |-> TRUE]
    /\ falseDeath = FALSE

(***************************************************************************)
(* State::free_gpus_of and try_place.                                      *)
(***************************************************************************)

Held(p, a) ==
    UNION {p[i].gpus : i \in {j \in PgIds : p[j].st = "created"}}
    \cup IF BugAccountPgsOnly
         THEN {}
         ELSE UNION {a[i].gpus : i \in {j \in PgIds : a[j].st \in Live}}

Free(p, a) == Gpus \ Held(p, a)

\* Rank 0's device. The claim's order turns so rank 0 sits on the driver's
\* node when the claim holds it.
First(s) == IF DriverGpu \in s /\ ~BugNoRotate THEN DriverGpu
            ELSE CHOOSE g \in s : TRUE

\* Each pending group, in slot order, gets devices if they are free. A
\* group inside a claim takes exactly the claim's devices, one bundle each,
\* and waits while the claim is gone or a device is held.
RECURSIVE PlaceFrom(_, _, _, _)
PlaceFrom(i, p, a, cl) ==
    IF i > MaxPgs THEN p
    ELSE IF p[i].st # "pending" THEN PlaceFrom(i + 1, p, a, cl)
    ELSE IF p[i].inClaim /\ ~BugClaimSpill
         THEN IF /\ cl.held
                 /\ Cardinality(cl.gpus) = p[i].need
                 /\ cl.gpus \subseteq Free(p, a)
              THEN PlaceFrom(i + 1,
                             [p EXCEPT ![i].st = "created", ![i].gpus = cl.gpus,
                                       ![i].fence = cl.gpus,
                                       ![i].first = First(cl.gpus)],
                             a, cl)
              ELSE PlaceFrom(i + 1, p, a, cl)
    ELSE IF Cardinality(Free(p, a)) >= p[i].need
         THEN LET s == CHOOSE s \in SUBSET Free(p, a) : Cardinality(s) = p[i].need
              IN PlaceFrom(i + 1,
                           [p EXCEPT ![i].st = "created", ![i].gpus = s,
                                     ![i].fence = IF p[i].inClaim THEN cl.gpus ELSE Gpus,
                                     ![i].first = IF p[i].inClaim THEN First(s) ELSE NoGpu],
                           a, cl)
         ELSE PlaceFrom(i + 1, p, a, cl)

TryPlace(p, a, cl) == PlaceFrom(1, p, a, cl)

\* The claim once `c` stops holding it. The last holder ends it.
Release(cl, c) ==
    IF c \notin cl.holders THEN cl
    ELSE IF cl.holders = {c} THEN NoClaim
    ELSE [cl EXCEPT !.holders = @ \ {c}]

(***************************************************************************)
(* Actions.                                                                *)
(***************************************************************************)

\* reap_client_resources: the owner's groups go, its actors are killed, its
\* claim is released, and placement runs.
Reap(c) ==
    LET p == [i \in PgIds |->
                IF pg[i].owner = c /\ pg[i].st \in {"pending", "created"}
                THEN [pg[i] EXCEPT !.st = "removed"] ELSE pg[i]]
        a == [i \in PgIds |->
                IF actor[i].owner = c /\ actor[i].st = "running"
                THEN [actor[i] EXCEPT !.st = "killing"] ELSE actor[i]]
        cl == IF BugClaimKept THEN claim ELSE Release(claim, c)
    IN /\ actor' = a
       /\ claim' = cl
       /\ pg' = TryPlace(p, a, cl)

\* A driver opens the session. The group allows one, so every other session
\* is closed. The daemon kills each running actor whose owner lacks both a
\* session and a pending reap.
Connect(c) ==
    /\ ~down
    /\ \A d \in Clients : ~session[d]
    /\ session' = [session EXCEPT ![c] = TRUE]
    /\ actor' = [i \in PgIds |->
                   IF /\ ~BugNoOrphanReap
                      /\ actor[i].st = "running"
                      /\ actor[i].owner # c
                      /\ BugKillDuringGrace \/ actor[i].owner \notin reapPending
                   THEN [actor[i] EXCEPT !.st = "killing"]
                   ELSE actor[i]]
    /\ UNCHANGED <<reapPending, stale, pg, nextPg, restarts, claim, down,
                   known, falseDeath>>

\* The session closes. With a grace the reap waits. Without one it runs now.
\* A disconnect inside a pending grace supersedes that grace's timer. The
\* model holds one superseded timer per client.
Disconnect(c) ==
    /\ session[c]
    /\ session' = [session EXCEPT ![c] = FALSE]
    /\ IF GraceDeferred
       THEN /\ c \in reapPending => stale[c] = 0
            /\ stale' = IF c \in reapPending
                        THEN [stale EXCEPT ![c] = 1] ELSE stale
            /\ reapPending' = reapPending \cup {c}
            /\ UNCHANGED <<pg, actor, claim>>
       ELSE /\ Reap(c)
            /\ UNCHANGED <<reapPending, stale>>
    /\ UNCHANGED <<nextPg, restarts, down, known, falseDeath>>

\* The grace ends. A driver that reopened its session keeps its groups,
\* actors and claim.
ReapFires(c) ==
    /\ c \in reapPending
    /\ reapPending' = reapPending \ {c}
    /\ IF session[c] /\ ~BugReapReconnected
       THEN UNCHANGED <<pg, actor, claim>>
       ELSE Reap(c)
    /\ UNCHANGED <<session, stale, nextPg, restarts, down, known, falseDeath>>

\* A superseded timer ends. Its token no longer matches, so it does nothing.
StaleFires(c) ==
    /\ stale[c] > 0
    /\ stale' = [stale EXCEPT ![c] = stale[c] - 1]
    /\ IF BugStaleTimer
       THEN /\ reapPending' = reapPending \ {c}
            /\ IF session[c] THEN UNCHANGED <<pg, actor, claim>> ELSE Reap(c)
       ELSE UNCHANGED <<reapPending, pg, actor, claim>>
    /\ UNCHANGED <<session, nextPg, restarts, down, known, falseDeath>>

\* claim(): the first holder solves it over some devices, and a later one
\* joins the same view.
Claim(c, s) ==
    /\ ModelClaims
    /\ ~down
    /\ session[c]
    /\ c \notin claim.holders
    /\ IF claim.held
       THEN /\ s = claim.gpus
            /\ claim' = [claim EXCEPT !.holders = @ \cup {c}]
       ELSE /\ s # {}
            /\ claim' = [held |-> TRUE, holders |-> {c}, gpus |-> s]
    /\ UNCHANGED <<session, reapPending, stale, pg, actor, nextPg, restarts,
                   down, known, falseDeath>>

\* pg_create, inside the driver's claim when `inClaim`.
PgCreate(c, n, inClaim) ==
    /\ session[c]
    /\ nextPg <= MaxPgs
    /\ inClaim => ModelClaims /\ c \in claim.holders
    /\ pg' = TryPlace([pg EXCEPT ![nextPg] = [EmptyPg EXCEPT !.st = "pending",
                                                 !.owner = c, !.need = n,
                                                 !.inClaim = inClaim]],
                      actor, claim)
    /\ nextPg' = nextPg + 1
    /\ UNCHANGED <<session, reapPending, stale, actor, restarts, claim, down,
                   known, falseDeath>>

\* An actor holds the devices of the group it is placed in.
ActorCreate(i) ==
    /\ pg[i].st = "created"
    /\ actor[i].st = "none"
    /\ session[pg[i].owner]
    /\ actor' = [actor EXCEPT ![i] = [st |-> "running", owner |-> pg[i].owner,
                                      gpus |-> pg[i].gpus]]
    /\ known' = [known EXCEPT ![i] = TRUE]
    /\ UNCHANGED <<session, reapPending, stale, pg, nextPg, restarts, claim,
                   down, falseDeath>>

\* mark_actor_dead: the agent reports the process gone, which frees its
\* devices. A down daemon hears it after it boots.
Dies(i) ==
    /\ ~down
    /\ actor' = [actor EXCEPT ![i].st = "dead"]
    /\ pg' = IF BugNoPlaceOnExit THEN pg ELSE TryPlace(pg, actor', claim)
    /\ UNCHANGED <<session, reapPending, stale, nextPg, restarts, claim, down,
                   known, falseDeath>>

\* A killed process exits. Fairness guarantees this step.
KilledExits(i) == actor[i].st = "killing" /\ Dies(i)

\* A running process crashes.
Crashes(i) == actor[i].st = "running" /\ Dies(i)

\* The daemon restarts and loses its sessions, reaps, groups and claim. With
\* ModelOutage it stays down until Boot, and knows no actor until the
\* actor's agent re-registers. Without it the agents re-register at once and
\* the daemon adopts their processes.
Restart ==
    /\ ~down
    /\ restarts < MaxRestarts
    /\ restarts' = restarts + 1
    /\ session' = [c \in Clients |-> FALSE]
    /\ reapPending' = {}
    /\ stale' = [c \in Clients |-> 0]
    /\ pg' = [i \in PgIds |->
                IF pg[i].st \in {"pending", "created"}
                THEN [pg[i] EXCEPT !.st = "removed"] ELSE pg[i]]
    /\ claim' = NoClaim
    /\ down' = ModelOutage
    /\ known' = IF ModelOutage THEN [i \in PgIds |-> FALSE] ELSE known
    /\ UNCHANGED <<actor, nextPg, falseDeath>>

Boot ==
    /\ down
    /\ down' = FALSE
    /\ UNCHANGED <<session, reapPending, stale, pg, actor, nextPg, restarts,
                   claim, known, falseDeath>>

\* An agent re-registers and reports the actor and its refs.
Reregister(i) ==
    /\ ~down
    /\ ~known[i]
    /\ known' = [known EXCEPT ![i] = TRUE]
    /\ UNCHANGED <<session, reapPending, stale, pg, actor, nextPg, restarts,
                   claim, down, falseDeath>>

\* The driver's monitor polls a rank's run() ref and reads it as finished.
\* Without a bug, a live rank's ref never reads so.
Monitor(i) ==
    /\ ModelOutage
    /\ actor[i].st \in Live
    /\ ~falseDeath
    /\ \/ down /\ BugNoRedial
       \/ ~down /\ ~known[i] /\ BugUnknownRefReady
    /\ falseDeath' = TRUE
    /\ UNCHANGED <<session, reapPending, stale, pg, actor, nextPg, restarts,
                   claim, down, known>>

Next ==
    \/ \E c \in Clients : Connect(c) \/ Disconnect(c) \/ ReapFires(c) \/ StaleFires(c)
    \/ \E c \in Clients, s \in SUBSET Gpus : Claim(c, s)
    \/ \E c \in Clients, n \in 1..Cardinality(Gpus), b \in BOOLEAN : PgCreate(c, n, b)
    \/ \E i \in PgIds : ActorCreate(i) \/ KilledExits(i) \/ Crashes(i)
                        \/ Reregister(i) \/ Monitor(i)
    \/ Restart \/ Boot

\* A killed process exits, a grace ends, a restarting daemon boots, and an
\* agent re-registers.
Fairness ==
    /\ \A i \in PgIds : WF_vars(KilledExits(i)) /\ WF_vars(Reregister(i))
    /\ \A c \in Clients : WF_vars(ReapFires(c)) /\ WF_vars(StaleFires(c))
    /\ WF_vars(Boot)

Spec == Init /\ [][Next]_vars /\ Fairness

(***************************************************************************)
(* Properties.                                                             *)
(***************************************************************************)

\* A group and the actor placed in it share devices. Every other pair of
\* live holders is disjoint.
NoDoubleAlloc ==
    /\ \A i, j \in PgIds :
           (i # j /\ pg[i].st = "created" /\ pg[j].st = "created")
               => pg[i].gpus \cap pg[j].gpus = {}
    /\ \A i, j \in PgIds :
           (i # j /\ actor[i].st \in Live /\ pg[j].st = "created")
               => actor[i].gpus \cap pg[j].gpus = {}
    /\ \A i, j \in PgIds :
           (i # j /\ actor[i].st \in Live /\ actor[j].st \in Live)
               => actor[i].gpus \cap actor[j].gpus = {}

\* A live actor whose devices come free without anyone's help: its process
\* was killed, or it belongs to a gone driver while another driver holds the
\* session.
Reclaimable(i) ==
    \/ actor[i].st = "killing"
    \/ /\ actor[i].st = "running"
       /\ ~session[actor[i].owner]
       /\ actor[i].owner \notin reapPending
       /\ \E c \in Clients : session[c]

\* Devices a pending group can count on: those held by nothing, or only by a
\* reclaimable actor.
Expected ==
    Gpus \ (UNION {pg[i].gpus : i \in {j \in PgIds : pg[j].st = "created"}}
            \cup UNION {actor[i].gpus :
                          i \in {j \in PgIds : actor[j].st \in Live
                                               /\ ~Reclaimable(j)}})

\* A group outside a claim fits when enough devices are expected. One inside
\* a claim fits when the claim is held, has a device per bundle, and each of
\* them is expected.
Fits(i) ==
    IF pg[i].inClaim
    THEN /\ claim.held
         /\ Cardinality(claim.gpus) = pg[i].need
         /\ claim.gpus \subseteq Expected
    ELSE Cardinality(Expected) >= pg[i].need

\* A pending group that fits is placed, or stops fitting because another
\* group got the devices first.
NoStall ==
    \A i \in PgIds :
        (pg[i].st = "pending" /\ Fits(i)) ~> (pg[i].st # "pending" \/ ~Fits(i))

\* The grace keeps a gone driver's actors up. Only the reap of its latest
\* disconnect kills them.
GraceHoldsActors ==
    [][\A i \in PgIds :
          (/\ actor[i].st = "running"
           /\ actor'[i].st = "killing"
           /\ actor[i].owner \in reapPending)
              => /\ actor[i].owner \notin reapPending'
                 /\ stale'[actor[i].owner] = stale[actor[i].owner]]_vars

\* A driver that holds the session keeps its actors.
SessionKeepsActors ==
    [][\A i \in PgIds :
          (actor[i].st = "running" /\ actor'[i].st = "killing")
              => ~session'[actor[i].owner]]_vars

\* A group placed inside a claim sits on the claim's devices.
ClaimFence ==
    \A i \in PgIds :
        (pg[i].st = "created" /\ pg[i].inClaim) => pg[i].gpus \subseteq pg[i].fence

\* Every holder of a claim is a driver with the session or inside its grace.
ClaimHeld ==
    /\ claim.held <=> claim.holders # {}
    /\ \A c \in claim.holders : session[c] \/ c \in reapPending

\* A claimed group puts rank 0 on the driver's node when the claim holds it,
\* where vLLM expects it.
RankZeroOnDriver ==
    \A i \in PgIds :
        (pg[i].st = "created" /\ pg[i].inClaim /\ DriverGpu \in pg[i].gpus)
            => pg[i].first = DriverGpu

\* The driver never reads a live rank as dead.
NoFalseDeath == ~falseDeath

\* An adopted actor's refs reach the daemon while the actor lives.
AdoptedBecomeKnown ==
    \A i \in PgIds :
        (actor[i].st \in Live /\ ~known[i]) ~> (known[i] \/ actor[i].st \notin Live)

=============================================================================
