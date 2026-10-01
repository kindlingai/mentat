------------------------------ MODULE Election ------------------------------
(***************************************************************************)
(* Head election across daemons: mesh::head_candidate and mesh::elector.  *)
(* Each daemon follows the lowest head its live peers report, else the     *)
(* lowest live id it knows, which before it hears anyone is its own.       *)
(*                                                                         *)
(* A booted daemon that expects peers waits to hear one before it elects.  *)
(* The wait is longer than a live daemon takes to answer, so the model     *)
(* lets it end only while no other daemon is up. Partitions are left out:  *)
(* a daemon in the mesh reaches every live daemon within the hold-down.    *)
(*                                                                         *)
(* BugNoBootWait restores the defect: a daemon elected itself before       *)
(* hearing its peers, and one with a lower id then took the head from the  *)
(* live one, which moved every group.                                      *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS
    N,              \* daemons 1..N. A lower number is a lower node id.
    MaxRestarts,    \* daemon crashes in one behaviour
    BugNoBootWait

Daemons == 1..N
None == 0

VARIABLES
    up,           \* [Daemons -> BOOLEAN]
    head,         \* [Daemons -> Daemons \cup {None}]: whom each follows
    link,         \* [Daemons -> SUBSET Daemons]: peers each holds a link to
    heard,        \* [Daemons -> BOOLEAN]: a peer's status arrived since boot
    waited,       \* [Daemons -> BOOLEAN]: the boot wait ran out
    restarts,
    settled       \* the head every live daemon agreed on, kept while it is up

vars == <<up, head, link, heard, waited, restarts, settled>>

TypeOK ==
    /\ up \in [Daemons -> BOOLEAN]
    /\ head \in [Daemons -> Daemons \cup {None}]
    /\ link \in [Daemons -> SUBSET Daemons]
    /\ heard \in [Daemons -> BOOLEAN]
    /\ waited \in [Daemons -> BOOLEAN]
    /\ restarts \in 0..MaxRestarts
    /\ settled \in Daemons \cup {None}

Min(S) == CHOOSE x \in S : \A y \in S : x <= y

\* A daemon counts itself and every peer it holds a link to as alive.
Alive(d, x) == x = d \/ x \in link[d]

\* mesh::head_candidate.
Candidate(d) ==
    LET live == {h \in {head[e] : e \in link[d]} \cup {head[d]} :
                    h # None /\ Alive(d, h)}
    IN IF live # {} THEN Min(live) ELSE Min(link[d] \cup {d})

\* h is up, and every live daemon follows it.
Agreed(u, hd, h) ==
    /\ h # None
    /\ u[h]
    /\ \A d \in Daemons : u[d] => hd[d] = h

\* The settled head after a step: one every live daemon follows, else the
\* old one while it stays up. A daemon that boots later does not unsettle
\* it.
NextSettled ==
    IF \E h \in Daemons : Agreed(up', head', h)
    THEN CHOOSE h \in Daemons : Agreed(up', head', h)
    ELSE IF settled # None /\ up'[settled] THEN settled ELSE None

Init ==
    /\ up = [d \in Daemons |-> TRUE]
    /\ head = [d \in Daemons |-> None]
    /\ link = [d \in Daemons |-> {}]
    /\ heard = [d \in Daemons |-> FALSE]
    /\ waited = [d \in Daemons |-> FALSE]
    /\ restarts = 0
    /\ settled = None

\* A link comes up and each side hears the other's status.
Link(d, e) ==
    /\ d # e /\ up[d] /\ up[e]
    /\ e \notin link[d] \/ d \notin link[e]
    /\ link' = [link EXCEPT ![d] = @ \cup {e}, ![e] = @ \cup {d}]
    /\ heard' = [heard EXCEPT ![d] = TRUE, ![e] = TRUE]
    /\ UNCHANGED <<up, head, waited, restarts>>
    /\ settled' = NextSettled

\* A daemon notices a peer is gone.
Drop(d, e) ==
    /\ e \in link[d] /\ ~up[e]
    /\ link' = [link EXCEPT ![d] = @ \ {e}]
    /\ UNCHANGED <<up, head, heard, waited, restarts>>
    /\ settled' = NextSettled

Crash(d) ==
    /\ up[d] /\ restarts < MaxRestarts
    /\ restarts' = restarts + 1
    /\ up' = [up EXCEPT ![d] = FALSE]
    /\ head' = [head EXCEPT ![d] = None]
    /\ link' = [link EXCEPT ![d] = {}]
    /\ heard' = [heard EXCEPT ![d] = FALSE]
    /\ waited' = [waited EXCEPT ![d] = FALSE]
    /\ settled' = NextSettled

Boot(d) ==
    /\ ~up[d]
    /\ up' = [up EXCEPT ![d] = TRUE]
    /\ UNCHANGED <<head, link, heard, waited, restarts>>
    /\ settled' = NextSettled

\* The boot wait runs out. A live daemon answers sooner, so this happens only
\* while no other daemon is up.
BootWaitEnds(d) ==
    /\ up[d] /\ ~heard[d] /\ ~waited[d]
    /\ \A e \in Daemons \ {d} : ~up[e]
    /\ waited' = [waited EXCEPT ![d] = TRUE]
    /\ UNCHANGED <<up, head, link, heard, restarts>>
    /\ settled' = NextSettled

\* mesh::elector commits the candidate once it holds still for the
\* hold-down. A daemon in the mesh reaches every live daemon within the
\* hold-down. A daemon that has just booted may take longer to hear anyone,
\* so it waits out the boot wait first.
Elect(d) ==
    /\ up[d]
    /\ Candidate(d) # head[d]
    /\ IF heard[d]
       THEN \A e \in Daemons \ {d} : up[e] => e \in link[d]
       ELSE waited[d] \/ BugNoBootWait
    /\ head' = [head EXCEPT ![d] = Candidate(d)]
    /\ UNCHANGED <<up, link, heard, waited, restarts>>
    /\ settled' = NextSettled

Next ==
    \/ \E d, e \in Daemons : Link(d, e) \/ Drop(d, e)
    \/ \E d \in Daemons : Crash(d) \/ Boot(d) \/ BootWaitEnds(d) \/ Elect(d)

Fairness ==
    /\ \A d, e \in Daemons : WF_vars(Link(d, e)) /\ WF_vars(Drop(d, e))
    /\ \A d \in Daemons : WF_vars(Boot(d)) /\ WF_vars(BootWaitEnds(d))
                          /\ WF_vars(Elect(d))

Spec == Init /\ [][Next]_vars /\ Fairness

(***************************************************************************)
(* Properties.                                                             *)
(***************************************************************************)

\* No daemon leaves the settled head while that head is up. A move would
\* take every group with it.
StableHead ==
    [][\A e \in Daemons :
          (/\ settled # None /\ up'[settled]
           /\ up[e] /\ up'[e] /\ head[e] = settled)
              => head'[e] = settled]_vars

\* The live daemons come to follow one live head, including a daemon left
\* alone with every peer down.
Converges ==
    <>[](\E h \in Daemons : up[h] /\ \A d \in Daemons : up[d] => head[d] = h)

=============================================================================
