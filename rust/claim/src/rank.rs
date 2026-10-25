//! The rank order of a solved ring, line or star.

use serde::{Deserialize, Serialize};

use crate::shape::{Driver, Layout, Shape};
use crate::view::{Member, View};
use crate::Error;

/// A set's members in rank order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ranks {
    /// What the set was placed as: a ring, a line or a star.
    pub layout: Layout,
    pub members: Vec<Member>,
}

/// The members of `set` in rank order, with rank 0 on `driver`'s node as
/// vLLM requires. A mesh or a hops answer has no order and returns None.
///
/// A ring turns to start at the driver's node. A line reverses when the
/// driver's node is its last member, and each member's `prev` and `next`
/// swap. A star keeps its hub first. The solved order stays when the shape
/// has `driver: ignore` or the driver's node is outside the set.
///
/// A line with the driver's node inside it, or a star with the driver's
/// node on a leaf, returns `Error::Driver`. No order of either puts rank 0
/// there.
pub fn ranks(
    shape: &Shape,
    view: &View,
    set: &str,
    driver: Option<&str>,
) -> Result<Option<Ranks>, Error> {
    let s = shape
        .sets
        .iter()
        .find(|s| s.name == set)
        .ok_or_else(|| Error::BadShape(format!("the shape has no set {set:?}")))?;
    let layout = match s.layout {
        Layout::Any => *view
            .layouts
            .get(set)
            .ok_or_else(|| Error::BadShape(format!("the view has no layout for set {set:?}")))?,
        l => l,
    };
    if !layout.ordered() {
        return Ok(None);
    }
    let mut members = view
        .sets
        .get(set)
        .ok_or_else(|| Error::BadShape(format!("the view has no set {set:?}")))?
        .clone();
    let driver = driver.filter(|_| shape.driver == Driver::First);
    let at = driver.and_then(|d| members.iter().position(|m| m.node == d));
    let last = members.len().saturating_sub(1);
    match (layout, at) {
        (Layout::Ring, Some(i)) => members.rotate_left(i),
        (Layout::Line, Some(i)) if i == last && i > 0 => {
            members.reverse();
            for m in &mut members {
                std::mem::swap(&mut m.prev, &mut m.next);
            }
        }
        (Layout::Line, Some(i)) if i > 0 => {
            return Err(Error::Driver(format!(
                "set {set:?} is a line and the driver's node is inside it. Start \
                 the driver on a node at one end of the line"
            )))
        }
        (Layout::Star, Some(i)) if i > 0 => {
            let hub = &members[0];
            let hub = if hub.host.is_empty() {
                &hub.node
            } else {
                &hub.host
            };
            return Err(Error::Driver(format!(
                "set {set:?} is a star with its hub on {hub}, and the driver's node \
                 is a leaf. Start the driver on {hub}"
            )));
        }
        _ => {}
    }
    Ok(Some(Ranks { layout, members }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::Neighbor;
    use serde_json::json;

    /// A view of one set `s` whose members link in order, each `prev` and
    /// `next` naming its neighbour.
    fn chain(nodes: &[&str], ring: bool) -> View {
        let k = nodes.len();
        let link = |to: usize| {
            Some(Neighbor {
                node: nodes[to].into(),
                ..Default::default()
            })
        };
        let members = (0..k)
            .map(|i| Member {
                node: nodes[i].into(),
                prev: if ring || i > 0 {
                    link((i + k - 1) % k)
                } else {
                    None
                },
                next: if ring || i + 1 < k {
                    link((i + 1) % k)
                } else {
                    None
                },
                ..Default::default()
            })
            .collect();
        View {
            sets: [("s".to_string(), members)].into(),
            ..Default::default()
        }
    }

    fn shape(layout: &str, driver: &str) -> Shape {
        Shape::from_json(
            &json!({"sets": [{"name": "s", "bundles": 4, "layout": layout}],
                                 "driver": driver}),
        )
        .unwrap()
    }

    fn order(r: &Ranks) -> Vec<&str> {
        r.members.iter().map(|m| m.node.as_str()).collect()
    }

    #[test]
    fn a_ring_turns_to_the_driver() {
        let v = chain(&["a", "b", "c", "d"], true);
        let r = ranks(&shape("ring", "first"), &v, "s", Some("c"))
            .unwrap()
            .unwrap();
        assert_eq!(order(&r), ["c", "d", "a", "b"]);
        assert_eq!(r.members[0].next.as_ref().unwrap().node, "d");
        let r = ranks(&shape("ring", "ignore"), &v, "s", Some("c"))
            .unwrap()
            .unwrap();
        assert_eq!(order(&r), ["a", "b", "c", "d"]);
    }

    #[test]
    fn a_line_reverses_for_a_driver_at_its_far_end() {
        let v = chain(&["a", "b", "c"], false);
        let r = ranks(&shape("line", "first"), &v, "s", Some("c"))
            .unwrap()
            .unwrap();
        assert_eq!(order(&r), ["c", "b", "a"]);
        assert!(r.members[0].prev.is_none());
        assert_eq!(r.members[0].next.as_ref().unwrap().node, "b");
        assert!(r.members[2].next.is_none());

        let e = ranks(&shape("line", "first"), &v, "s", Some("b")).unwrap_err();
        assert_eq!(e.code(), "driver_position");
        assert!(e.message().contains("inside"), "{e}");
        assert!(ranks(&shape("line", "ignore"), &v, "s", Some("b")).is_ok());
    }

    #[test]
    fn a_star_refuses_a_driver_on_a_leaf() {
        let mut v = chain(&["hub", "x", "y"], false);
        v.sets.get_mut("s").unwrap()[0].host = "box-h".into();
        let r = ranks(&shape("star", "first"), &v, "s", Some("hub"))
            .unwrap()
            .unwrap();
        assert_eq!(order(&r), ["hub", "x", "y"]);
        let e = ranks(&shape("star", "first"), &v, "s", Some("y")).unwrap_err();
        assert!(e.message().contains("hub on box-h"), "{e}");
    }

    /// An `any` set orders as what it was placed as.
    #[test]
    fn an_any_set_orders_as_its_answer() {
        let mut v = chain(&["a", "b", "c", "d"], true);
        v.layouts.insert("s".into(), Layout::Ring);
        let r = ranks(&shape("any", "first"), &v, "s", Some("b"))
            .unwrap()
            .unwrap();
        assert_eq!(
            (r.layout, order(&r)),
            (Layout::Ring, vec!["b", "c", "d", "a"])
        );
        for pooled in [Layout::Mesh, Layout::Hops] {
            v.layouts.insert("s".into(), pooled);
            assert_eq!(ranks(&shape("any", "first"), &v, "s", Some("b")), Ok(None));
        }
        assert_eq!(ranks(&shape("mesh", "first"), &v, "s", None), Ok(None));
        assert!(ranks(&shape("ring", "first"), &v, "t", None).is_err());
    }
}
