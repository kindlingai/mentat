//! An agent registers diagnostic metadata: string keys and values. The
//! daemon stores it on the agent row, and placement and routing ignore it.
//!
//! Sources, in order: the box itself, then `MENTAT_META_FILE`, then
//! `MENTAT_META_<NAME>` variables. A later source overrides an earlier one.

use std::collections::BTreeMap;
use std::path::Path;

use mentat_common::logfmt::log;
use serde_json::Value;

pub type Meta = BTreeMap<String, String>;

const MAX_KEYS: usize = 64;
const MAX_KEY_LEN: usize = 64;
const MAX_VALUE_LEN: usize = 1024;

/// Kernel modules whose version is recorded as `driver.<module>`.
const DRIVER_MODULES: [&str; 3] = ["nvidia", "amdgpu", "mlx5_core"];

/// Reads every source and applies `bounded`.
pub fn collect() -> Meta {
    let mut meta = probed(Path::new("/"));
    if let Ok(path) = std::env::var("MENTAT_META_FILE") {
        if !path.is_empty() {
            match from_file(Path::new(&path)) {
                Ok(m) => meta.extend(m),
                Err(e) => log(
                    "agent_meta_file_bad",
                    &[("path", path.clone()), ("error", e)],
                ),
            }
        }
    }
    meta.extend(from_env(std::env::vars()));
    let (meta, dropped) = bounded(meta);
    if !dropped.is_empty() {
        log("agent_meta_dropped", &[("keys", dropped.join(","))]);
    }
    meta
}

/// A key is lowercase ASCII letters, digits, `.`, `_` and `-`.
fn valid_key(k: &str) -> bool {
    !k.is_empty()
        && k.len() <= MAX_KEY_LEN
        && k.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"._-".contains(&b))
}

/// Splits `meta` into the entries within the limits and the keys of the
/// rest. The daemon applies this too, since an agent of another build sends
/// its own.
pub fn bounded(meta: Meta) -> (Meta, Vec<String>) {
    let mut kept = Meta::new();
    let mut dropped = Vec::new();
    for (k, v) in meta {
        if valid_key(&k) && v.len() <= MAX_VALUE_LEN && kept.len() < MAX_KEYS {
            kept.insert(k, v);
        } else {
            dropped.push(k);
        }
    }
    (kept, dropped)
}

/// `MENTAT_META_<NAME>=value` gives the key `<name>` in lowercase.
fn from_env(vars: impl Iterator<Item = (String, String)>) -> Meta {
    vars.filter_map(|(k, v)| {
        let name = k.strip_prefix("MENTAT_META_")?;
        (name != "FILE").then(|| (name.to_ascii_lowercase(), v))
    })
    .collect()
}

/// A JSON object. A string value is used as it is. Any other value is used
/// as its JSON text.
fn from_file(path: &Path) -> Result<Meta, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let v: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let obj = v.as_object().ok_or("not a JSON object")?;
    Ok(obj
        .iter()
        .map(|(k, v)| {
            let s = match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            (k.clone(), s)
        })
        .collect())
}

/// Reads the kernel release, the version of each loaded driver module, and
/// each RDMA port's state and RoCE v2 GIDs from the filesystem at `root`.
fn probed(root: &Path) -> Meta {
    let mut out = Meta::new();
    if let Some(v) = read_trim(&root.join("proc/sys/kernel/osrelease")) {
        out.insert("kernel".into(), v);
    }
    for m in DRIVER_MODULES {
        if let Some(v) = read_trim(&root.join("sys/module").join(m).join("version")) {
            out.insert(format!("driver.{m}"), v);
        }
    }
    out.extend(rdma_ports(&root.join("sys/class/infiniband")));
    out
}

/// Per port of each RDMA device: `rdma.<dev>.<port>.state`, and
/// `rdma.<dev>.<port>.gid` listing each RoCE v2 GID that maps an IPv4
/// address as `<index> <address> <netdev>`. `NCCL_IB_GID_INDEX` takes that
/// index.
fn rdma_ports(class: &Path) -> Meta {
    let mut out = Meta::new();
    for dev in sorted_names(class) {
        let ports = class.join(&dev).join("ports");
        for port in sorted_names(&ports) {
            let p = ports.join(&port);
            let key = format!("rdma.{dev}.{port}");
            // "4: ACTIVE" reads as "ACTIVE".
            if let Some(s) = read_trim(&p.join("state")) {
                let s = s.split_once(": ").map(|(_, s)| s).unwrap_or(&s);
                out.insert(format!("{key}.state"), s.to_string());
            }
            let mut gids: Vec<(u32, String)> = Vec::new();
            for idx in sorted_names(&p.join("gids")) {
                let Ok(i) = idx.parse::<u32>() else { continue };
                // An unused slot fails to read its type.
                let Some(ty) = read_trim(&p.join("gid_attrs/types").join(&idx)) else {
                    continue;
                };
                if ty != "RoCE v2" {
                    continue;
                }
                let Some(v4) = read_trim(&p.join("gids").join(&idx)).and_then(|g| ipv4_of_gid(&g))
                else {
                    continue;
                };
                let ndev = read_trim(&p.join("gid_attrs/ndevs").join(&idx)).unwrap_or_default();
                gids.push((i, format!("{i} {v4} {ndev}").trim_end().to_string()));
            }
            gids.sort();
            if !gids.is_empty() {
                let list: Vec<String> = gids.into_iter().map(|(_, s)| s).collect();
                out.insert(format!("{key}.gid"), list.join(", "));
            }
        }
    }
    out
}

/// The IPv4 address in an IPv4-mapped GID, `::ffff:a.b.c.d`, written as
/// sysfs does: eight groups of four hex digits.
fn ipv4_of_gid(gid: &str) -> Option<String> {
    let groups: Vec<&str> = gid.split(':').collect();
    if groups.len() != 8 || groups[..5].iter().any(|g| *g != "0000") || groups[5] != "ffff" {
        return None;
    }
    let hi = u16::from_str_radix(groups[6], 16).ok()?;
    let lo = u16::from_str_radix(groups[7], 16).ok()?;
    Some(format!(
        "{}.{}.{}.{}",
        hi >> 8,
        hi & 0xff,
        lo >> 8,
        lo & 0xff
    ))
}

fn read_trim(p: &Path) -> Option<String> {
    let s = std::fs::read_to_string(p).ok()?;
    let s = s.trim();
    (!s.is_empty()).then(|| s.to_string())
}

fn sorted_names(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    v.sort();
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(root: &Path, rel: &str, body: &str) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, body).unwrap();
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("mentat-meta-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn the_box_reports_its_kernel_drivers_and_roce_gids() {
        let root = tmp("probe");
        write(&root, "proc/sys/kernel/osrelease", "6.11.0-1016-nvidia\n");
        write(&root, "sys/module/nvidia/version", "580.95.05\n");
        let port = "sys/class/infiniband/mlx5_0/ports/1";
        write(&root, &format!("{port}/state"), "4: ACTIVE\n");
        // 0 is RoCE v1 and 1 is an IPv6 link-local GID, so only 3 counts.
        for (i, gid, ty) in [
            (0, "0000:0000:0000:0000:0000:ffff:0a64:0001", "IB/RoCE v1"),
            (1, "fe80:0000:0000:0000:0000:0000:0000:0001", "RoCE v2"),
            (3, "0000:0000:0000:0000:0000:ffff:0a64:0001", "RoCE v2"),
        ] {
            write(&root, &format!("{port}/gids/{i}"), gid);
            write(&root, &format!("{port}/gid_attrs/types/{i}"), ty);
            write(&root, &format!("{port}/gid_attrs/ndevs/{i}"), "enp1s0f0np0");
        }
        // An unused slot has a GID file and no readable type.
        write(
            &root,
            &format!("{port}/gids/2"),
            "0000:0000:0000:0000:0000:0000:0000:0000",
        );

        let m = probed(&root);
        assert_eq!(m["kernel"], "6.11.0-1016-nvidia");
        assert_eq!(m["driver.nvidia"], "580.95.05");
        assert!(!m.contains_key("driver.amdgpu"));
        assert_eq!(m["rdma.mlx5_0.1.state"], "ACTIVE");
        assert_eq!(m["rdma.mlx5_0.1.gid"], "3 10.100.0.1 enp1s0f0np0");
    }

    #[test]
    fn a_box_without_the_files_reports_nothing() {
        assert!(probed(&tmp("empty")).is_empty());
    }

    #[test]
    fn variables_name_their_keys_in_lowercase() {
        let vars = [
            ("MENTAT_META_IMAGE", "vllm:0.11"),
            ("MENTAT_META_FILE", "/etc/meta.json"),
            ("MENTAT_GROUP", "g"),
        ]
        .map(|(k, v)| (k.to_string(), v.to_string()));
        let m = from_env(vars.into_iter());
        assert_eq!(m, Meta::from([("image".into(), "vllm:0.11".into())]));
    }

    #[test]
    fn a_file_gives_strings_and_the_json_text_of_anything_else() {
        let root = tmp("file");
        let p = root.join("meta.json");
        std::fs::write(&p, r#"{"image.base": "cuda:12.8", "layers": 14}"#).unwrap();
        let m = from_file(&p).unwrap();
        assert_eq!(m["image.base"], "cuda:12.8");
        assert_eq!(m["layers"], "14");
        std::fs::write(&p, "[1]").unwrap();
        assert!(from_file(&p).is_err());
    }

    #[test]
    fn bad_keys_long_values_and_extra_keys_are_dropped() {
        let mut m = Meta::new();
        m.insert("Upper".into(), "x".into());
        m.insert("long".into(), "x".repeat(MAX_VALUE_LEN + 1));
        for i in 0..MAX_KEYS + 1 {
            m.insert(format!("k{i:03}"), "v".into());
        }
        let (kept, dropped) = bounded(m);
        assert_eq!(kept.len(), MAX_KEYS);
        assert!(dropped.contains(&"Upper".to_string()));
        assert!(dropped.contains(&"long".to_string()));
        assert_eq!(dropped.len(), 3);
    }

    #[test]
    fn only_ipv4_mapped_gids_give_an_address() {
        assert_eq!(
            ipv4_of_gid("0000:0000:0000:0000:0000:ffff:c0a8:0146").as_deref(),
            Some("192.168.1.70")
        );
        assert_eq!(ipv4_of_gid("fe80:0000:0000:0000:0000:0000:0000:0001"), None);
    }
}
