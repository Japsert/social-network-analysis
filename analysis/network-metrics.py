#!/usr/bin/env python3
"""
Network metrics for a Delicious bookmark dataset (dataset.tsv).

Line format (tab separated, no header):
    url_md5  user_id  url  unix_timestamp  tag1  [tag2 ...]

Two graph modes (--graph):
  user-url  Directed bipartite graph, user -> URL (one edge per distinct
            user/URL pair).  out-degree(user) = distinct URLs bookmarked,
            in-degree(URL) = distinct users who bookmarked it.   [default]
  tag-cooc  Undirected tag co-occurrence graph (two tags are linked when
            they appear on the same bookmark).  In/out degree does not exist
            here and is skipped (noted at the end).

Metrics: vertices, edges, degree distribution (in/out), centrality indices,
clustering coefficient, diameter, density, number and sizes of connected
components.

Usage:
    uv run network_metrics.py dataset.tsv --graph user-url
    uv run network_metrics.py dataset.tsv --graph tag-cooc --max-rows 2000000
"""
import argparse
import itertools
import os
import random
from collections import Counter

import igraph as ig
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

REPORT = []
NOTES = []


def out(text=""):
    print(text)
    REPORT.append(text)


def note(text):
    NOTES.append(text)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def read_bookmarks(path, max_rows=None):
    """Yield (url_md5, user_id, url, timestamp, [tags]) per line."""
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        for i, line in enumerate(fh):
            if max_rows and i >= max_rows:
                break
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4 or not parts[3].strip().isdigit():
                continue  # malformed line or header
            tags = [t.strip().lower() for t in parts[4:] if t.strip()]
            yield parts[0], parts[1], parts[2], int(parts[3]), tags


def edges_user_url(path, max_rows):
    edges, url_of, n_rows = set(), {}, 0
    for md5, user, url, _, _ in read_bookmarks(path, max_rows):
        n_rows += 1
        edges.add((f"user:{user}", f"url:{md5}"))
        url_of.setdefault(f"url:{md5}", url)
    out(f"Read {n_rows:,} bookmarks -> {len(edges):,} distinct user-URL edges")
    return edges, url_of


def edges_tag_cooc(path, max_rows, max_tags):
    pairs, n_rows, truncated = Counter(), 0, 0
    for _, _, _, _, tags in read_bookmarks(path, max_rows):
        n_rows += 1
        tags = sorted(set(tags))
        if len(tags) > max_tags:
            tags = tags[:max_tags]
            truncated += 1
        pairs.update(itertools.combinations(tags, 2))
    out(f"Read {n_rows:,} bookmarks -> {len(pairs):,} distinct tag-tag edges")
    if truncated:
        note(f"{truncated:,} bookmarks had more than {max_tags} tags; only the first "
             f"{max_tags} (alphabetically) were used to limit pair explosion.")
    note("Tag co-occurrence edge weights (co-occurrence counts) are not used; "
         "all metrics are computed on the unweighted graph.")
    return set(pairs), {}


# --------------------------------------------------------------------------
# Graph construction (networkx + igraph share the same integer node ids)
# --------------------------------------------------------------------------
def build_graphs(edges, directed):
    index, names = {}, []

    def idx(x):
        i = index.get(x)
        if i is None:
            i = index[x] = len(names)
            names.append(x)
        return i

    int_edges = [(idx(a), idx(b)) for a, b in edges]
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(len(names)))
    G.add_edges_from(int_edges)
    g = ig.Graph(n=len(names), edges=int_edges, directed=directed)
    g.vs["name"] = names
    return G, g, names


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def describe(label, arr):
    arr = np.asarray(arr)
    if arr.size == 0:
        out(f"  {label}: (no nodes)")
        return
    out(f"  {label}: n={arr.size:,}  min={arr.min():.4g}  max={arr.max():.4g}  "
        f"mean={arr.mean():.4g}  median={np.median(arr):.4g}")


def top_k(label, values, names, pretty, k=10):
    order = np.argsort(values)[::-1][:k]
    out(f"  Top {k} by {label}:")
    for i in order:
        out(f"    {values[i]:.6g}\t{pretty(names[i])}")


def loglog_hist(ax, arr, title):
    arr = np.asarray(arr)
    arr = arr[arr > 0]
    if arr.size == 0:
        ax.set_title(title + " (empty)")
        return
    vals, counts = np.unique(arr, return_counts=True)
    ax.loglog(vals, counts, "o", ms=3)
    ax.set_xlabel("degree")
    ax.set_ylabel("number of nodes")
    ax.set_title(title)


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def analyse(G, g, names, directed, mode, url_of, args):
    rng = random.Random(args.seed)
    n, m = G.number_of_nodes(), G.number_of_edges()
    is_user = np.array([nm.startswith("user:") for nm in names])
    pretty = lambda nm: f"{nm}  {url_of[nm]}" if nm in url_of else nm  # noqa: E731

    # ---- size, density -----------------------------------------------------
    out("\n=== BASIC SIZE ===")
    out(f"Number of vertices: {n:,}")
    out(f"Number of edges:    {m:,}")
    out(f"Density: {nx.density(G):.6g}")
    if mode == "user-url":
        nu, nv = int(is_user.sum()), int((~is_user).sum())
        out(f"  users: {nu:,}   URLs: {nv:,}")
        out(f"  bipartite density m/(|users|*|URLs|): {m / (nu * nv):.6g}")
        note("Density is reported both in the standard form (m/(n(n-1)) for a directed "
             "graph) and in the bipartite form; the standard form is tiny by construction.")

    # ---- degree distribution ----------------------------------------------
    out("\n=== DEGREE DISTRIBUTION ===")
    fig, axes = plt.subplots(1, 2 if directed else 1, figsize=(11 if directed else 6, 4.5))
    if directed:
        in_deg = np.array([d for _, d in G.in_degree()])
        out_deg = np.array([d for _, d in G.out_degree()])
        describe("out-degree of users (distinct URLs bookmarked)", out_deg[is_user])
        describe("in-degree of URLs (distinct users bookmarking)", in_deg[~is_user])
        loglog_hist(axes[0], in_deg[~is_user], "In-degree (URLs)")
        loglog_hist(axes[1], out_deg[is_user], "Out-degree (users)")
        note("In a user->URL graph users have in-degree 0 and URLs have out-degree 0, so "
             "out-degree is summarised over users only and in-degree over URLs only.")
    else:
        deg = np.array([d for _, d in G.degree()])
        describe("degree", deg)
        loglog_hist(axes, deg, "Degree distribution (tags)")
        note("In-degree and out-degree: SKIPPED. The tag co-occurrence graph is undirected, "
             "so only plain degree exists.")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "degree_distribution.png"), dpi=150)
    plt.close(fig)

    # ---- connected components ---------------------------------------------
    out("\n=== CONNECTED COMPONENTS ===")
    comps = nx.weakly_connected_components(G) if directed else nx.connected_components(G)
    sizes = np.array(sorted((len(c) for c in comps), reverse=True))
    kind = "weakly connected" if directed else "connected"
    out(f"Number of {kind} components: {len(sizes):,}")
    out(f"Largest component: {sizes[0]:,} nodes ({100 * sizes[0] / n:.2f}% of all nodes)")
    out(f"Top 10 component sizes: {sizes[:10].tolist()}")
    describe("component size", sizes)
    out(f"Isolated nodes (size 1): {int((sizes == 1).sum()):,}")
    if directed:
        scc = [len(c) for c in nx.strongly_connected_components(G)]
        out(f"Number of strongly connected components: {len(scc):,} "
            f"(largest: {max(scc):,})")
        if mode == "user-url":
            note("Strongly connected components are trivial in a user->URL graph (no "
                 "directed cycles), so every node is its own SCC; weak components are the "
                 "meaningful ones.")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    vals, counts = np.unique(sizes, return_counts=True)
    ax.loglog(vals, counts, "o", ms=3)
    ax.set_xlabel("component size")
    ax.set_ylabel("number of components")
    ax.set_title(f"Size of {kind} components")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "component_sizes.png"), dpi=150)
    plt.close(fig)

    # ---- clustering coefficient -------------------------------------------
    out("\n=== CLUSTERING COEFFICIENT ===")
    if mode == "user-url":
        out("Classical (triangle-based) clustering: 0 by definition (bipartite graph "
            "has no triangles).")
        U = G.to_undirected(as_view=True)
        user_ids = [i for i in range(n) if is_user[i]]
        url_ids = [i for i in range(n) if not is_user[i]]
        for label, ids in (("users", user_ids), ("URLs", url_ids)):
            sample = rng.sample(ids, min(args.samples, len(ids)))
            cc = nx.bipartite.clustering(U, nodes=sample, mode="dot")
            out(f"Bipartite (Latapy) clustering, {label}: "
                f"{np.mean(list(cc.values())):.6g}  (sample of {len(sample):,} nodes)")
        note("Standard clustering coefficient is 0 for a bipartite graph; the Latapy "
             "bipartite clustering coefficient is reported instead, on a random node "
             "sample for speed (--samples).")
    else:
        if n <= args.exact_limit:
            out(f"Average clustering coefficient: {nx.average_clustering(G):.6g}")
            out(f"Global clustering (transitivity): {nx.transitivity(G):.6g}")
        else:
            sample = rng.sample(range(n), args.samples * 5)
            out(f"Average clustering coefficient (sample of {len(sample):,} nodes): "
                f"{nx.average_clustering(G, nodes=sample):.6g}")
            note("Clustering was estimated on a random node sample; global transitivity "
                 "was skipped because the graph exceeds --exact-limit.")

    # ---- diameter + centralities on giant component -----------------------
    giant = g.connected_components(mode="weak").giant()
    giant_u = giant.as_undirected(mode="collapse") if directed else giant
    ng = giant_u.vcount()
    gnames = giant_u.vs["name"]
    exact = ng <= args.exact_limit

    out("\n=== DIAMETER ===")
    if exact:
        diam = giant_u.diameter()
        out(f"Diameter of the largest component ({ng:,} nodes): {diam} (exact)")
    else:
        sample = rng.sample(range(ng), min(args.samples, ng))
        diam = int(max(giant_u.eccentricity(vertices=sample)))
        out(f"Diameter of the largest component ({ng:,} nodes): >= {diam} "
            f"(lower bound from {len(sample):,} sampled BFS sources)")
        note("Diameter is a lower bound (sampled eccentricities) because the largest "
             "component exceeds --exact-limit.")
    if directed:
        note("Diameter is computed on the undirected version of the largest weakly "
             "connected component; the directed diameter of a user->URL graph is "
             "trivially 1 (paths have length 1) and most pairs are unreachable.")
    else:
        note("Diameter is defined only within a connected component, so the largest "
             "component is used.")

    out("\n=== CENTRALITY INDICES ===")
    # Degree centrality (whole graph)
    if directed:
        in_c = in_deg / (n - 1)
        out_c = out_deg / (n - 1)
        top_k("in-degree centrality (most bookmarked URLs)", in_c, names, pretty)
        top_k("out-degree centrality (most active users)", out_c, names, pretty)
        cent_for_plot = {"in-degree centrality": in_c[~is_user],
                         "out-degree centrality": out_c[is_user]}
    else:
        dc = deg / (n - 1)
        top_k("degree centrality", dc, names, pretty)
        cent_for_plot = {"degree centrality": dc}

    # PageRank (whole graph)
    pr = np.array(g.pagerank(directed=directed))
    top_k("PageRank", pr, names, pretty)
    cent_for_plot["PageRank"] = pr

    # Eigenvector centrality (giant component, undirected view)
    try:
        ev = np.array(giant_u.eigenvector_centrality())
        top_k("eigenvector centrality (largest component)", ev, gnames, pretty)
        cent_for_plot["eigenvector centrality"] = ev
    except Exception as exc:  # ARPACK may fail to converge
        out(f"  Eigenvector centrality failed: {exc}")
        note(f"Eigenvector centrality SKIPPED (did not converge: {exc}).")

    # Closeness (giant component)
    if exact:
        cl = np.array(giant_u.closeness(normalized=True))
        top_k("closeness centrality (largest component)", cl, gnames, pretty)
    else:
        cand = np.argsort(giant_u.degree())[::-1][: args.samples].tolist()
        cl = np.array(giant_u.closeness(vertices=cand, normalized=True))
        cand_names = [gnames[i] for i in cand]
        top_k(f"closeness centrality (among top-{len(cand)} degree nodes)", cl,
              cand_names, pretty)
        note("Closeness was computed only for the highest-degree nodes of the largest "
             "component (exact all-pairs closeness is too expensive at this size).")
    cent_for_plot["closeness centrality"] = cl

    # Betweenness (giant component)
    if exact:
        bc = np.array(giant_u.betweenness())
        top_k("betweenness centrality (largest component)", bc, gnames, pretty)
    else:
        bc = np.array(giant_u.betweenness(cutoff=args.bc_cutoff))
        top_k(f"betweenness centrality (paths up to length {args.bc_cutoff})", bc,
              gnames, pretty)
        note(f"Betweenness was approximated by only counting shortest paths of length "
             f"<= {args.bc_cutoff} (--bc-cutoff).")
    cent_for_plot["betweenness centrality"] = bc

    note("Closeness, betweenness and eigenvector centrality are computed on the "
         "undirected view of the largest component; degree centrality and PageRank "
         "use the full graph.")

    # Centrality distribution plot
    k = len(cent_for_plot)
    cols = 3
    rows = -(-k // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows))
    axs = np.atleast_1d(axs).ravel()
    for ax, (label, vals) in zip(axs, cent_for_plot.items()):
        vals = np.asarray(vals)
        n_unique = np.unique(vals).size
        ax.hist(vals, bins=max(1, min(50, n_unique)), log=True)
        ax.set_title(label)
        ax.set_xlabel("value")
        ax.set_ylabel("nodes (log)")
    for ax in axs[k:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "centralities.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", default="dataset.tsv")
    ap.add_argument("--graph", choices=["user-url", "tag-cooc"], default="user-url")
    ap.add_argument("--max-rows", type=int, default=None, help="only read the first N lines")
    ap.add_argument("--max-tags", type=int, default=15,
                    help="tag-cooc: max tags per bookmark used for pairs")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--exact-limit", type=int, default=20000,
                    help="largest component size up to which exact (slow) algorithms run")
    ap.add_argument("--samples", type=int, default=1000,
                    help="sample size for approximations")
    ap.add_argument("--bc-cutoff", type=int, default=4,
                    help="path-length cutoff for approximate betweenness")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    directed = args.graph == "user-url"
    out(f"Graph mode: {args.graph} ({'directed' if directed else 'undirected'})")
    if directed:
        edges, url_of = edges_user_url(args.file, args.max_rows)
    else:
        edges, url_of = edges_tag_cooc(args.file, args.max_rows, args.max_tags)
    if not edges:
        raise SystemExit("No edges found - check the file path and format.")

    G, g, names = build_graphs(edges, directed)
    del edges
    analyse(G, g, names, directed, args.graph, url_of, args)

    out("\n=== NOTES (skipped / adapted metrics) ===")
    for i, t in enumerate(dict.fromkeys(NOTES), 1):
        out(f"{i}. {t}")

    with open(os.path.join(args.outdir, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(REPORT) + "\n")
    print(f"\nReport and figures saved in ./{args.outdir}/")


if __name__ == "__main__":
    main()
