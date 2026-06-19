from pyvis.network import Network


def render_graph(G, filename="graph_latest.html"):
    """
    Interactive crisis belief graph visualization.
    """

    net = Network(
        height="750px",
        width="100%",
        bgcolor="#0f1117",
        font_color="white",
        directed=False
    )

    # =====================================================
    # ADD NODES
    # =====================================================

    for node_id, data in G.nodes(data=True):

        stance = data.get("stance", "NEUTRAL")

        if stance == "SUPPORT":
            color = "#4CAF50"
        elif stance == "CONTRADICT":
            color = "#F44336"
        else:
            color = "#9E9E9E"

        net.add_node(
            node_id,
            label=data["text"][:25],
            title=f"""
Text: {data['text']}

Stance: {stance}
Relevance: {data.get('relevance', 0):.2f}
Confidence: {data.get('confidence', 0):.2f}
Subreddit: {data.get('subreddit', 'unknown')}
""",
            color=color,
            size=8 + data.get("relevance", 0) * 20
        )

    # =====================================================
    # ADD EDGES
    # =====================================================

    for u, v, data in G.edges(data=True):

        edge_type = data.get("type", "SUPPORT")

        color = "#4CAF50" if edge_type == "SUPPORT" else "#F44336"

        net.add_edge(
            u,
            v,
            value=data.get("weight", 1.0),
            color=color
        )

    # =====================================================
    # PHYSICS
    # =====================================================

    net.force_atlas_2based()

    net.write_html(filename)

    import webbrowser
    webbrowser.open(f"file://{filename}")

    print(f"\n[GRAPH SAVED] Open {filename} in your browser\n")