import plotly.graph_objects as go


def render_timeline(history):
    """
    Visualizes how confidence changes over time.
    Each point = one pipeline run.
    """

    if not history:
        print("[Timeline] No history to display.")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[h["timestamp"] for h in history],
        y=[h["confidence"] for h in history],
        mode="lines+markers",
        line=dict(width=3),
        marker=dict(size=6)
    ))

    fig.update_layout(
        title="Crisis Belief Evolution Over Time",
        xaxis_title="Timestamp",
        yaxis_title="Confidence Score",
        yaxis=dict(range=[0, 1])
    )

    fig.show()