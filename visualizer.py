
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def plot_predictions(res: dict, ticker: str) -> go.Figure:
    """
    Plots Train Actual/Preds, Test Actual/Preds, and Future Forecast seamlessly.
    """
    fig = go.Figure()

    # Historical Train Data
    fig.add_trace(go.Scatter(
        x=res["train_dates"], y=res["y_train"],
        name="Train Actual", line=dict(color="#636EFA",
                                        width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=res["train_dates"], y=res["train_pred"],
        name="Train Predicted", line=dict(color="#EF553B", 
                                          width=1.5, 
                                          dash="dash")
    ))

    # Historical Test Data
    fig.add_trace(go.Scatter(
        x=res["test_dates"], y=res["y_test"],
        name="Test Actual", line=dict(color="#00CC96", 
                                      width=2)
    ))
    fig.add_trace(go.Scatter(
        x=res["test_dates"], y=res["test_pred"],
        name="Test Predicted", line=dict(color="#AB63FA", 
                                         width=2, 
                                         dash="dash")
    ))

    # Future Out-of-Sample Forecast
    fig.add_trace(go.Scatter(
        x=res["future_dates"], y=res["future_pred"],
        name="Future Forecast", line=dict(color="#FFA15A",
                                           width=2.5)
    ))

    fig.update_layout(
        title=f"{ticker} - Model Performance & Future Projection",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_feature_importance(fi: pd.Series) -> go.Figure:
    """Plots a horizontal bar chart showing what elements drove the model."""
    if fi is None or fi.empty:
        # Return empty placeholder figure if Linear Regression was used
        fig = go.Figure()
        fig.update_layout(title="Feature Importance not available for this model type.")
        return fig
        
    df_fi = fi.reset_index()
    df_fi.columns = ["Feature", "Importance"]
    df_fi = df_fi.sort_values(by="Importance", ascending=True)

    fig = px.bar(
        df_fi, x="Importance", y="Feature", orientation="h",
        title="Feature Importance Breakdown",
        labels={"Importance": "Relative Score", "Feature": "Indicator/Lag"},
        template="plotly_dark",
        color="Importance",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    return fig