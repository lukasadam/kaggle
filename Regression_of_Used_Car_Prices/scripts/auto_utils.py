import autogluon.eda.analysis as eda
import autogluon.eda.visualization as viz
import autogluon.eda.auto as auto
import pandas as pd
import numpy as np

def _render_distribution_fit_information_if_available(state, label):
    if state.distributions_fit is not None:  # type: ignore # state is always present
        dist_fit_state = state.distributions_fit.train_data  # type: ignore
        dist_info = ["### Distribution fits for target variable"]
        if (label in dist_fit_state) and (len(dist_fit_state[label]) > 0):
            for d, p in state.distributions_fit.train_data[label].items():  # type: ignore
                dist_info.append(
                    f" - [{d}](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.{d}.html)"
                )
                if p.param is not None and len(p.param) > 0:
                    params = ", ".join([f"{shape}: {param}" for shape, param in zip(p.shapes, p.param)])
                    dist_info.append(f'   - p-value: {p["pvalue"]:.3f}')
                    dist_info.append(f"   - Parameters: ({params})")
        else:
            dist_info.append(
                f" - ⚠️ none of the [attempted](https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions) "  # type: ignore
                f"distribution fits satisfy specified minimum p-value threshold: `{state.distributions_fit_pvalue_min}`"
            )
        auto.analyze(viz_facets=[viz.MarkdownSectionComponent("\n".join(dist_info))])
    return state

def _render_correlation_analysis(state, train_data, label, corr_threshold=0.5):
    state = auto.analyze(
        train_data=train_data,
        state=state,
        return_state=True,
        label=label,
        anlz_facets=[eda.ApplyFeatureGenerator(category_to_numbers=True, children=[eda.interaction.Correlation(method='spearman', 
                                                                                                               focus_field=label, 
                                                                                                               focus_field_threshold=corr_threshold)])],
    )
    corr_info = ["### Target variable correlations"]
    if len(state.correlations_focus_high_corr.train_data) < 1:  # type: ignore
        corr_info.append(
            f" - ⚠️ no fields with absolute correlation greater than "  # type: ignore
            f"`{state.correlations_focus_field_threshold}` found for target variable `{label}`."
        )
    auto.analyze(
        state=state,
        viz_facets=[
            viz.MarkdownSectionComponent("\n".join(corr_info)),
            viz.CorrelationVisualization(
                headers=True
            ),
        ],
    )
    return state

def _render_features_highly_correlated_with_target(
    state, train_data, label,
):
    fields = state.correlations_focus_high_corr.train_data.index.tolist()  # type: ignore
    auto.analyze(
        train_data=train_data,
        state=state,
        return_state=True,
        anlz_facets=[eda.FeatureInteraction(key=f"{f}:{label}", x=f, y=label) for f in fields],
        viz_facets=[
            viz.FeatureInteractionVisualization(
                headers=True,
                key=f"{f}:{label}"
            )
            for f in fields
        ],
    )
    return state
