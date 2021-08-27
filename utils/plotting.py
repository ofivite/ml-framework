import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import seaborn as sns

def plot_class_score(df, class_id, class_to_info, how='density', weight=None):
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    if how=='density':
        if 'pred_class' not in df.columns or \
            'gen_target' not in df.columns or \
              'pred_class_proba' not in df.columns:
            raise KeyError('Couldn\'t find pred_class/gen_target/pred_class_proba in DataFrame with predictions')
        hist_data = [df.query(f'pred_class == {class_id} and gen_target == {i}')['pred_class_proba'] for i in class_to_info]
        class_labels = [class_to_info[i].name for i in class_to_info]
        class_colors = [f'rgba({class_to_info[i].color}, 1.)' if i==class_id # emphasize category class by increase in transparency
                                                              else f'rgba({class_to_info[i].color}, .2)'
                                                              for i in class_to_info]
        fig = ff.create_distplot(hist_data, class_labels, bin_size=5e-3, histnorm='probability', show_curve=True, show_rug=False, colors=class_colors)
        fig.update_layout(
            xaxis_title='model score',
            yaxis_title='a.u.',
            title_text=f'{class_to_info[class_id].name} category',
            autosize=False,
            width=800,
            height=500,
            margin=dict(l=20, r=20, t=20, b=20),
            # paper_bgcolor="LightSteelBlue",
        )
        return fig
    elif how=='stacked':
        fig = px.histogram(df.query(f'pred_class == {class_id}'), x="pred_class_proba", y=weight,
                   color="gen_target",
                   marginal="box", # or violin, or rug
                   barmode='group',
                   histfunc='sum',
                   nbins=50,
                   color_discrete_map={i: f'rgba({class_to_info[i].color}, {class_to_info[i].alpha})' for i in class_to_info},
                   log_y=True)
        fig.update_layout(
            xaxis_title='model score',
            yaxis_title='a.u.',
            title_text=f'{class_to_info[class_id].name} category',
            barmode='stack',
            autosize=False,
            width=1000,
            height=500,
            margin=dict(l=20, r=20, t=20, b=20),
        #     paper_bgcolor="LightSteelBlue",
        )
        return fig
    else:
        raise ValueError(f'Unknown value of how={how}: should be either \"density\" or \"stacked\"')

def plot_curves(df, class_to_info):
    curve_dict = {'roc': {}, 'pr': {}}
    fig_roc_curve = go.Figure()
    fig_pr_curve = go.Figure()

    fig_roc_curve.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig_pr_curve.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0.5, y1=0.5
    )

    for class_i, class_info in class_to_info.items():
        y_true = df['gen_target'] == class_i
        y_score = df[f'pred_class_{class_i}_proba']

        # weights to account for class imbalance (PR curve is sensitive to that)
        sample_weight_map = {}
        for class_label in set(y_true):
            sample_weight_map[class_label] = len(y_true)/np.sum(y_true == class_label)
        sample_weight = y_true.map(sample_weight_map)

        fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=sample_weight)
        auc_score = roc_auc_score(y_true, y_score, sample_weight=sample_weight)
        precision, recall, _ = precision_recall_curve(y_true, y_score, sample_weight=sample_weight)
        ap_score = average_precision_score(y_true, y_score, sample_weight=sample_weight)

        class_name = class_info.name
        curve_dict['roc'][f'auc_{class_name}'] = auc_score
        curve_dict['pr'][f'auc_{class_name}'] = ap_score
        name_roc = f"{class_name} (AUC={auc_score:.2f})"
        name_pr = f"{class_name} (AUC={ap_score:.2f})"
        fig_roc_curve.add_trace(go.Scatter(x=fpr, y=tpr, name=name_roc, mode='lines', marker_color=f'rgb({class_to_info[class_i].color})'))
        fig_pr_curve.add_trace(go.Scatter(x=recall, y=precision, name=name_pr, mode='lines', marker_color=f'rgb({class_to_info[class_i].color})'))

    fig_roc_curve.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=500
    )
    fig_pr_curve.update_layout(
        xaxis_title='Recall',
        yaxis_title='Precision',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=500
    )
    # fig_roc_curve.update_yaxes(type="log", range=[np.log(0.001), np.log(1.1)])
    fig_pr_curve.update_yaxes(range=[0.45, 1.1])

    curve_dict['roc']['figure'] = fig_roc_curve
    curve_dict['pr']['figure'] = fig_pr_curve
    return curve_dict