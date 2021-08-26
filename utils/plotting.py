import plotly.express as px
import plotly.figure_factory as ff
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
