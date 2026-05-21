import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
from data_processing import load_data_visemb_from_lightning_ckpt, plot_scatter_hyper

app = dash.Dash(__name__)

@app.callback(
    Output('click-data', 'children'),
    [Input('scatter-plot', 'clickData')]
)
def display_click_data(clickData):
    if clickData is not None and 'points' in clickData and len(clickData['points']) > 0:
        customdata = clickData['points'][0].get('customdata')
        if customdata:
            return html.Img(src=customdata, style={'width': '200px', 'height': '200px'})
    return "Click on a point to see the image."


if __name__ == '__main__':
    gathered_datas, gathered_val_vis, all_noist_test_result_dict, gathered_val_label = load_data_visemb_from_lightning_ckpt(
        'zzl_saveckpt/best_model_mnist_acc0.9453999999999999.pth', sample_num=3000)
    image_files = [f'path/to/your/images/image{i}.png' for i in range(1, 101)]  # Replace with actual paths to your images
    fig = plot_scatter_hyper(
        gathered_datas, 
        gathered_val_vis, 
        gathered_val_label, 
        R=1.5, 
        image_files=image_files
    )

    app.layout = html.Div([
        html.Div([
            dcc.Graph(id='scatter-plot', figure=fig)
        ], style={'display': 'inline-block', 'width': '80%'}),
        html.Div([
            html.Div(id='click-data', style={'textAlign': 'center', 'padding': '20px'})
        ], style={'display': 'inline-block', 'width': '20%'})
    ])

    app.run_server(debug=True, port=8051)