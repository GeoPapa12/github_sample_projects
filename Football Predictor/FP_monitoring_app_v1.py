

import dash

import dash_core_components as dcc
import dash_html_components as html
# import dash_daq as daq
from dash.dependencies import Input, Output  # State
# import dash_bootstrap_components as dbc
# import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as tls
from plotly.subplots import make_subplots

from flask import request
import webbrowser
from threading import Timer
import numpy as np
import pandas as pd
import sys
import datetime
from datetime import timedelta, date

from Football_Predictor import *
from Data_Construction_Functions import *
from Parsing_Functions import *


global df_stock_data
# function that scraps the sp500 comapany names from wikipedia
peri_list = ['1011', '1112', '1213', '1314', '1415', '1516', '1617', '1718', '1819', '1920', '2021']
leagues_list = ['E0', 'TBC']
leagues_df = pd.DataFrame(leagues_list, columns=['label'])
leagues_df['value'] = leagues_df['label']  # 'League'
peri_df = pd.DataFrame(peri_list, columns=['label'])
peri_df['value'] = peri_df['label']  # 'period'
fixt_list = list(range(5,39))
fixt_list = [str(x) for x in fixt_list]
fixt_df = pd.DataFrame(fixt_list, columns=['label'])
fixt_df['value'] = fixt_df['label']  # 'fixt'

plot_type_list = ['Goals', 'Shoots', 'Shoots on Target']
plot_type = pd.DataFrame(plot_type_list, columns=['label'])
plot_type['value'] = plot_type['label']  # 'fixt'

fixt_names_df = pd.DataFrame(columns=['label', 'value'])
# ==============================================#==============================================
# ==============================================#==============================================
# ==============================================#==============================================
# ==============================================#==============================================
# ==============================================#==============================================
# ==============================================#==============================================

def shutdown():
    '''
    the function is used when the user closes the browser window,
    this function pauses the server
    '''
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


def fig_layout_function(fig, game, graph_title):
    '''
    dark layout (transparent)
    '''
    fig['layout'].update({
                  'colorway': ["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                  'template': 'plotly_dark',
                  'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                  'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                  'margin': {'l': 10, 'r': 10, 'b': 50, 't': 25},
                  'hovermode': 'x',
                  'autosize': True,
                  'legend': {'x': 0, 'y': -0.1},
                  'legend_orientation': "h",
                  'title': {'text': graph_title, 'font': {'color': 'white'}, 'x': 0.5},
                  'yaxis1': {'title': 'Home Team:   ' + game.split("/ ")[0]},
                  'yaxis2': {'title': 'Away Team:   ' + game.split("/ ")[1]},
                  'xaxis2': {'title': 'Game No'}
                  # 'xaxis': {'range': [df.index.min(), df.index.max()]},
                  })
    return fig


# ==============================================#==============================================
# ==============================================#==============================================
# ==============================================#==============================================
# ==============================================#==============================================


# Start of the dashboard
app = dash.Dash(__name__)
server = app.server

app.title = 'Stock Value Monitoring'


app.layout = html.Div([
    html.Div(
        # left hand part of the dashboard (dropdown, news...)
        [
            dcc.Interval(id="interval", interval=5 * 1000, n_intervals=0),
            dcc.Interval(id="interval_2", interval=5 * 1000, n_intervals=0),

            html.H2('Result Predictor'),

            html.Div(
                className="my-class",
                children=[

                    html.H6('Choose League'),
                    dcc.Dropdown(
                        id='drop_down_League',
                        options=leagues_df.to_dict('r'),
                        value='E0',  # initial value to be displayed
                        style=dict(width='75%', verticalAlign="middle"), clearable=False),

                    html.H6('Choose Period & Fixture'),
                    dcc.Dropdown(
                        id='drop_down_period',
                        options=peri_df.to_dict('r'),
                        value='2021',  # initial value to be displayed
                        style=dict(width='75%', verticalAlign="middle"), clearable=False),

                    dcc.Dropdown(
                         id='drop_down_fixt',
                         options=fixt_df.to_dict('r'),
                         value=34,  # initial value to be displayed
                         style=dict(width='75%', verticalAlign="middle"), clearable=False),

                    html.P(''),

                    html.H6('Choose Game & Graph type'),

                    dcc.Dropdown(
                        id='drop_down_game',
                        options=fixt_names_df.to_dict('r'),
                        value="Arsenal/ Tottenham",  # initial value to be displayed
                        style=dict(width='75%', verticalAlign="middle"), clearable=False),

                    dcc.Dropdown(
                        id='drop_down_graph_type',
                        options=plot_type.to_dict('r'),
                        value="Goals",  # initial value to be displayed
                        style=dict(width='75%', verticalAlign="middle"), clearable=False),

                    html.Table(id='ranking-Table', style={"margin-top": "20px", "margin-bottom": "150px"}),

                    html.P('\n\n\n'),

                    html.Table(id='factors_table'),

                    ],

                ),

        ], style={'marginLeft': 5, 'width': '25%', 'display': 'inline-block', 'vertical-align': 'Top'}),


    html.Div(

        # right hand part of the dashboard (financial table, graphs)
        [
            html.P(''),

            html.H6(id='general_heading'),
            html.P(''),

            html.Table(id='results_table'),

            html.P(''),

            html.Div(children=[
                html.P(''),
                dcc.Graph(id='perXgames_graph', style=dict(border="0px groove grey")),
                html.P(''),
                html.P('')], style={"border": "1px groove grey"},
            ),

            html.P(''),



        ], style={'marginLeft': 15, 'width': '52%', 'display': 'inline-block', 'vertical-align': 'Top'}),

    # code to pause the server when the browser window is closed
    dcc.Location(id='url', refresh=False),
    dcc.Link('', href='/'),
    html.Br(),
    dcc.Link('', href='/page-2'),
    # https://stackoverflow.com/questions/55620642/plotly-dash-python-how-to-stop-execution-after-time
    # content will be rendered in this element
    html.Div(id='page-content')

])


# Callback to pause server when the browser page is closed
@app.callback(dash.dependencies.Output('page-content', 'children'), [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/shutdown':
        shutdown()
    return html.Div([
        html.H3('')
    ])


# Callback to update live clock
@app.callback(Output("live_clock", "children"),
              [Input("interval", "n_intervals")])
def update_time(n):
    return datetime.datetime.now().strftime("%H:%M:%S")


# Callback to update the general title for each company
@app.callback(Output(component_id='general_heading', component_property='children'),
              [Input(component_id='drop_down_period', component_property='value'),
               Input(component_id='drop_down_fixt', component_property='value')])
def gen_heading_update(selected_period, selected_fixt):
    company_name = selected_period  # (df_stock_data[df_stock_data['value'] == selected_period]['label'])[0]
    return 'Premier League/ Period: ' + company_name + '/ Fixture: ' + str(selected_fixt)


# Callback to update the results_table
@app.callback(Output('factors_table', 'children'),
              [Input('drop_down_period', 'value'),
               Input('drop_down_fixt', 'value'),
               Input('drop_down_League', 'value')])
def table_factors_table(selected_period, selected_fixt, selected_League):
    global results_df  # Needed to modify global copy of financialreportingdf

    max_rows = 10
    results_df = factors_current_fixt(selected_period, int(selected_fixt), selected_League)

    # Header
    return [html.Tr([html.Th(col) for col in results_df.columns])] + [html.Tr([
        html.Td(results_df.iloc[i][col]) for col in results_df.columns
    ]) for i in range(min(len(results_df), max_rows))]


# Callback to update the results_table
@app.callback(Output('results_table', 'children'),
              [Input('drop_down_period', 'value'),
               Input('drop_down_fixt', 'value'),
               Input('drop_down_League', 'value')])
def table_results_table(selected_period, selected_fixt, selected_League):
    global results_df  # Needed to modify global copy of financialreportingdf

    max_rows = 10

    results_df = ML_Prediction(selected_period, int(selected_fixt), selected_League, ML_train=0, print_selection=0)

    # Header
    return [html.Tr([html.Th(col) for col in results_df.columns])] + [html.Tr([
        html.Td(results_df.iloc[i][col]) for col in results_df.columns
    ]) for i in range(min(len(results_df), max_rows))]


# Callback to update the results_table
@app.callback(Output('ranking-Table', 'children'),
              [Input('drop_down_period', 'value'),
               Input('drop_down_fixt', 'value'),
               Input('drop_down_League', 'value')])
def table_ranking_table(selected_period, selected_fixt, selected_League):
    global table_ranking_df  # Needed to modify global copy of financialreportingdf

    table_ranking_df = pd.DataFrame()
    max_rows = 20

    table_ranking_df = table_ranking(selected_period, int(selected_fixt), selected_League)

    # Header
    return [html.Tr([html.Th(col) for col in table_ranking_df.columns])] + [html.Tr([
        html.Td(table_ranking_df.iloc[i][col]) for col in table_ranking_df.columns
    ]) for i in range(min(len(table_ranking_df), max_rows))]


# Callback to update the 'perXgames_graph'
@app.callback(Output(component_id='perXgames_graph', component_property='figure'),
              [Input(component_id='drop_down_period', component_property='value'),
               Input(component_id='drop_down_fixt', component_property='value'),
               Input(component_id='drop_down_game', component_property='value'),
               Input(component_id='drop_down_League', component_property='value'),
               Input(component_id='drop_down_graph_type', component_property='value')])
def chart_team_factors(selected_period, selected_fixt, selected_game, selected_League, selected_graph_type):

    try:
        selected_game = selected_game['value']
        selected_League = selected_League['value']

    except Exception:
        pass

    df1, df2 = plot_factors_dataframes(selected_period, int(selected_fixt), selected_game, selected_graph_type, selected_League)

    if selected_graph_type == "Goals":
        plot_key = "Goals"
    elif selected_graph_type == 'Shoots':
        plot_key = "Sht"
    elif selected_graph_type == 'Shoots on Target':
        plot_key = "ShTar"

    fig = tls.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.009, horizontal_spacing=0.009)
    fig.add_trace(go.Scatter(x=df1.fixt, y=df1['Htm_' + plot_key + '_favour'], name='Htm_' + plot_key + '_favour', line=dict(color="#006600")), 1, 1)
    fig.add_trace(go.Scatter(x=df1.fixt, y=df1['Htm_' + plot_key + '_against'], name='Htm_' + plot_key + '_against', line=dict(color="#990000")), 1, 1)
    fig.add_trace(go.Scatter(x=df2.fixt, y=df2['Atm_' + plot_key + '_favour'], name='Atm_' + plot_key + '_favour', line=dict(color="#009933")), 2, 1)
    fig.add_trace(go.Scatter(x=df2.fixt, y=df2['Atm_' + plot_key + '_against'], name='Atm_' + plot_key + '_against', line=dict(color="#ff6666")), 2, 1)

    fig_layout_function(fig, selected_game, 'Home & Away ' + selected_graph_type + ' Per 3 Games- ' + selected_game)
    fig['layout'].update(height=750)
    fig['layout'].update(width=1000)

    return fig
  

@app.callback([Output('drop_down_game', 'options'),
               Output('drop_down_game', 'value')],
              [Input('drop_down_period', 'value'),
               Input('drop_down_fixt', 'value'),
               Input('drop_down_League', 'value')])
def update_dropdown_menu(selected_period, selected_fixt, selected_League):
    fixt_names_df = last_fixture_teams(selected_period, int(selected_fixt), selected_League)
    print(fixt_names_df)
    return fixt_names_df.to_dict('r'), fixt_names_df.to_dict('r')[0]


# this function automatically opens the browser
def open_browser():
    webbrowser.open_new('http://127.0.0.1:8050/')
    # Timer(1, open_browser).start();


# the timer is used so the open_browser() function to automatically open the browser
if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run_server(debug=False)
