# -*- coding: utf-8 -*-
__author__ = "Luc Marechal"
__copyright__ = ""
__credits__ = ["Luc Marechal", "Lukas Lindenroth"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Luc Marechal"
__email__ = "luc.marechal(at)univ-smb.fr"
__status__ = "Debugg"

# Resources and documentation
# Beautifulsoup : https://www.digitalocean.com/community/tutorials/how-to-work-with-web-data-using-requests-and-beautiful-soup-with-python-3
# Greek symbol display : https://community.plotly.com/t/greek-symbols-with-latex-or-unicode/5531/9
# Determining which Input has fired : https://dash.plotly.com/advanced-callbacks
# Change range slider color : https://community.plotly.com/t/slide-box-color-of-plotly-rangeslider/34849

# Dash
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import dash_table
# Bootstrap
import dash_bootstrap_components as dbc
###########import simplejson as json
# Pandas and table
import pandas as pd
# Opimization
from scipy.optimize import least_squares
import numpy as np
from Hyperelastic import Hyperelastic
from HyperelasticStats import HyperelasticStats
# URL
import os
import requests
from bs4 import BeautifulSoup
import re

# Custom colors
sorored = '#CD3959' #'rgba(205,57,89,1)'
soroblack = '#4D4D4D'


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',dbc.themes.GRID]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, static_folder='/assets/')


def list_files(url):
    """List the files in the directory."""
    database = requests.get(url)
    soup = BeautifulSoup(database.text, 'html.parser')
    csvfiles = soup.find_all(title=re.compile("\.csv$")) # only lists *.csv files

    materials = []
    for filename in csvfiles:
        materials.append(filename.extract().get_text()[:-4])  # -4 to remove the file extension .csv

    return materials


def generate_data_table(dataframe, max_rows=10):
    dataframe=dataframe.round(4)
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])



#############################################################################
#  OPTIMIZATION
#############################################################################
# cost function to calculate the residuals. The fitting function holds the parameter values.  
def objectiveFun_Callback(parameters, exp_strain, exp_stress, hyperelastic):  
    theo_stress = hyperelastic.ConsitutiveModel(parameters, exp_strain)   
    residuals = theo_stress - exp_stress 
    return residuals

def optimization(model,dataframe):
    # Hyperelastic object
    hyperelastic = Hyperelastic(model, np.array([0]), order=3)  # Order is fixed to 3 for now for Ogden and the others models when applicable
    
    exp_strain = dataframe['True Strain'].values
    exp_stress = dataframe['True Stress (MPa)'].values
    # The least_squares package calls the Levenberg-Marquandt algorithm.
    # best-fit paramters are kept within optim_result.x
    optim_result = least_squares(objectiveFun_Callback, hyperelastic.initialGuessParam, method ='lm', args=(exp_strain, exp_stress, hyperelastic))
    optim_parameters = optim_result.x

    df_model_param = pd.DataFrame(optim_parameters, index=hyperelastic.param_names, columns=[model]).transpose()
    
    theo_stress = hyperelastic.ConsitutiveModel(optim_parameters, exp_strain)
    data_model = pd.DataFrame({'True Strain': exp_strain, 'True Stress (MPa)': theo_stress})
	
    stats = HyperelasticStats(exp_stress, theo_stress, hyperelastic.nbparam)
    aic = stats.aic()

    return df_model_param, data_model, aic



#####file2 = 'https://raw.githubusercontent.com/LucMarechal/Soft-Robotics-Materials-Database/master/Tensile-Tests-Data/RTV615.csv'
######data = pd.read_csv(file2, delimiter = ';',skiprows=16, names = ['Time (s)','True Strain','True Stress (MPa)','Eng. Strain','Eng. Stress (MPa)']) # the column headers are on line 8 from the top of the file 


# URL on the Github where the raw files are stored
github_url = 'https://github.com/LucMarechal/Soft-Robotics-Materials-Database/tree/master/Tensile-Tests-Data'
github_raw_url = 'https://raw.githubusercontent.com/LucMarechal/Soft-Robotics-Materials-Database/master/Tensile-Tests-Data/'
# Content of the GitHub repository. Lists all *.csv file name in the database
materials = list_files(github_url)
# Constitutive models
models = np.array(['Ogden', 'Mooney Rivlin', 'Veronda Westmann', 'Yeoh', 'Neo Hookean', 'Humphrey'])




nav = html.Nav(className = "nav nav-pills", children=[
html.A("Constitutive Models", className = "item2"),
html.A("Materials Comparison", className = "item3"),
html.A("Setup & Characterisation", className = "item4"),
html.A("GitHub",href='https://github.com/LucMarechal/Soft-Robotics-Materials-Database',className = "item1"),

	]),


app.title = "Soft Robotics Materials Database"

app.layout = html.Div(children=[

    dbc.Row([
        	
        dbc.Col(html.Img(
            	src='https://raw.githubusercontent.com/LucMarechal/data/master/logosoro.png',#'https://user-images.githubusercontent.com/36209435/72664756-f5b6d600-3a01-11ea-88b2-a3f3e46fe9f6.png',
            	style={'width': '80%'}
                ), width=3),

        dbc.Col(nav, align="center", width=9),

        
        #dbc.Col(dbc.Row([
        #	dbc.Col(html.A(html.Img(src=app.get_asset_url('logo_SYMME.svg'),style={'width': '350%'}),href='https://www.univ-smb.fr/symme/en/', className = "logos")), 
        #	dbc.Col(html.A(html.Img(src=app.get_asset_url('logo_USMB.svg'), style={'width': '350%'}),href='https://www.univ-smb.fr/en/', className = "logos")),
        #	], justify="end"), align="center", width=2),
    ], justify="between",),   


    html.H1(children='Constitutive Models',style={'padding': '15px'}),

 
    dbc.Row([

    dbc.Col([
    # Dropdown to select the meterial
        html.Label('Material'),
        
        dcc.Dropdown(
            id='dropdown-material',
            options=[{'label': i, 'value': i} for i in materials],   # dynamically fill in the dropdown menu
            value='RTV615',
            style={'width': '100%', 'marginBottom': '1em'}
        ),
        
        dcc.RadioItems(
    	    id='radio-item-data-type',
    	    options=[
        	    {'label': 'True', 'value': 'True'},
        	    {'label': 'Engineering', 'value': 'Engineering'},
    	    ],
    	    value='True',
            style={'marginBottom': '1em'}
        ),

        dash_table.DataTable(
            id='table-material-info',
            columns=[{"name": 'PARAMETER', "id": "PARAMETER"}, {"name": 'INFO', "id": "INFO"}],#[],
            data=[],
            style_cell={'textAlign': 'left', 'textOverflow': 'ellipsis'},
            style_table={
            'maxWidth': '450px',
            'overflowX': 'scroll',
            'border': 'thin lightgrey solid',
            'padding': '15px'
            },
            ),    	     
    ], width=3),
    

    dbc.Col([
        # Dropdown to select the constitutive model     
        html.Label('Constitutive model'),
        
        dcc.Dropdown(
            id='dropdown-constitutive-model',
            options=[{'label': i, 'value': i} for i in models],   # dynamically fill in the dropdown menu
            value=models[0],  # Default value
            style={'width': '100%', 'marginBottom': '1em'}
        ),

        html.Button('Fit Data', id='button-fit-data', style={'marginBottom': '1em', 'background-color': sorored, 'color': 'white'}),

        html.Div(id='header-table-param',children=''' '''),

        dash_table.DataTable(
            id='table-param',
            columns=[],
            data=[],
            #style_data={'width': '90%'}  ###NOT WORKING
            ),

        html.Div(id='AIC-model',children=''' '''),
    ], width=2),


    dbc.Col([ 
        dcc.Graph(id='stress-strain-graph'),

        dcc.RangeSlider(
            id='range-slider',
            min=0,
            max=2,
            step=0.01,
            value=[],
            allowCross=False
        ),

        html.Div(id='output-container-range-slider'),
html.Div(className = "footer", children=[
html.A(html.Img(src=app.get_asset_url('logo_SYMME.svg'), width='15%'),href='https://www.univ-smb.fr/symme/en/', className = "logos"),
html.A(html.Img(src=app.get_asset_url('logo_USMB.svg'), width='15%'),href='https://www.univ-smb.fr/en/', className = "logos"),
]),
    ], width=7),



    ]),


    # Hidden div inside the app that stores the intermediate value
    html.Div(id='intermediate-exp-data', style={'display': 'none'}),  
    html.Div(id='intermediate-model-data', style={'display': 'none'}),
    html.Div(id='intermediate-selected-exp-data', style={'display': 'none'}),
])




@app.callback(
        [Output('intermediate-model-data', 'children'),
        Output('table-param', 'data'),
        Output('table-param', 'columns'),
        Output('header-table-param', 'children'),
        Output('AIC-model', 'children')],
        [Input('button-fit-data', 'n_clicks'),
        Input('dropdown-material', 'value'),
        Input('dropdown-constitutive-model', 'value'),
        Input('radio-item-data-type', 'value')],
        [State('intermediate-selected-exp-data', 'children'),
        State('range-slider', 'value')])
def fit_data_on_click_button(n_clicks, material, constitutive_model, data_type, jsonified_selected_exp_data,slider_value):
    model_data = pd.DataFrame({'True Strain' : [], 'True Stress (MPa)' : []})  # Model is computed only for True Strain True Stress Data for now
    table_param_column = []
    table_param_data = []
    header_table_param = ''' '''
    aic_model = ''' '''
    
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] # Determining which Input has fired

    if triggered_id == "button-fit-data" and data_type == 'True':
        if n_clicks is not None:
    	    selected_exp_data = pd.read_json(jsonified_selected_exp_data)
    	    df_model_param, model_data, aic = optimization(constitutive_model, selected_exp_data)     # Check here IF THERE IS A BUG  !!!!
    	    
    	    df_model_param = df_model_param.round(4) # To send to dash_table.DataTable 

    	    table_param_data = df_model_param.to_dict('records')
    	    table_param_column = [{"name": i, "id": i} for i in df_model_param.columns]
    	    header_table_param = constitutive_model + " parameters : " + '\n' + '(on ε true range {})'.format(slider_value)
    	    aic_model = "AIC : " + np.array2string(aic.round(1))

    return model_data.to_json(), table_param_data, table_param_column, header_table_param, aic_model


@app.callback(
    Output('output-container-range-slider', 'children'),
    [Input('range-slider', 'value')])
def update_output(slider_value):
    return 'Selected strain ε: {}'.format(slider_value)


@app.callback(
    [Output('intermediate-exp-data', 'children'),
    Output('table-material-info', 'data'),
    Output('range-slider', 'max'),
    Output('range-slider', 'value')],
    [Input('dropdown-material', 'value'),
    Input('radio-item-data-type', 'value')])
def update_data(material,data_type):
    file = github_raw_url+material.replace(" ", "%20")+'.csv' # Replace space by %20 for html url 
    header = pd.read_csv(file, delimiter = ';', usecols = ["PARAMETER", "INFO"]).head(15).to_dict('records')
    data = pd.read_csv(file, delimiter = ';',skiprows=18, names = ['Time (s)','True Strain','True Stress (MPa)','Engineering Strain','Engineering Stress (MPa)']) # the column headers are on line 16 from the top of the file     
    range_slider_max = data[data_type+' Strain'].iloc[-1]
    range_slider_value = [0,range_slider_max]
    return data.to_json(), header, range_slider_max, range_slider_value # or, more generally, json.dumps(cleaned_df)


@app.callback(
    [Output('stress-strain-graph', 'figure'),
    Output('intermediate-selected-exp-data', 'children')],
    [Input('dropdown-material', 'value'),
    Input('range-slider', 'value'),
    Input('dropdown-constitutive-model', 'value'),
    Input('intermediate-model-data', 'children')],
    [State('intermediate-exp-data', 'children'),
    State('radio-item-data-type', 'value')]
    )
def update_figure(material,slider_range,constitutive_model,jsonified_model_data,jsonified_exp_data,data_type):
    exp_data = pd.read_json(jsonified_exp_data)
    idx_low= pd.Index(exp_data['True Strain']).get_loc(slider_range[0],method='nearest')   # Find the index of the nearest strain value selected with the slider
    idx_high = pd.Index(exp_data['True Strain']).get_loc(slider_range[1],method='nearest') # Find the index of the nearest strain value selected with the slider
    selected_exp_data = exp_data[idx_low:idx_high] # Trimm the data to the selected range

    model_data = pd.read_json(jsonified_model_data)
    if data_type == 'True':
    	selected_model_data = model_data[idx_low:idx_high] # Trimm the data to the selected range
    else:
        selected_model_data = pd.DataFrame({'Engineering Strain' : [], 'Engineering Stress (MPa)' : []})

    trace_exp_data = dict(x = selected_exp_data[data_type+' Strain'].values,
                y = selected_exp_data[data_type+' Stress (MPa)'].values,
                mode='markers', #'lines+markers'
                opacity=1,
                marker=dict(size=8, color=soroblack),
                name=material+" exp data")
    
    trace_model_data = dict(x = selected_model_data[data_type+' Strain'].values,
            y = selected_model_data[data_type+' Stress (MPa)'].values,
            mode='lines', #'lines+markers'
            line={'color' : sorored},
            opacity=1,
            name=constitutive_model+" model") 

    figure={
        'data': [trace_exp_data,trace_model_data],
        'layout': dict(
            xaxis={'title': data_type + ' Strain ' + u"\u025B"},            # U025B unicode for greek letter epsilon
            yaxis={'title': data_type + ' Stress ' + u"\u03C3" + ' (MPa)'}, # U03C3 unicode for greek letter sigma
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            hovermode='closest',
            legend={'x': 0.05, 'y': 1},                
            showlegend=True,
        )
    }
    return figure, selected_exp_data.to_json()


if __name__ == '__main__':
    app.run_server(debug=True)


