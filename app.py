# -*- coding: utf-8 -*-
__author__ = "Luc Marechal"
__copyright__ = ""
__credits__ = ["Luc Marechal", "Lukas Lindenroth"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Luc Marechal"
__email__ = "luc.marechal(at)univ-smb(dot)fr"
__status__ = "Prod1"

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
import dash_daq as daq  # for Toggle switch button
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
line_colors = ['#b72367','#ef5454','#f9875a','#1dbc8e','#3e8ccc','#5fd0db','#8bce00','#f7cb00','#b72367','#7ce214','#ef5454','#f9875a','#1dbc8e','#3e8ccc','#5fd0db','#8bce00','#f7cb00','#b72367','#00ACAA']
# Special characters
unicode_epsilon = "\U0000025B"
unicode_sigma = "\U000003C3"

external_stylesheets = ['SoRo_Material_Database.css', dbc.themes.GRID,] #dbc.themes.BOOTSTRAP, 'https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server
app.title = "Soft Robotics Materials Database"



#mathjax = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML'
#app.scripts.append_script({ 'external_url' : mathjax })


def list_files(url):
    """List the files in the directory."""
    database = requests.get(url)
    soup = BeautifulSoup(database.text, 'html.parser')
    csvfiles = soup.find_all(title=re.compile("\.csv$")) # only lists *.csv files

    materials = []
    for filename in csvfiles:
        materials.append(filename.extract().get_text()[:-4])  # -4 to remove the file extension .csv
    return materials


def read_csv_exp_data_files(material_name):
    file = github_raw_url + material_name.replace(" ", "%20") + '.csv' # Replace space by %20 for html url 
    header = pd.read_csv(file, delimiter = ';', usecols = ["PARAMETER", "INFO", "URL"]).head(15)
    data = pd.read_csv(file, delimiter = ';',skiprows=18, names = ['Time (s)','True Strain','True Stress (MPa)','Engineering Strain','Engineering Stress (MPa)']) # the column headers are on line 16 from the top of the file   
    return data, header



#############################################################################
#  OPTIMIZATION
#############################################################################
# cost function to calculate the residuals. The fitting function holds the parameter values.  
def objectiveFun_Callback(parameters, exp_strain, exp_stress, hyperelastic):  
    theo_stress = hyperelastic.ConsitutiveModel(parameters, exp_strain)   
    residuals = theo_stress - exp_stress 
    return residuals

def optimization(model, order, dataframe, data_type):
    # Hyperelastic object
    hyperelastic = Hyperelastic(model, np.array([0]), order, data_type)
    
    exp_strain = dataframe[data_type+' Strain'].values
    exp_stress = dataframe[data_type+' Stress (MPa)'].values
    # The least_squares package calls the Levenberg-Marquandt algorithm.
    # best-fit paramters are kept within optim_result.x
    optim_result = least_squares(objectiveFun_Callback, hyperelastic.initialGuessParam, method ='lm', args=(exp_strain, exp_stress, hyperelastic))
    optim_parameters = optim_result.x

    df_model_param = pd.DataFrame(optim_parameters, index=hyperelastic.param_names, columns=[model]).transpose()
    
    theo_stress = hyperelastic.ConsitutiveModel(optim_parameters, exp_strain)
    data_model = pd.DataFrame({data_type+' Strain': exp_strain, data_type+' Stress (MPa)': theo_stress})
	
    stats = HyperelasticStats(exp_stress, theo_stress, hyperelastic.nbparam)
    aic = stats.aic()

    return df_model_param, data_model, aic


# URL on the Github where the raw files are stored
github_url = 'https://github.com/LucMarechal/Soft-Robotics-Materials-Database/tree/master/Tensile-Tests-Data'
github_raw_url = 'https://raw.githubusercontent.com/LucMarechal/Soft-Robotics-Materials-Database/master/Tensile-Tests-Data/'

# Content of the GitHub repository. Lists all *.csv file name in the database
materials = list_files(github_url)
nb_materials_in_db = len(materials)

# Constitutive models
models = np.array(['Ogden', 'Mooney Rivlin', 'Veronda Westmann', 'Yeoh', 'Neo Hookean', 'Humphrey'])


nav = html.Nav(className = "nav nav-pills", children=[
    dcc.Link("Constitutive Models", href='/constitutive_models'),
    dcc.Link("Materials Comparison", href='/materials_comparison'),
    html.A("Setup & Characterisation", href="https://github.com/LucMarechal/Soft-Robotics-Materials-Database/wiki/Setup-and-Characterisation", target='_blank'),
    html.A("About",href='https://github.com/LucMarechal/Soft-Robotics-Materials-Database/wiki', target='_blank'),
    html.A("GitHub",href='https://github.com/LucMarechal/Soft-Robotics-Materials-Database', target='_blank'),
    ]),



#############################################################################
#  MAIN PAGE
#############################################################################
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),

### NAV BAR ###
    dbc.Row([            
        dbc.Col(html.Img(
                src=app.get_asset_url('logo_SoRo.svg'),#src='https://raw.githubusercontent.com/LucMarechal/data/master/logosoro.png',#'https://user-images.githubusercontent.com/36209435/72664756-f5b6d600-3a01-11ea-88b2-a3f3e46fe9f6.png',
                style={'width': '80%'}
                ), width=3),

        dbc.Col(nav, align="center", width=7),

        dbc.Col(
        #html.Div(className = "footer", children=[
        [html.A(html.Img(src=app.get_asset_url('logo_SYMME.svg'), width='42%'),href='https://www.univ-smb.fr/symme/en/', target='_blank'),
        html.A(html.Img(src=app.get_asset_url('logo_USMB.svg'), width='42%'),href='https://www.univ-smb.fr/en/',style={'padding': '15px'}, target='_blank'),
        ], align="center", width=2),

    ], justify="between"),   

### DISPLAYED PAGE ###
    html.Div(id='page-content')
])



#############################################################################
#  PAGE : CONSTITUTIVE MODELS
#############################################################################
app_constitutive_models_layout = html.Div(children=[

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

        html.A(html.Button('Vendor Material Info', id='button-vendor-material-info', style={'marginTop': '-0.2em', 'marginBottom': '0.5em'}), id='url-material', href='', target='_blank'),
        
        dash_table.DataTable(
            id='table-material-info',
            columns=[{"name": 'PARAMETER', "id": "PARAMETER"}, {"name": 'INFO', "id": "INFO"}],#[],
            data=[],
            style_cell={'textAlign': 'left', 'textOverflow': 'ellipsis'},
            style_table={
            'maxWidth': '500px',
            'overflowX': 'scroll',
            'border': 'thin lightgrey solid',
            'padding': '15px'
            },
            ),    	     
    ], width=3),
    

    dbc.Col([
        # Dropdown to select the constitutive model     
        html.Label('Constitutive model'),      
        
        dcc.Input(
            id='textarea-constitutive-model',
            value='',
            style={'display': 'none'}
        ),

        dcc.Dropdown(
            id='dropdown-constitutive-model',
            options=[{'label': i, 'value': i} for i in models],   # dynamically fill in the dropdown menu
            value=models[0],  # Default value
            style={'width': '100%', 'marginBottom': '1em'}
        ),
        html.Label('Order'),
        dcc.Dropdown(
            id='dropdown-order-model',
            options=[{'label': i, 'value': i} for i in range(1,4)],   # dynamically fill in the dropdown menu
            value=3,  # Default value
            style={'width': '50%', 'marginBottom': '1em'}
        ), 

        # NOT WORKinG : To display the constitutive model mathematical equation 
        # dcc.Textarea(
        #     id='textarea-constitutive-model',
        #     value=r"$y(x)=\frac{1}{{2\pi\sigma^2}}e^{-\frac{x^2}{2\sigma^2}}$",
        #     style={'width': '100%', 'height': 50, 'marginBottom': '1em'},
        # ),

        #html.Div(children=[
        html.Label('Principal Cauchy Stress', style={'marginBottom': '0.5em'}),
        html.Img(id='constitutive-model-formula', src=app.get_asset_url('Ogden.svg'), width='100%', style={'marginBottom': '2em'}),
        #], style={'marginBottom': '1em'}),

        dbc.Row([
            daq.BooleanSwitch(
                id='toggle-fit-mode',
                on=False,
                #label='True / Engineering',
                #labelPosition='bottom',
                color=sorored,
                style={'padding': '10px'} 
            ),
            html.Label('Model selection : Manual/Auto'),
        ]),

        dbc.Row([
            daq.BooleanSwitch(
                id='toggle-data-type',
                on=False,
                #label='True / Engineering',
                #labelPosition='bottom',
                color=sorored,
                style={'padding': '10px'} 
            ),
            html.Label('Data type: True/Engineering'),
        ]),

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

        html.Div(id='output-container-range-slider', style={'margin-bottom': '15px'}),

        html.Div("The data is fitted on the selected strain range. Adjust the range accordingly to your application before clicking on the 'Fit data' button to obtain a better accuracy of the model."),
        
        ], width=7),

    ]),

#############################################################################
#  Hidden div inside the app that stores the intermediate value
#############################################################################
    html.Div(id='intermediate-exp-data', style={'display': 'none'}),  
    html.Div(id='intermediate-model-data', style={'display': 'none'}),
    html.Div(id='intermediate-selected-exp-data', style={'display': 'none'}),
    html.Div(id='intermediate-best-model', style={'display': 'none'}),
]),


#############################################################################
#  PAGE : MATERIALS COMPARISON
#############################################################################
app_materials_comparison_layout = html.Div(children=[

html.H1(children='Materials Comparison',style={'padding': '15px'}),

dbc.Row([
    dbc.Col([
        html.Img(src=app.get_asset_url('logo_soroDB.svg'), style={'width': '30%', 'display': 'inline-block'}), 
        html.Div(nb_materials_in_db, className="w3-badge w3-xlarge w3-sorored w3-padding", style={'display': 'inline-block'}),
        html.Div("materials in the Database", style={'margin-bottom': '100px'}),
        html.Div("Double click on legend to isolate one trace. Show or hide trace by click on the materials name.", style={'margin-bottom': '50px'}),
                
        dbc.Row([
            daq.BooleanSwitch(
                id='toggle-true-eng-data',
                on=False,
                #label='True / Engineering',
                #labelPosition='bottom',
                color=sorored,
                style={'padding': '20px'} 
            ),
            html.Label('Data type: True/Engineering'),
        ]),

        html.Label('Constitutive model'),
        dcc.Dropdown(
            id='dropdown-my-constitutive-model',
            options=[{'label': i, 'value': i} for i in models],   # dynamically fill in the dropdown menu
            value=models[0],  # Default value
            style={'width': '100%', 'marginBottom': '1em'}
        ),
        
        html.Label('Order'),
        dcc.Dropdown(
            id='dropdown-my-order-model',
            options=[{'label': i, 'value': i} for i in range(1,4)],   # dynamically fill in the dropdown menu
            value=3,  # Default value
            style={'width': '50%', 'marginBottom': '1em'}
        ),

        dash_table.DataTable(
            id='my-table-param',
            columns=[],
            data=[],
            editable=True,
            ),

        html.Button('Find material', id='button-find-material', style={'marginBottom': '1em', 'background-color': sorored, 'color': 'white'}),

        html.Div(id='output-test',children=[""]), 
                
        ], width=2),
    
    dbc.Col([html.Div(id='text-loading-data',children=["Loading data..."], style={"color": sorored,"font-weight": 'bold', "text-align": 'center'}), 
    
    dcc.Loading(id="loading-graph", children=[dcc.Graph(id='materials-comparison-graph')], color=sorored,type='cube')], width={"size": 10}),
    ],no_gutters=True,style={'padding': '20px', 'marginBottom': '-1.5em'}),

]),




#############################################################################
#  CALLBACKS
#############################################################################
@app.callback(
        [Output('my-table-param', 'columns'),
        Output('my-table-param', 'data')],
        [Input('dropdown-my-constitutive-model', 'value'),
        Input('dropdown-my-order-model', 'value')])
def update_my_table_param(my_constitutive_model, my_model_order):
    my_hyperelastic = Hyperelastic(my_constitutive_model, np.array([0]), my_model_order)
    my_param_names = my_hyperelastic.param_names
    my_table_param_column = [{"name": i, "id": i} for i in my_param_names]
    my_df = pd.DataFrame(0, index=np.arange(1), columns=my_param_names)
    my_table_data = my_df.to_dict('records')
    return my_table_param_column, my_table_data


@app.callback(
        Output('output-test', 'children'),
        [Input('button-find-material', 'n_clicks'),
        Input('my-table-param', 'data')],
        [State('dropdown-my-constitutive-model', 'value'),
        State('dropdown-my-order-model', 'value')],
        )
def find_material_on_click_button(n_clicks_find_material, my_model_param, my_constitutive_model, my_model_order):
    my_hyperelastic = Hyperelastic(my_constitutive_model, my_model_param, my_model_order)

    print(my_hyperelastic)
    return 'ok'


@app.callback(
        [Output('dropdown-constitutive-model', 'style'),
        Output('textarea-constitutive-model', 'style')],
        [Input('toggle-fit-mode', 'on')])
def show_hide_constitutve_model_dropdown(fit_mode):
    if fit_mode is True: # auto mode
        style_dropdown={'display': 'none'}
        style_textarea={'width': '100%', 'marginBottom': '1em'}
    else:
        style_dropdown={'width': '100%', 'marginBottom': '1em'}
        style_textarea={'display': 'none'}
    return style_dropdown, style_textarea


@app.callback(
        [Output('intermediate-model-data', 'children'),
        Output('table-param', 'data'),
        Output('table-param', 'columns'),
        Output('header-table-param', 'children'),      
        Output('AIC-model', 'children'),
        Output('intermediate-best-model', 'children'),
        Output('textarea-constitutive-model', 'value'),
        Output('constitutive-model-formula', 'src')],  #Test
        [Input('button-fit-data', 'n_clicks'),
        Input('dropdown-material', 'value'),
        Input('dropdown-constitutive-model', 'options'), 
        Input('dropdown-constitutive-model', 'value'),    
        Input('dropdown-order-model', 'value'),
        Input('toggle-data-type', 'on'),
        Input('toggle-fit-mode', 'on')],
        [State('intermediate-selected-exp-data', 'children'),
        State('range-slider', 'value')])
def fit_data_on_click_button(n_clicks_fit_data, material, all_constitutive_models, selected_constitutive_model, order, data_type_toggle, fit_mode_toggle, jsonified_selected_exp_data,slider_value):
    table_param_column = []
    table_param_data = []
    models = []
    header_table_param = ''' '''
    aic_model = ''' '''
    best_model = ''
    url_material = 'https://github.com/LucMarechal/Soft-Robotics-Materials-Database'

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] # Determining which Input has fired

    if data_type_toggle is True:
        data_type = 'Engineering'
    else:
        data_type = 'True'

    model_data = pd.DataFrame({data_type+' Strain' : [], data_type+' Stress (MPa)' : []})


    if fit_mode_toggle is True:
        for dicts in all_constitutive_models:
            models.append(dicts["label"])
    else:
        models.append(selected_constitutive_model)


    if triggered_id == "button-fit-data":
        if n_clicks_fit_data is not None:
            selected_exp_data = pd.read_json(jsonified_selected_exp_data)
            
            # loop to test all constitutive models and find the best one
            for num, constitutive_model in enumerate(models):
                # Optimization algorithm
                df_model_param, model_data_optimized, aic = optimization(constitutive_model, order, selected_exp_data, data_type)
                # initialize best aic to the first tested model
                if num == 0:
                    best_aic = aic
                
                if aic <= best_aic:
                    best_aic = aic
                    df_model_param = df_model_param.round(4) # To send to dash_table.DataTable
                    model_data = model_data_optimized
                    table_param_data = df_model_param.to_dict('records')
                    table_param_column = [{"name": i, "id": i} for i in df_model_param.columns]
                    best_model = constitutive_model
                    header_table_param = best_model + " parameters : " + '\n' + '(on ε ' + data_type + ' data range {})'.format([f"{num:.2f}" for num in slider_value])#slider_value)
                    aic_model = "AIC : " + np.array2string(aic.round(1))

    #update displayed formula
    if fit_mode_toggle is True: # auto mode
        if triggered_id == "button-fit-data":
            formula_image = app.get_asset_url(best_model + '_' + data_type + '.svg')
        else:
            formula_image = app.get_asset_url('blank.svg')
    else:
        formula_image = app.get_asset_url(selected_constitutive_model + '_' + data_type +'.svg')

    return model_data.to_json(), table_param_data, table_param_column, header_table_param, aic_model, best_model, best_model, formula_image



@app.callback(
    Output('output-container-range-slider', 'children'),
    [Input('range-slider', 'value')])
def update_output(slider_value):
    return 'Selected strain ε: {}'.format([f"{num:.2f}" for num in slider_value])


@app.callback(
    [Output('intermediate-exp-data', 'children'),
    Output('table-material-info', 'data'),
    Output('url-material', 'href'),
    Output('range-slider', 'max'),
    Output('range-slider', 'value')],
    [Input('dropdown-material', 'value'),
    Input('toggle-data-type', 'on')])
def update_data(material,data_type_toggle):
    if data_type_toggle is True:
        data_type = 'Engineering'
    else:
        data_type = 'True'
    [data, header] = read_csv_exp_data_files(material)
    range_slider_max = data[data_type+' Strain'].iloc[-1]
    range_slider_value = [0,range_slider_max]

    url_material = header['URL'].dropna().values[0] # Get the URL of the material from the csv file header

    return data.to_json(), header.to_dict('records'), url_material, range_slider_max, range_slider_value # or, more generally, json.dumps(cleaned_df)


@app.callback(
    [Output('stress-strain-graph', 'figure'),
    Output('intermediate-selected-exp-data', 'children')],
    [Input('dropdown-material', 'value'),
    Input('range-slider', 'value'),
    Input('dropdown-constitutive-model', 'value'),
    Input('intermediate-model-data', 'children')],
    [State('intermediate-exp-data', 'children'),
    State('intermediate-best-model', 'children'),
    State('toggle-data-type', 'on')]
    )
def update_figure(material,slider_range,constitutive_model,jsonified_model_data,jsonified_exp_data,best_model,data_type_toggle):
    if data_type_toggle is True:
        data_type = 'Engineering'
    else:
        data_type = 'True'

    exp_data = pd.read_json(jsonified_exp_data)
    idx_low= pd.Index(exp_data[data_type+' Strain']).get_loc(slider_range[0],method='nearest')   # Find the index of the nearest strain value selected with the slider
    idx_high = pd.Index(exp_data[data_type+' Strain']).get_loc(slider_range[1],method='nearest') # Find the index of the nearest strain value selected with the slider
    selected_exp_data = exp_data[idx_low:idx_high] # Trimm the data to the selected range

    model_data = pd.read_json(jsonified_model_data)
    
    selected_model_data = model_data[idx_low:idx_high] # Trimm the data to the selected range

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
            name=best_model+" model") 

    figure={
        'data': [trace_exp_data,trace_model_data],
        'layout': dict(
            xaxis={'title': data_type + ' Strain ' + unicode_epsilon},
            yaxis={'title': data_type + ' Stress ' + unicode_sigma + ' (MPa)'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            hovermode='closest',
            legend={'x': 0.05, 'y': 1},                
            showlegend=True,
        )
    }
    return figure, selected_exp_data.to_json()




@app.callback(
    [Output('materials-comparison-graph', 'figure'),
    Output('text-loading-data', 'children')],
    [Input('toggle-true-eng-data', 'on')],
    )
def update_graph_comparison(data_type_toggle):
    if data_type_toggle is True:
        data_type = 'Engineering'
    else:
        data_type = 'True'
    
    materials = list_files(github_url)
    
    traces_data = []
    for i, material in enumerate(materials):
        [exp_data,header] = read_csv_exp_data_files(material)

        trace_exp_data = dict(x = exp_data[data_type+' Strain'].values,
                    y = exp_data[data_type+' Stress (MPa)'].values,
                    mode='lines',
                    opacity=1,
                    marker=dict(size=8, color=line_colors[i]),
                    name=material)
        traces_data.append(trace_exp_data)

    figure={
        'data': traces_data,
        'layout': dict(
            xaxis={'title': data_type + ' Strain ' + unicode_epsilon},
            yaxis={'title': data_type + ' Stress ' + unicode_sigma + ' (MPa)'},
            autosize=False,
            #width=1000,#500,
            height=550,
            #margin={'l': 40, 'b': 40, 't': 5, 'r': 20},
            margin={'t': -3,},
            hovermode='closest',
            legend={'x':-.2, 'y': 0},     
            showlegend=True,
        )
    }

    loading_text = " "
    return figure, loading_text


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/' or pathname == '/constitutive_models':
        return app_constitutive_models_layout
    elif pathname == '/materials_comparison':
        return app_materials_comparison_layout
    elif pathname == '/setup_characterisation':
        return app_setup_characterisation_layout
    else:
        return '404'



if __name__ == '__main__':
    app.run_server(debug=True)