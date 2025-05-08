import os
import numpy as np
import pandas as pd
from bokeh.plotting import gmap,ColumnDataSource,figure,output_file,show,save
from bokeh.layouts import column,row,layout
from bokeh.tile_providers import get_provider,STAMEN_TERRAIN, CARTODBPOSITRON
from bokeh.models import Toggle,Spinner,HoverTool,LabelSet,ColumnDataSource,\
                         GMapOptions,Arrow,OpenHead,NormalHead,CustomJS, Slider

def wgs84_web_mercator_point(lon,lat):
    """Function converting single point coordinates in GCS WGS84 model (elliptic coordinates) to mercator"""
    k = 6378137
    x = lon * (k * np.pi/180.0)
    y = np.log(np.tan((90 + lat) * np.pi/360.0)) * k
    return x,y

def wgs84_to_web_mercator(df, lon="longitude", lat="latitude", new_lon="x", new_lat="y"):
    """Vectorized function to convert Data Frame rows with coordinates in GCS WGS84 model (elliptic) to mercator"""
    k = 6378137
    df[new_lon] = df[lon] * (k * np.pi/180.0)
    df[new_lat] = np.log(np.tan((90 + df[lat]) * np.pi/360.0)) * k
    return df


def vis(sim, run=None, to_file=False):
    """ Function to visualize particular run in the given simulation.

    Args:
        sim(Simulation): instance of Simulation class
        run(int): number of a run to visualize
        to_file(bool): If True results will be saved to a file, otherwise shown in the browser.

    """

    # Creating new dict which is basically graph's master_dict, but with longitude and latitude tuple decupled into
    # separate values of another dict. It is done to prepare graph data to be stored in pandas data frame.
    global wgs84_to_web_mercator
    new_dict = {}
    for key, value in sim.graph.master_dict.items():
        new_dict[key] = {}
        for key2, value2 in value.items():
            if key2 == 'coordinates':
                new_dict[key]['latitude'] = value2[0]
                new_dict[key]['longitude'] = value2[1]
            else:
                new_dict[key][key2] = value2

    # creating the data frame
    nodes_df = pd.DataFrame(new_dict).transpose()

    # Addind information on adjacent nodes to the Data Frame
    id_end_node, lats, lons = list(), list(), list()
    for row in nodes_df.iterrows():
        end_node = sim.graph.adjacency[row[0]][0]
        if end_node in sim.graph.master_dict:
            id_end_node.append(end_node)
            lat, lon = nodes_df.loc[(end_node), ['latitude', 'longitude']]
            lats.append(lat)
            lons.append(lon)
        else:
            lats.append(np.nan), lons.append(np.nan), id_end_node.append(np.nan)
    nodes_df['end_node'], nodes_df['end_node_lats'], nodes_df['end_node_lons'] = id_end_node, lats, lons

    def agents_positions(agents_dict):
        """Function responsible for creating a data frame with agents positions from Simulation's agents_dict for
        particular evolution"""
        longitude = list()
        latitude = list()
        ids = list()
        altitude = list()
        speed = list()
        ahead = list()
        consecutive = list()
        adjusted_vox_fast = list()
        adjusted_vox_slow = list()
        in_holding = list()
        for agent_id, agent in agents_dict.items():
            if agent.ahead_agent[0]:
                ahead.append(agent.ahead_agent[0].id)
            else:
                ahead.append('|------|')
            if agent.consecutive_agent:
                consecutive.append(agent.consecutive_agent[0].id)
            else:
                consecutive.append(None)
            if agent.adjusted_vox == 'Fast':
                adjusted_vox_fast.append('Fast')
                adjusted_vox_slow.append(None)
            elif agent.adjusted_vox == 'Slow':
                adjusted_vox_fast.append(None)
                adjusted_vox_slow.append('Slow')
            else:
                adjusted_vox_fast.append(None)
                adjusted_vox_slow.append(None)
            ids.append(agent_id)
            altitude.append(f"{round(agent.altitude)} ft")
            speed.append(f"{round(agent.speed)}kt")
            position = sim.locate_agent(agent)
            latitude.append(position[0])
            longitude.append(position[1])
            if agent.in_holding:
                in_holding.append('Held')
            else:
                in_holding.append(None)
        agents_frame = pd.DataFrame({'agent': ids, 'longitude': longitude, 'latitude': latitude,
                                     'speed': speed,
                                     'altitude': altitude,
                                     'ahead': ahead,
                                     'consecutive': consecutive,
                                     'in_holding': in_holding,
                                     'adjusted_vox_fast': adjusted_vox_fast,
                                     'adjusted_vox_slow': adjusted_vox_slow,
                                     })

        # print(agents_frame['speed'].isnull().values.any())
        return agents_frame

    # applying agents_positions to every saved evolution state of the simulation
    df_list = list()
    if run is not None:
        for key, val in sim.runs[run].history_save.items():
            df_list.append(agents_positions(val))
    # else:
    #     for key, val in sim.history_save.items():
    #         df_list.append(agents_positions(val))

    #mercator projection boundaries:
    lon_min, lat_min = 19.274, 51.238 #TODO zrobić argumentem w funkcji,żeby była uniwersalna a nie tylko dla EPWA
    lon_max, lat_max = 22.500, 52.780

    #Switching to mercator coordinates
    for el in df_list:
        wgs84_to_web_mercator(el)
    xy_min = wgs84_web_mercator_point(lon_min, lat_min)
    xy_max = wgs84_web_mercator_point(lon_max, lat_max)
    wgs84_to_web_mercator(nodes_df)
    wgs84_to_web_mercator(nodes_df, "end_node_lons", "end_node_lats", "x_ends", "y_ends")

    #Changing data format for to bokeh's CDS
    df_sources = list()
    for df in df_list:
        df_sources.append(ColumnDataSource(df))
    agents_source = df_sources[0]
    node_source = ColumnDataSource(nodes_df)
    hovering_circle_source = ColumnDataSource(dict(x=[], y=[]))


    #Setting up visualisation framework
    #Background and figure setting
    x_range,y_range=([xy_min[0],xy_max[0]], [xy_min[1],xy_max[1]])
    p = figure(x_range=x_range, y_range=y_range, x_axis_type='mercator', y_axis_type='mercator',
              sizing_mode='scale_width', height=300)
    p.add_tile(CARTODBPOSITRON, level='image')

    #Adding STAR graph representation to the plot
    nodes = p.circle('x', 'y', source=node_source, fill_color='red', hover_color='yellow', size=10, fill_alpha=0.8,
                     line_width=0)
    p.add_layout(Arrow(end=NormalHead(size=0), line_alpha=0.3, line_width=5, line_cap="round", source=node_source,
                       x_start='x', y_start='y',
                       x_end='x_ends', y_end='y_ends'))
    nodes_hover = HoverTool(tooltips=[('','@index')], renderers=[nodes])

    #Setting up framework for adding agents to the plot
    agents_circles = p.circle(x='x',y='y',source=agents_source, fill_color='blue',hover_color='green', hit_dilation=3,
                              size=10,fill_alpha=0.8,line_width=0)
    #Hovering actions (circle radius 5nm to appear when pointing on particular agent) and labeling
    separation_radius = p.circle('x', 'y', source=hovering_circle_source, fill_color=None, line_color='red',
                                 line_width=3, line_dash='dashed', radius=14925)
    labels1 = LabelSet(x='x', y='y', text='agent', level='glyph',
                      x_offset=5, y_offset=5, source=agents_source, text_color='black', text_font_style = 'bold',
                      text_font = {'value': 'arial'}, background_fill_color = 'grey', background_fill_alpha = 0.7, text_font_size="10pt")

    labels2 = LabelSet(x='x', y='y', text='altitude', level='glyph',
                      x_offset=5, y_offset=-10, source=agents_source, text_color='black', text_font_style = 'normal',
                      text_font = {'value': 'arial'}, background_fill_color = 'grey', background_fill_alpha = 0.7, text_font_size="10pt")

    labels3 = LabelSet(x='x', y='y', text='speed', level='glyph',
                      x_offset=5, y_offset=-25, source=agents_source, text_color='black', text_font_style = 'normal',
                      text_font = {'value': 'arial'}, background_fill_color = 'grey', background_fill_alpha = 0.7, text_font_size="10pt")

    labels4 = LabelSet(x='x', y='y', text='ahead', level='glyph',
                      x_offset=5, y_offset=-40, source=agents_source, text_color='black', text_font_style = 'normal',
                      text_font = {'value': 'arial'}, background_fill_color = 'grey', background_fill_alpha = 0.7, text_font_size="10pt")

    labels5 = LabelSet(x='x', y='y', text='adjusted_vox_slow', level='glyph',
                      x_offset=5, y_offset=-70, source=agents_source, text_color='red', text_font_style = 'normal',
                      text_font = {'value': 'arial'}, background_fill_color = 'grey', background_fill_alpha = 0.7, text_font_size="10pt")

    labels6 = LabelSet(x='x', y='y', text='adjusted_vox_fast', level='glyph',
                       x_offset=5, y_offset=-70, source=agents_source, text_color='green', text_font_style='normal',
                       text_font={'value': 'arial'}, background_fill_color='grey', background_fill_alpha=0.7,
                       text_font_size="10pt")

    labels7 = LabelSet(x='x', y='y', text='consecutive', level='glyph',
                       x_offset=5, y_offset=-55, source=agents_source, text_color='black', text_font_style='normal',
                       text_font={'value': 'arial'}, background_fill_color='grey', background_fill_alpha=0.7,
                       text_font_size="10pt")

    labels8 = LabelSet(x='x', y='y', text='in_holding', level='glyph',
                       x_offset=5, y_offset=-70, source=agents_source, text_color='red', text_font_style='normal',
                       text_font={'value': 'arial'}, background_fill_color='grey', background_fill_alpha=0.7,
                       text_font_size="10pt")

    p.add_tools(nodes_hover)
    p.add_layout(labels3)
    p.add_layout(labels2)
    p.add_layout(labels1)
    p.add_layout(labels4)
    p.add_layout(labels5)
    p.add_layout(labels6)
    p.add_layout(labels7)
    p.add_layout(labels8)


    #Widgets adding
    slider = Slider(start=0, end=sim.evolutions_number, value=0, step=1, title="Evolution", min_width=2000, max_width=2000)

    spinner = Spinner(title="Evolution", low=0, high=sim.evolutions_number, step=1, width=80, value=slider.value)

    agents_callback_js = CustomJS(args=dict(source=agents_source, sources=df_sources, t=slider, t2=spinner),
                        code="""
        var data = source.data;
        var time = t.value;
        var t2 = t;
        var dfl = sources[time].data
        console.log(sources.data)
        data['index'] = []
        data['latitude']=[];
        data['longitude']=[];
        data['agent'] = [];
        data['ahead'] = [];
        data['consecutive'] = [];
        data['adjusted_vox_fast'] = [];
        data['adjusted_vox_slow'] = [];
        data['in_holding']=[];
        data['altitude'] = [];
        data['speed'] = [];
        data['x'] = [];
        data['y'] = [];
        for (var i = 0; i < dfl.x.length; i++) {
            data['agent'].push(dfl['agent'][i]);
            data['x'].push(dfl['x'][i]);
            data['y'].push(dfl['y'][i]);
            data['speed'].push(dfl['speed'][i]);
            data['altitude'].push(dfl['altitude'][i]);
            data['adjusted_vox_fast'].push(dfl['adjusted_vox_fast'][i]);
            data['adjusted_vox_slow'].push(dfl['adjusted_vox_slow'][i]);
            data['in_holding'].push(dfl['in_holding'][i]);
            data['ahead'].push(dfl['ahead'][i]);
            data['consecutive'].push(dfl['consecutive'][i]);
            data['longitude'].push(dfl['longitude'][i]);
            data['latitude'].push(dfl['latitude'][i]);
            data['index'].push(dfl['index'][i]);
    
        }
        source.change.emit();
        """)

    toggl_js = CustomJS(args=dict(slider=slider),code="""
    // A little lengthy but it works for me, for this problem, in this version.
        var check_and_iterate = function(){
            var slider_val = slider.value;
            var toggle_val = cb_obj.active;
            if(toggle_val == false) {
                cb_obj.label = '► Play';
                clearInterval(looop);
                } 
            else if(slider_val == slider.end - 1) {
                cb_obj.label = '► Play';
                slider.value = 0;
                cb_obj.active = false;
                clearInterval(looop);
                }
            else if(slider_val !== slider.end - 1){
                slider.value = slider.value + 1;
                }
            else {
            clearInterval(looop);
                }
        }
        if(cb_obj.active == false){
            cb_obj.label = '► Play';
            clearInterval(looop);
        }
        else {
            cb_obj.label = '❚❚ Pause';
            var looop = setInterval(check_and_iterate, 200);
        };
    """)

    hover_circle_callback_js = CustomJS(args=dict(agents_circles=agents_circles, radius=separation_radius.data_source),
                                        code="""
        var ind = cb_data.index.indices;
        var data = {"x" : [], "y": []};
        if(ind != undefined){
            for (var i=0; i<ind.length; i++){
                console.log(ind[i])
                console.log(agents_circles.data_source.data)
                data["x"].push(agents_circles.data_source.data.x[ind[i]])
                data["y"].push(agents_circles.data_source.data.y[ind[i]])
            }
            radius.data = data
            console.log(data)
        }
    """)

    p.add_tools(HoverTool(tooltips=None, callback=hover_circle_callback_js, renderers=[separation_radius,
                                                                                       agents_circles]))

    slider.js_on_change('value', agents_callback_js)
    spinner.js_link("value", slider, 'value')
    slider.js_link('value', spinner, 'value')
    toggl = Toggle(label='► Play', active=False)
    toggl.js_on_change('active', toggl_js)

    layout = column(
        toggl,
        spinner,
        p,
        slider,
    )
    if to_file == True:
        path = input('Enter directory where visualisation should be save:')
        name = input('Enter how file should be named:')
        output_file(os.path.join(path, name), title="slider.py example")
        save(layout)
    else:
        show(layout)

def positions_over_time_export(sim, run=None, to_file=False):
    """Returns a dict with all information on agents positions over the simulation duration
    It's purpose is data comparison (with other runs or potentially real records)
    """
    # Creating new dict which is basically graph's master_dict, but with longitude and latitude tuple decupled into
    # separate values of another dict. It is done to prepare graph data to be stored in pandas data frame.
    global wgs84_to_web_mercator
    new_dict = {}
    for key, value in sim.graph.master_dict.items():
        new_dict[key] = {}
        for key2, value2 in value.items():
            if key2 == 'coordinates':
                new_dict[key]['latitude'] = value2[0]
                new_dict[key]['longitude'] = value2[1]
            else:
                new_dict[key][key2] = value2

    # creating the data frame
    nodes_df = pd.DataFrame(new_dict).transpose()

    # Addind information on adjacent nodes to the Data Frame
    id_end_node, lats, lons = list(), list(), list()
    for row in nodes_df.iterrows():
        end_node = sim.graph.adjacency[row[0]][0]
        if end_node in sim.graph.master_dict:
            id_end_node.append(end_node)
            lat, lon = nodes_df.loc[(end_node), ['latitude', 'longitude']]
            lats.append(lat)
            lons.append(lon)
        else:
            lats.append(np.nan), lons.append(np.nan), id_end_node.append(np.nan)
    nodes_df['end_node'], nodes_df['end_node_lats'], nodes_df['end_node_lons'] = id_end_node, lats, lons

    def agents_positions(agents_dict):
        """Function responsible for creating a data frame with agents positions from Simulation's agents_dict for
        particular evolution"""
        longitude = list()
        latitude = list()
        ids = list()
        altitude = list()
        speed = list()
        for agent_id, agent in agents_dict.items():
            ids.append(agent_id)
            altitude.append(f"{round(agent.altitude)} ft")
            speed.append(f"{round(agent.speed)}kt")
            position = sim.locate_agent(agent)
            latitude.append(position[0])
            longitude.append(position[1])
        agents_frame = pd.DataFrame({'agent': ids, 'longitude': longitude, 'latitude': latitude,
                                     'speed': speed,
                                     'altitude': altitude,
                                     })

        # print(agents_frame['speed'].isnull().values.any())
        return agents_frame

    # applying agents_positions to every saved evolution state of the simulation
    dict_list = list()
    if run is not None:
        for key, val in sim.runs[run].history_save.items():
            dict_list.append(agents_positions(val).to_dict())

    return dict_list