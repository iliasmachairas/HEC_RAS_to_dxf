# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 23:01:58 2022

@author: ilias
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import h5py
import numpy as np

from shapely.geometry import Point
import geopandas as gpd
import shapely
import ezdxf
# import zipfile
import matplotlib.pyplot as plt

# Functions for rotation

###################################################
def a_grad_comp(Dx, Dy):
    a_rad = np.arctan(Dx/Dy)
    a_grad = a_rad* 200 / np.pi
    return a_grad

def azimuth(x0,y0,x1,y1):
    """This function is used to determine azimuth between two points
    - it takes 4 arguements(x of 1st point, y of 1st point, x of 2nd point, y of 2nd point)
    - it returns azimuth in grad
    """
    Dx = x1 - x0
    Dy = y1 - y0
    
    # Check conditions
    if (Dy == 0 and Dx == 0):
        print('You gave two ientical points - Please fix it')
    elif (Dy == 0 and Dx > 0):
         a_final = 100
    elif (Dy == 0 and Dx < 0):
        a_final = 300
    elif (Dx == 0 and Dy > 0):   
        a_final = 0
    elif (Dx == 0 and Dy < 0):
        a_final = 200
    elif (Dx> 0 and Dy > 0):
        a_final = a_grad_comp(Dx, Dy)
    elif (Dx> 0 and Dy < 0):
        a_final = a_grad_comp(Dx, Dy) + 200
    elif (Dx< 0 and Dy < 0):  
        a_final = a_grad_comp(Dx, Dy) + 200
    elif (Dx< 0 and Dy > 0):
        a_final = a_grad_comp(Dx, Dy) + 400

    return a_final

def azimuth_mod(x0,y0,x1,y1):
    a = azimuth(x0,y0,x1,y1)
    azimuth_deg = a * 360 / 400
    azimuth_deg_cor = azimuth_deg - 90
    if azimuth_deg_cor <0:
        azimuth_deg_cor = azimuth_deg_cor+360
    azimuth_deg_cor = 360 -azimuth_deg_cor
    return azimuth_deg_cor

#######################################################

st.set_page_config(layout = "wide")
st.markdown("""# HEC-RAS - Velocity - dxf""")
col2, space2, col3 = st.columns((10,1,10))

with col2:
    
    number_1 = st.number_input('Insert the upper limit of \'low values\' class', value=2.0, min_value=0.1,
                               max_value=2.9)
    number_2 = st.number_input('Insert the lower limit of \'high values\' class', value=3.0, min_value=number_1,
                               max_value=15.0)
                        
    # Read data
    upload_data = st.file_uploader('Select the file', type=['hdf'])
    if upload_data is not None:
        
        ##############
        #######      River Centerline
        ##############
        
        f = h5py.File(upload_data, "r")
        polyline = f['Geometry']['River Centerlines']['Polyline Points'][:]
        polyline_df = pd.DataFrame(polyline, columns=["x", "y"])
        
        
        geometry = [Point(xy) for xy in zip(polyline_df.x, polyline_df.y)]
        polyline_geo_df = gpd.GeoDataFrame(polyline_df, geometry=geometry)
               
        lines_array = []
        for i in range(1,polyline_geo_df.shape[0]):
            point_line = shapely.geometry.LineString([polyline_geo_df.geometry[i-1], polyline_geo_df.geometry[i]])
            line_gdf = gpd.GeoDataFrame({'geometry':[point_line]}, geometry='geometry')
            lines_array.append(line_gdf)
        
        pl_geometry_concat = pd.concat(lines_array,)
        pl_geometry_concat_dis = pl_geometry_concat.dissolve()
        
        ##############
        #######      Cross sections
        ###############
        
        cross_sections_points = f['Geometry']['Cross Sections']['Polyline Points'][:]
        
        cross_sections_points_df = pd.DataFrame(cross_sections_points, columns=['x','y'])
        
        geometry = [Point(xy) for xy in zip(cross_sections_points[:,0], cross_sections_points[:,1])]
        cross_sections_geo_df = gpd.GeoDataFrame(cross_sections_points_df, geometry=geometry)
        
        lines_array = []
        for i in range(0,cross_sections_geo_df.shape[0],2):
            point_line = shapely.geometry.LineString([cross_sections_geo_df.geometry[i], cross_sections_geo_df.geometry[i+1]])
            line_gdf = gpd.GeoDataFrame({'geometry':[point_line]}, geometry='geometry')
            lines_array.append(line_gdf)
        
        cross_sections_geo_df_concat = pd.concat(lines_array,)
        
        ### Intersection
        
        points_intersect = pl_geometry_concat.unary_union.intersection(cross_sections_geo_df_concat.unary_union)
        points_intersect_x = np.array([pt.x for pt in points_intersect])
        points_intersect_y = np.array([pt.y for pt in points_intersect])
        points_merge = np.vstack((points_intersect_x, points_intersect_y)).T
        points_intersect_coords = pd.DataFrame(points_merge, columns=['x','y'])
        
        upload_data_2 = st.file_uploader('Select the file', type=['txt'])
        if upload_data_2 is not None:   
                    output = pd.read_fwf(upload_data_2, skiprows=9)
                    output.drop(labels=0, axis=0, inplace=True)
                    
                    output_1 = output.iloc[:,1]
                    output_2 = output['Vel Chnl']
                    
                    output_clip = pd.concat((output_1, output_2),axis=1)
                    output_clip.columns = ['River Station', 'Vel Chnl']
                    
                    output_clip.reset_index(inplace=True)
                    output_clip_final = output_clip[['River Station', 'Vel Chnl']]
                    
                    merge_table = pd.concat([points_intersect_coords, output_clip_final], axis=1)
                    
                    # Creating lines geo-dataframe
                    geometry = [Point(xy) for xy in zip(merge_table.x, merge_table.y)]
                    geo_df = gpd.GeoDataFrame(merge_table, geometry=geometry)
                    
                    lines_array = []
                    for i in range(1,geo_df.shape[0]):
                        point_line = shapely.geometry.LineString([geo_df.geometry[i-1], geo_df.geometry[i]])
                        line_gdf = gpd.GeoDataFrame({'geometry':[point_line]}, geometry='geometry')
                        lines_array.append(line_gdf)
                    
                    geometry_concat = pd.concat(lines_array,)
                    
                    geometry_concat['station'] = geo_df['River Station'].values[1:]
                    geometry_concat['velocity'] = geo_df['Vel Chnl'].values[1:]
                    geometry_concat['mid_x'] = geometry_concat.centroid.x
                    geometry_concat['mid_y'] = geometry_concat.centroid.y
                    
                    
                    geometry_concat['start_x'] = points_intersect_coords[:-1].x.values
                    geometry_concat['start_y'] = points_intersect_coords[:-1].y.values
                    
                    geometry_concat['end_x'] = points_intersect_coords[1:].x.values
                    geometry_concat['end_y'] = points_intersect_coords[1:].y.values
                    
                    
                    geo_df['thousand_count'] = divmod(geo_df['River Station'],1000)[0]
                    geo_df['thousand_count'] = geo_df['thousand_count'].astype('int').astype('str')
                    geo_df['thousand_2nd_part'] = divmod(geo_df['River Station'],1000)[1].round(1)
                    geo_df['thousand_2nd_part'] = geo_df['thousand_2nd_part'].astype('str')
                    geo_df['thousand_2nd_part'] = geo_df['thousand_2nd_part'].str.zfill(5)
                    geo_df['text_pos'] = geo_df['thousand_count'] + '+'+ geo_df['thousand_2nd_part']
                    
                    
                    # Rotation function
                    geometry_concat['azimuth'] = geometry_concat.apply(lambda row : azimuth_mod(row['start_x'],row['start_y'],
                      row['end_x'], row['end_y']), axis = 1)
                    
                    geometry_concat.velocity = geometry_concat.velocity.astype('float')
                    
                    # splitting dataset
                    geometry_concat_high = geometry_concat[geometry_concat.velocity>number_2]
                    geometry_concat_medium = geometry_concat[(geometry_concat.velocity>number_1) & (geometry_concat.velocity<=number_2)]
                    geometry_concat_low = geometry_concat[geometry_concat.velocity<=number_1]
                    
                    
                    
                    
                    ##################
                    ## Diagram
                    ###################
                    

                    
                               
                    with col3:
                        st.markdown("""# Pick colors""")
                        
                        color_high = st.color_picker('low velocity', value='#FF0000')
                        color_medium = st.color_picker('Medium velocity', value='#FFA500')
                        color_low = st.color_picker('Low velocity', value='#008000')
                        
                        
                        # color_high = '#FF0000'
                        # color_medium = '#FFA500'
                        # color_low = '#008000'

                        fig, ax = plt.subplots(figsize=(8,6))
                        
                        if geometry_concat_high.empty == False:
                            geometry_concat_high.plot(color=color_high, ax=ax, label='high')
                            
                        if geometry_concat_medium.empty == False:
                            geometry_concat_medium.plot(color=color_medium, ax=ax, label='medium')
                            
                        if geometry_concat_low.empty == False:
                            geometry_concat_low.plot(color=color_low, ax=ax, label='low')    
                        
                        plt.legend(loc=2)
                        plt.xticks([])
                        plt.yticks([])
                        plt.title("Velocity (Plan View)");
                        plt.savefig('Velocity_diagram.png')
                        
                        #####################
                        # Export to dxf
                        #####################
                        
                        
                        
                        st.markdown("""# Sizes""")
                            # select time step
                        min_radius, mean_radius, max_radius = 5, 50, 100
                        sel_radius = st.slider("Select radius size", min_value=min_radius,
                        max_value=max_radius, value=mean_radius, step=5, key='radius_slider')
                        
                        min_text, mean_text, max_text = 5, 50, 100
                        sel_text = st.slider("Select text height", min_value=min_text,
                        max_value=max_text, value=mean_text, step=5, key='text_slider')                        
                        
                        doc = ezdxf.new(setup=True)
                        msp = doc.modelspace()
                        
                        user_text_height = 100.0
                        user_radius = 100.0
                        
                        low_velocity = doc.layers.add("Low_velocity")
                        low_velocity.color = 3
                        
                        medium_velocity = doc.layers.add("Medium_velocity")
                        medium_velocity.color = 2
                        
                        high_velocity = doc.layers.add("High_velocity")
                        high_velocity.color = 1
                        
                        river_position = doc.layers.add("River_position")
                        river_position.color = 6
                        
                        # split dataframe in three df
                        # df_low 0-3
                        # df_mid 3-7 
                        # df_high 7-
                        
                        # high velocity
                        if geometry_concat_high.empty == False:
                            for i in range(geometry_concat_high.shape[0]):
                                msp.add_lwpolyline(geometry_concat_high.iloc[i].geometry.coords,
                                                   dxfattribs={"layer": "High_velocity"})
                                
                            for i in range(geometry_concat_high.shape[0]):
                                x_h = geometry_concat_high.iloc[i].mid_x
                                y_h = geometry_concat_high.iloc[i].mid_y
                                azim_h = geometry_concat_high.iloc[i].azimuth
                                msp.add_text(str(geometry_concat_high.iloc[i].velocity), dxfattribs={'style': 'LiberationSerif',
                                            'height': user_text_height, 'rotation':azim_h, "layer": "High_velocity"}).set_pos((x_h+1, y_h+1), align='MIDDLE_CENTER')
                                
                        if geometry_concat_medium.empty == False:
                            for i in range(geometry_concat_medium.shape[0]):
                                msp.add_lwpolyline(geometry_concat_medium.iloc[i].geometry.coords,
                                                   dxfattribs={"layer": "Medium_velocity"})
                                
                            for i in range(geometry_concat_medium.shape[0]):
                                x_m = geometry_concat_medium.iloc[i].mid_x
                                y_m = geometry_concat_medium.iloc[i].mid_y
                                azim_m = geometry_concat_medium.iloc[i].azimuth
                                msp.add_text(str(geometry_concat_medium.iloc[i].velocity), dxfattribs={'style': 'LiberationSerif',
                                            'height': user_text_height, 'rotation':azim_m, "layer": "Medium_velocity"}).set_pos((x_m+1, y_m+1), align='MIDDLE_CENTER')
                        
                        if geometry_concat_low.empty == False:
                            for i in range(geometry_concat_low.shape[0]):
                                msp.add_lwpolyline(geometry_concat_low.iloc[i].geometry.coords,
                                                   dxfattribs={"layer": "Low_velocity"}) 
                            
                            for i in range(geometry_concat_low.shape[0]):
                                x_l = geometry_concat_low.iloc[i].mid_x
                                y_l = geometry_concat_low.iloc[i].mid_y
                                azim_l = geometry_concat_low.iloc[i].azimuth
                                msp.add_text(str(geometry_concat_low.iloc[i].velocity), dxfattribs={'style': 'LiberationSerif',
                                            'height': user_text_height, 'rotation':azim_l, "layer": "Low_velocity"}).set_pos((x_l+1, y_l+1), align='MIDDLE_CENTER')
                        
                        
                        for i in range(geo_df.shape[0]):
                            msp.add_circle(center=(geo_df.x.iloc[i],geo_df.y.iloc[i]), radius=user_radius, dxfattribs={"layer": "river_position"})
                            msp.add_text(str(geo_df.iloc[i].text_pos), dxfattribs={'style': 'LiberationSerif',
                                    'height': user_text_height, "layer": "river_position"}).set_pos((geo_df.x.iloc[i]+1.3*user_radius+4.5,
                                                                                                     geo_df.y.iloc[i]+1.3*user_radius+1.5), align='MIDDLE_CENTER')    
                                               
                        doc.saveas("Velocity_output.dxf")
                        
                        
                        st.markdown("""# Download""")
                        
                        # Zip folder
                        zipObj = zipfile.ZipFile('Output_data.zip', 'w')
                        zipObj.write("Velocity_output.dxf")
                        zipObj.close()
                        
                        btn_image = st.download_button(
                          label="Download Image of Velocities",
                          data=open('Velocity_diagram.png', 'rb').read(),
                          file_name="Velocity_diagram.png",
                          mime="image/jpeg",
                          )
                                                
                        with open("Output_data.zip", "rb") as fp:
                            btn = st.download_button(
                                label="Download dxf",
                                data=fp,
                                file_name="Output_Data.zip",
                                mime="application/zip"
                        )
                            
                        ## 2nd download 
                        # Downlaod shapefile
    #st.download_button(           
                    
                    
                    