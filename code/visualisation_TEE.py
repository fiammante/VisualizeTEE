"""
Author: Marc Fiammante
Copyright (c) 2024 Marc Fiammante

This file is part of [Visualization of TEE TransoEsophagial Echography for Institut Arnault Tzanck].

Licensed under the MIT License. You may obtain a copy of the License at

    https://opensource.org/licenses/MIT

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions: 

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import dash
from dash import dcc, html, ctx, no_update
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import imageio.v3 as iio
import base64
import numpy as np
import os
from skimage import measure
from skimage.draw import disk
import plotly.graph_objects as go
from io import BytesIO
from PIL import Image, ImageDraw
import io
import pymediainfo
import trimesh
import math
from skimage import morphology,filters, draw,transform
from skimage.measure import approximate_polygon
from scipy.fft import fft, fftfreq, fftshift
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import pickle
import uuid
import glob
import re
# end imports

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# Create placeholders for the line plots
# Create placeholders for the line plots with no axes visible
section1 = dcc.Graph(id='plot1',figure={'layout': {'xaxis': {'visible': False},'yaxis': {'visible': False}}})
section2 = dcc.Graph(id='plot2',figure={'layout': {'xaxis': {'visible': False},'yaxis': {'visible': False}}})
section3 = dcc.Graph(id='plot3',figure={'layout': {'xaxis': {'visible': False},'yaxis': {'visible': False}}})
# Create the layout with dbc.Row s and dbc.Col s

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Click here to select the TEE"),
                dbc.CardBody([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select File')
                        ]),
                        style={
                            'width': '100%',
                            'height': '20px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        multiple=False  # Allow only one file
                    ),
                ]),
                dbc.CardFooter(
                    [
                        html.Div("*", id="session-id", style={'display': 'none'})
                    ]
                )
            ]),
            width=6
        ),
        dbc.Col(
            html.Div(id='file-name'),
            width=6
        )
    ],style={"height": '120px'}),
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Drag & Drop to define the temporal 3D with a rectangle or a line"),
                dbc.CardBody(
                    dcc.Graph(id='first-frame', style={"height": "400px", "overflow": "hidden"}, config={'modeBarButtonsToAdd': ['drawline','drawrect']},
                                figure={'layout': {'xaxis': {'visible': False},'yaxis': {'visible': False}}})
                ),
                dbc.CardFooter(
                    [
                        html.Pre(id='annotations', children='No annotations yet.', style={'whiteSpace': 'pre-wrap'}),
                    ]
                )
            ]),
            width=6
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("3D Vision 3D, time becomes depth"),
                dbc.CardBody(
                    html.Div(id='output-data-upload')
                )
            ]),
            width=6
        )
    ],style={"height": '520px'}),
    dbc.Row([        
            dbc.Col(dbc.Card([
                dbc.CardHeader("Temporal slice"),
                dbc.CardBody(section1)]), width=4),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Smoothed Contours"),
                dbc.CardBody(section2)]), width=4),       
            dbc.Col(dbc.Card([
                dbc.CardHeader("Smoothed Contours speed cm/sec"),
                dbc.CardBody(section3)]), width=4),   
            ]),
      dbc.Row([
        dbc.Col(html.Hr(), width=12)
    ]),
])

def get_mask(height,width):   
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    # Define the bounding box for the pie slice
    bbox = [0, -height, width, height]
    # Draw the 90° pie slice
    draw.pieslice(bbox, start=45, end=135, fill=255)
    return mask
    
def mask_image(frame):
    (height,width)=frame.shape[:2]
    # Convert the NumPy array to a PIL image
    image = Image.fromarray(frame)
    # Create a mask image with the same size as the original image
    mask = get_mask(height,width)
    
    # Apply the mask to the image
    result = Image.composite(image, Image.new('RGB', (width, height), (0, 0, 0)), mask)
    
    return np.array(result)

def process_video(contents, filename):
  # Decode the base64 data
  decoded_data = base64.b64decode(contents.split(',')[1])
  print("extract frames") 
  # Create an in-memory buffer for the video fps
  with io.BytesIO(decoded_data) as video_buffer:
    # Get media information
    media_info = pymediainfo.MediaInfo.parse(video_buffer)
    # Extract duration and FPS
    fps = float(media_info.tracks[0].frame_rate)
    print("fps",fps)
  # Create an in-memory buffer for the video data
  with io.BytesIO(decoded_data) as video_buffer:
    # Use imageio to read frames from the buffer
    frames = []
    gray_frames = []
    print("get frames")    
    for frame in iio.imiter(video_buffer, plugin="pyav"):
        m_frame=mask_image(frame.astype(np.uint8))
        gray_frame = np.dot(m_frame[...,:3], [0.2989, 0.5870, 0.1140])
        gray_frames.append(gray_frame)
        frames.append(frame)
    return np.array(frames), np.array(gray_frames),fps 

def generate_mesh(frames):
    # Create mesh
    print("generate_mesh")
    npframes=np.array(frames)
    verts, faces, _, _ = measure.marching_cubes(npframes, 92, step_size=1) 
    x, z, y = verts.T
    i, j, k = faces.T 
    print("shapes x,y,z",npframes.shape,"shapes x",np.amax(x),"shapes y",np.amax(y),"shapes z",np.amax(z),(verts.T).shape)
    print("shapes i,j,k",i.shape,j.shape,k.shape)
    mesh = go.Figure(data=[
        go.Mesh3d(
            #  vertices of a cube
            x=x,
            y=y,
            z=np.amax(z)-z,
            colorbar_title='Point<br>relative<br>index 0 to 1',
            colorscale=[[0, 'gold'],
                        [0.2, 'mediumturquoise'],
                        [1, 'magenta']],
            # Intensity of each vertex, which will be interpolated and color-coded
            intensity = np.linspace(0, 1, 1388585, endpoint=True),
            # i, j and k give the vertices of triangles
            i = i,
            j = j,
            k = k,
            name='y',
            showscale=True
        )
    ])
    mesh.update_layout(
        scene=dict(
            xaxis=dict( title='Frame #'),
            yaxis=dict(title='width'),
            zaxis=dict(title='height'),
        )
    )
       
    tripoints=np.stack((x,y, np.amax(z)-z), axis=-1)
    tricells=faces 
    print(tripoints.shape,tricells.shape)
    return mesh,tripoints,tricells
    
def get_upper(all_points):
    # Separate the combined array into x and y arrays
    x_combined = all_points[:, 0]
    y_combined = all_points[:, 1]

    # Find unique x values and their indices
    unique_x, indices = np.unique(x_combined, return_inverse=True)

    # Initialize an array to hold the maximum y values
    max_y = np.zeros_like(unique_x, dtype=y_combined.dtype)

    # Find the maximum y value for each unique x
    for i in range(len(unique_x)):
        max_y[i] = np.max(y_combined[indices == i])

    # Combine the results
    result = np.column_stack((unique_x, max_y))
    print("result.shape",result.shape)
    return result
    
# Function to convert a frame to Base64
def frame_to_base64(frame):
    buffer = BytesIO()
    img = Image.fromarray(frame)
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')
 
# Function to get the sign of consecutive x-values using numpy.diff
def get_x_signs_with_diff(polygon):
    x_values = polygon[:, 0]
    x_diffs = np.diff(x_values)
    signs = np.sign(x_diffs)
    return signs
    
def remove_consecutive_duplicates(points):
    result = [points[0]]   
    for point in points[1:]: 
        if (point != result[-1]).any():
            result.append(point)    
    return np.array(result)
    
# Function to split the polygon at x-direction changes using np.argwhere
def split_polygon_by_x_direction(polygon):
    polygon=remove_consecutive_duplicates(polygon)
    signs = get_x_signs_with_diff(polygon)
    change_indices = np.argwhere(np.diff(signs) != 0).flatten() + 1
    segments = []
    start_idx = 0
    for idx in change_indices:
        segments.append(polygon[start_idx:idx+1])
        start_idx = idx   
    segments.append(polygon[start_idx:])
    return segments

# find intersection of line with axis
def find_intersection(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    w=abs(x2-x1)
    h=abs(y2-y1)
    # Check if the line is vertical
    if x1 == x2:
        return x1,0        
    # Calculate the slope (m) and y-intercept (b) of the line
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    if h>w:
        return (-b / m),0
    else:
        return 0,b
        
def get_polygons(path3D,intersect):
    polygons_3D=[]
    polygons_2D=[]
    for paths in path3D.paths:
        path_vert=[]
        for path in paths:
            entity=path3D.entities[path] 
            nodes = entity.nodes
            lastend=-1
            for n,node in enumerate(nodes):
                start,end=node
                if n==0:
                    first=start
                path_vert.append(path3D.vertices[start])
                if n>0 & lastend!=start:
                    path_vert.append(path3D.vertices[lastend])
                lastend=end
            # close path
            path_vert.append(path3D.vertices[first])
        # Prepare data for Plotly
        vectors_3d=np.array(path_vert)
        polygons_3D.append(vectors_3d) 
        vectors_2d = np.column_stack((vectors_3d[:, 0], np.sqrt((vectors_3d[:, 1]-intersect[1])**2 + (vectors_3d[:, 2]-intersect[2])**2)))
        polygons_2D.append(vectors_2d)      
    return polygons_3D,polygons_2D

def slice_with_inclined_plane(mesh,origin, w,h):
    # Define the plane normal (angle degrees along z-axis)
    cos=w/math.sqrt(w*w+h*h)
    sin=h/math.sqrt(w*w+h*h)
    normal = [0,sin,cos]   
    print("normal",normal)
    print("origin",origin)
    print("bounds",mesh.bounds)
    print("centroid",mesh.centroid)
    # Slice the mesh with the inclined plane
    sliced_meshes = mesh.section(plane_origin=origin, plane_normal=normal)
    print("sliced_meshes",sliced_meshes)
    return sliced_meshes 


def meshsection(tripoints,tricells,segment,pixel_by_cm,fps):
    # mesh objects can be created from existing faces and vertex data
    xmin,xmax=np.amin(tripoints[:,0]),np.amax(tripoints[:,0])
    ymin,ymax=np.amin(tripoints[:,1]),np.amax(tripoints[:,1])
    zmin,zmax=np.amin(tripoints[:,2]),np.amax(tripoints[:,2])
    mesh = trimesh.Trimesh(vertices=tripoints,
                           faces=tricells)
   
    p1,p2=segment
    x1, y1 = p1
    x2, y2 = p2
    w=x2-x1
    h=y2-y1
    print("w",w,"h",h)
    print("xmin,xmax",xmin,xmax)    
    print("ymin,ymax",ymin,ymax)
    print("zmin,zmax",zmin,zmax)
    origin=(0,(x1+x2)/2,(y1+y2)/2)
    print("origin",origin)
    path3D =slice_with_inclined_plane(mesh,origin, w,h)
    print("path3D",path3D)
    intersect=find_intersection(p1, p2)
    polygons_3D,polygons_2D=get_polygons(path3D,(0,intersect[0],intersect[1]))
    fig = go.Figure()
    
    line_data = []
    diffs=[]
    for line in polygons_2D:
        #print("polygons_2D line",line.shape)
        x = line[:, 0]
        y = line[:, 1]

            
        slopes=[]
        slope=str(0)
        for i in range(len(line)-1):
            if line[i, 0]!=line[i+1, 0]:
                slope=round(fps*(line[i+1, 1]-line[i, 1])/(pixel_by_cm*(line[i+1, 0]-line[i, 0])),2)
                slopes.append('v='+str(slope))
                diffs.append(line[i+1, 0]-line[i, 0])
            else:
                slope=0
                slopes.append('v='+str(slope)) 
        slopes.append('v='+str(slope))
        line_data.append(go.Scatter(x=line[:, 0]/fps, y=line[:, 1]/pixel_by_cm, mode='lines', hovertemplate =
                            '<br>t: %{x}<br>'+
                            'h: %{y}<br>'+
                            '<b>%{text}</b>',
                            text = slopes, showlegend = False, line=dict(color='blue')))
    if len(diffs) >0:    
        print(np.amax(diffs),np.amin(diffs),np.average(np.absolute(diffs)))
    # Define layout
    layout = go.Layout(
        title=f'Intersection',
        xaxis=dict(title='Seconds', fixedrange=True  ),
        yaxis=dict(title='Height cm', range=[np.amin(tripoints[:, 2])/pixel_by_cm, np.amax(tripoints[:, 2])/pixel_by_cm]),
        showlegend=False 
    )
    fig = go.Figure(data=line_data, layout=layout)

    xrange=[np.amin(tripoints[:, 0])/fps, np.amax(tripoints[:, 0])/fps]    
    fig.update_xaxes(range=xrange)
    
    
    
    polylinesapprox = []
    for polygon in polygons_2D:
        polylinesapproxes=split_polygon_by_x_direction(polygon)
        for polyline  in polylinesapproxes: 
            xlength=abs(polyline[-1][0]-polyline[0][0])
            if xlength>fps/30:
                polylinesapprox.append(polyline)
             
 
    upper_line_data = []
    print("polylinesapprox",len(polylinesapprox))
    for line in polylinesapprox:
        slopes=[]
        slope=str(0)
        for i in range(len(line)-1):
            if line[i, 0]!=line[i+1, 0]:
                slope=round(fps*(line[i+1, 1]-line[i, 1])/(pixel_by_cm*(line[i+1, 0]-line[i, 0])),2)
                slopes.append('v='+str(slope))
                diffs.append(line[i+1, 0]-line[i, 0])
            else:
                slope=0
                slopes.append('v='+str(slope)) 
        slopes.append('v='+str(slope))
        upper_line_data.append(go.Scatter(x=line[:, 0]/fps, y=line[:, 1]/pixel_by_cm, mode='lines', hovertemplate =
                            '<br>t: %{x}<br>'+
                            'h: %{y}<br>'+
                            '<b>%{text}</b>',
                            text = slopes))
    # Create figure
    # Define layout
    upperlayout = go.Layout(
        title=f'Major moves',
        xaxis=dict(title='Seconds', fixedrange=True  ),
        yaxis=dict(title='Height cm', range=[np.amin(tripoints[:, 2])/pixel_by_cm, np.amax(tripoints[:, 2])/pixel_by_cm]),
        showlegend=True
    )  
    upperfig = go.Figure(data=upper_line_data, layout=upperlayout)
    upperfig.update_xaxes(range=xrange)
    # Extract the colors used for each trace in the first figure
    colors = [trace['line']['color'] for trace in upperfig.data]
    gradientdata=[]
    for l,line in enumerate(polylinesapprox):    
        dy_dx = np.gradient(line[:,1],line[:,0])
        gradient=go.Scatter(x=line[:, 0]/fps, y=dy_dx/pixel_by_cm, mode='lines', line=dict(color=colors[l]))
        gradientdata.append(gradient)
    gradfig = go.Figure(data=gradientdata)
    gradfig.update_layout(xaxis=dict(title='Seconds', fixedrange=True), yaxis_title="Speed cm/s",yaxis_range=[-5,5],showlegend=True  )
   
    gradfig.update_xaxes(range=xrange)
    return fig,upperfig,gradfig   

def getscale(img):
    # Compute the 2D FFT of the image
    f_transform = np.fft.fft2(img)
    # Shift the zero frequency component to the center
    f_shift = np.fft.fftshift(f_transform)

    # Create a mask with the same size as the image, 
    # but only keep the low frequencies so that only the regularly spaced dots are kept
    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow-100:crow+100, ccol-100:ccol+100] = 1

    # Apply the mask to the shifted DFT
    f_shift_filtered = f_shift * (1-mask)

    # Inverse shift and compute the inverse DFT
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    # apply threshold to get a clearer image
    img_back=np.where(img_back>63,255,0)
    
    
    # Apply the dilation to increase size of scale dots
    # Create a structuring element (e.g., a disk of radius 2)
    selem = morphology.disk(2)
    img_backlow = morphology.dilation(img_back, selem)
    # now find the upper dot in the top center of image where the center of the probe is
    left=ccol-5
    img_backlow_middle=img_backlow[:,ccol-5:ccol+5]
    img_backlow_middle_sum=np.sum(img_backlow_middle,axis=-1) 
    non_zero_indices_y = np.nonzero(img_backlow_middle)
    cx,cy=(non_zero_indices_y[0][0],non_zero_indices_y[1][0]+left)
    
    # now that we have the center make all rays from center horizontal with a polar warp
    img_backlow_polar=transform.warp_polar(img_backlow, center=(cx,cy))
    # only keep the noise free zones (remove too close or too far from center
    img_backlow_polar[:,0:25]=0
    img_backlow_polar[:,-200:-1]=0
    # Fuse points on a line by applying dilation with a large horizontal structuring element 
    selem = morphology.ellipse(width=25, height=1)
    dilated_image = morphology.dilation(img_backlow_polar, selem)
    # find the longest line 
    sumx=np.sum(dilated_image,axis=1)
    max_index = np.argmax(sumx)
    longest=img_backlow_polar[ max_index,:]
    # Apply Gaussian filter to only keep major peaks as some noise still occur 
    sigma = 9.0  # Standard deviation for Gaussian kernel
    smoothed_array = gaussian_filter(longest, sigma=sigma)
    peaks, _ = find_peaks(smoothed_array)
    pixel_by_cm=(peaks[-1]-peaks[0])/(len(peaks)-1)
    return pixel_by_cm, img_backlow
    
def parse_path(path_string):
    # Regular expression pattern to match coordinates
    pattern = r'M([\d.,L\s]+)'
    
    # Extract the coordinate string
    match = re.search(pattern, path_string)
    if match:
        coordinate_string = match.group(1)
    else:
        return None
    
    # Split the coordinate string into individual coordinates
    coordinates = coordinate_string.split('L')
    
    # Parse each coordinate
    parsed_coordinates = []
    for coord in coordinates:
        x, y = map(float, coord.split(','))
        parsed_coordinates.append((x, y))
    
    return parsed_coordinates
    
def thickline(image,start,end):
    # Calculate the direction vector of the line
    p1=(start[1],start[0])
    p2=(end[1],end[0])
    direction = np.array(p2) - np.array(p1)
    length = np.linalg.norm(direction)
    direction = direction / length
    # Draw the line segment with thickness 10
    thickness = 2
    for i in range(int(length) + 1):
        point = np.array(p1) + direction * i
        rr, cc = disk((point[0], point[1]), thickness / 2)
        image[rr, cc] = 1  
        
@app.callback(
    [Output('file-name', 'children'),
     Output('first-frame', 'figure'),
     Output('output-data-upload', 'children'),
     Output('annotations', 'children'),
     Output('plot1', 'figure'),
     Output('plot2', 'figure'),
     Output('plot3', 'figure'),
     Output('session-id', 'children')],
    [Input('upload-data', 'contents'),
     Input('first-frame', 'relayoutData')],
    [State('upload-data', 'filename'),
     State('upload-data', 'last_modified'),
     State('first-frame', 'figure'),
     State('session-id', 'children')]
)
def update_output(contents, relayoutData, filename, date,fig,session_id):
    triggered_id = ctx.triggered_id 
    print("triggered id",triggered_id)    
    print("session-id",session_id)    
    if session_id=="*":
        file_name_display=html.Div('No file selected')
        fig=no_update
        graph=no_update
        annotation_text='No annotations yet.'
        section=no_update 
        upperfig=no_update
        gradfig=no_update        
        session_id = str(uuid.uuid4())
        unique_filename = session_id + ".pkl"
        grayframes=[]
        pixel_by_cm=40
        fps=40
    else:
        # Generate a unique filename based on a UUID
        unique_filename = session_id + ".pkl"
        # Restore the variables from the pickle file
        with open(unique_filename, 'rb') as pickle_file:
            restored_variables = pickle.load(pickle_file)
            # Unpack the restored variables
            file_name_display, fig, graph, annotation_text, section, upperfig, gradfig,grayframes,pixel_by_cm,fps = restored_variables
    if triggered_id == 'upload-data':
        print( 'upload-data' )
        if contents is not None:
            file_name_display = html.Div(f'Selected file: {filename}')
            frames,grayframes,fps = process_video(contents, filename)
            
            first_frame = frames[0].astype(np.uint8)
            try:
                pixel_by_cm, img_backlow=getscale(grayframes[0].astype(np.uint8))
            except:
                pixel_by_cm=40
                height,width=grayframes[0].shape
                img_backlow=np.array(get_mask(height,width))
                annotation_text = "pas d'échelle trouvée"
                
            img_mask=np.where(img_backlow>0,0,1)
            for i,grayframe in enumerate(grayframes):
                grayframes[i]=grayframe*img_mask
            image = Image.fromarray(first_frame)
            img_width, img_height = image.size
            first_frame_base64 = frame_to_base64(first_frame)
            first_frame_img = 'data:image/jpeg;base64,{}'.format(first_frame_base64)
            
            fig = go.Figure(data=[go.Image(z=first_frame,x0=0,y0=0)])  # Use go.Image instead of px.imshow


            # Update layout to use drawrect
            fig.update_layout(
                dragmode='drawline',
                newshape=dict(line_color='cyan'),
                xaxis=dict(visible=False, range=[0, img_width]),
                yaxis=dict(visible=False, range=[img_height,0]), 
                margin=dict(l=0, r=0, t=0, b=0),
            )
            newgframes=[]     
            firstframe=frames[0]
            (height,width)=firstframe.shape[:2]  
            textzone=255-firstframe[70:90, 0:110, :]

            midx=int(width/2)
            midh=int(height/2)
            quartx=int(width/4)
            quarth=int(height/4)            
            eighth=int(height/8)
            y0=100
            y1=midh
            x0=midx-eighth
            x1=midx+eighth
            print(midx,midh)
            for gf in grayframes:
                newgf=gf.copy()*0
                newgf[0:2,0:2]=255
                newgf[-4:-1,-4:-1]=255
                newgf[int(y0):int(y1),int(x0):int(x1)]=gf[int(y0):int(y1),int(x0):int(x1)]               
                newgframes.append(newgf)
                
            mesh_fig,tripoints,tricells = generate_mesh(newgframes)
            shape={'editable': True, 'visible': True, 'xref': 'x', 'yref': 'y', 
                'layer': 'above', 'opacity': 1, 'line': {'color': 'cyan', 'width': 4, 'dash': 'solid'}, 'fillcolor': 'rgba(0,0,0,0)',
                'fillrule': 'evenodd', 'type': 'rect', 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}
            fig['layout']['shapes']=[shape]
            graph = dcc.Graph(figure=mesh_fig)
            segment=((midx,y0),(midx,y1))
            section,upperfig,gradfig=meshsection(tripoints,tricells,segment,pixel_by_cm,fps)
            print( 'End of content computation' )
 
    if triggered_id == 'first-frame' or triggered_id == 'upload-data': 
        if relayoutData and 'shapes' in relayoutData:
            
            fh,fw=grayframes[0].shape
            shapes = relayoutData['shapes']
            if shapes:
                mask=np.zeros((fh,fw), dtype=np.uint8)
                shape = shapes[-1]  # Get the latest shape
                relayoutData['shapes']=[shape]
                if shape['type']=='rect':
                    lx0,ly0=shape['x0'], shape['y0']
                    lx1,ly1=shape['x1'], shape['y1']
                    x0, y0 = min(lx0,lx1),min(ly0,ly1)
                    x1, y1 = max(lx0,lx1),max(ly0,ly1) 
                    w=abs(x1-x0)
                    h=abs(y1-y0)
                    midx=int((x1+x0)/2)
                    midy=fh-int((y1+y0)/2)
                    print("y0",y0,"y1",y1,"h",h)
                    if h>w:
                        segment=((midx,fh-y0),(midx,fh-y1))                        
                        p0=(midx,y0)
                        p1=(midx,y1)
                    else:
                        segment=((x0,midy),(x1,midy))                       
                        p0=(x0,midy)
                        p1=(x1,midy)
                    if x0==x1:
                        x1=x0+1
                    if y0==y1:
                        y1=y0+1
                    mask[int(y0):int(y1),int(x0):int(x1)]=1
                elif shape['type']=='path':
                    path_string=shape['path']
                    parsed_path = parse_path(path_string)
                    # Print the parsed coordinates 
                    path_coords=np.array(parsed_path).T.astype(int)
                    x0,y0=np.amin(path_coords[0,:]),np.amin(path_coords[1,:])
                    x1,y1=np.amax(path_coords[0,:]),np.amax(path_coords[1,:])
                    w=abs(x1-x0)
                    h=abs(y1-y0)
                    midx=int((x1+x0)/2)
                    midy=fh-int((y1+y0)/2)
                    
                    if h>w:
                        segment=((midx,fh-y0),(midx,fh-y1))
                        p0=(midx,y0)
                        p1=(midx,y1)
                    else:
                        segment=((x0,midy),(x1,midy))                       
                        p0=(x0,midy)
                        p1=(x1,midy)
                    if x0==x1:
                        x1=x0+1
                    if y0==y1:
                        y1=y0+1
                        
                    mask[int(y0):int(y1),int(x0):int(x1)]=gf[int(y0):int(y1),int(x0):int(x1)]=1
                elif shape['type']=='line':
                    line_string=shape['line']
                    print('line_string',line_string)
                    lx0,ly0=shape['x0'], shape['y0']
                    lx1,ly1=shape['x1'], shape['y1']
                    p0=(lx0,ly0)
                    p1=(lx1,ly1)
                    thickline(mask,p0,p1)
                    x0, y0 = min(lx0,lx1),min(ly0,ly1)
                    x1, y1 = max(lx0,lx1),max(ly0,ly1) 
                    segment=((x0,fh-y0),(x1,fh-y1))
                    margin=10
                    if abs(x0-x1)<margin:
                        x1=x0+margin
                        x0=x0-margin
                    if abs(y0-y1)<margin:
                        y1=y0+margin
                        y0=y0-margin
                    print('x0, y0,x1, y1',x0, y0,x1, y1)
                    
                    
                else:
                    print('shape type',shape['type'])
                    
                print("segment",segment)
                # Convert relative coordinates to absolute pixel coordinates
                top_left_abs = (int(x0), int(y0))
                bottom_right_abs = (int(x1), int(y1))
                print("top_left_abs",top_left_abs)
                print("bottom_right_abs",bottom_right_abs)
                annotations = [{'top_left': top_left_abs,'bottom_right': bottom_right_abs}]
                annotation_text = f'Coordinates: Top-left {top_left_abs}, Bottom-right {bottom_right_abs}'
                fig['layout']['shapes']=relayoutData['shapes']
                (height,width)=grayframes[0].shape[:2] 
                newgframes=[]       
                ngf=[]
                print('x0, y0,x1, y1',int(x0), int(y0),int(x1), int(y1)  )              
                for gf in grayframes:
                    newgf=gf.copy()*mask  
                    ngf.append(newgf)
                    newgf[0:2,0:2]=255
                    newgf[-4:-1,-4:-1]=255
                    newgframes.append(newgf)
                print(np.amax(newgframes[0]))
                mesh_fig,tripoints,tricells = generate_mesh(newgframes)
                
                graph = dcc.Graph(figure=mesh_fig)
                try:
                    section,upperfig,gradfig=meshsection(tripoints,tricells,segment,pixel_by_cm,fps)
                except Exception as inst:
                    print(type(inst))    # the exception type
                    print(inst.args)     # arguments stored in .args
                    print(inst)      
                    annotation_text=str(inst)
                print( 'End of shape computation' )   
        
    print("general return")
    with open(unique_filename, 'wb') as pickle_file:
         pickle.dump((file_name_display, fig, graph, annotation_text, section, upperfig, gradfig,grayframes,pixel_by_cm,fps), pickle_file)
    return file_name_display, fig, graph, annotation_text ,section, upperfig, gradfig,session_id 

if __name__ == '__main__':
    if not os.path.exists('temp'):
        os.makedirs('temp')   
    # remove previous session files
    pkl_files = glob.glob('*.pkl')
    for file in pkl_files:
        os.remove(file)
    app.run(debug=True,host="127.0.0.1")
