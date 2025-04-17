import folium
import pandas as pd
from folium.plugins import HeatMap

# Sample data for safe locations
safe_locations = pd.DataFrame({
    'Location': ['Park Street', 'Library Road', 'Greenway', 'Sunset Blvd'],
    'Latitude': [13.0827, 13.0350, 13.0455, 13.0802],
    'Longitude': [80.2707, 80.2315, 80.2502, 80.2837],
    'SafetyLevel': ['High', 'High', 'Moderate', 'High']
})

# Initialize folium map
m = folium.Map(
    location=[safe_locations['Latitude'].mean(), safe_locations['Longitude'].mean()],
    zoom_start=13,
    tiles='CartoDB positron'
)

# ✅ Add a title using HTML injected into the map
title_html = '''
     <div style="position: fixed; top: 10px; width: 100%; text-align: center; z-index:9999;">
         <h3 style="font-size:22px; color:green; background-color: white; display: inline-block; padding: 5px 15px; border-radius: 8px;">
         <b>Safety Locations</b></h3>
     </div>
'''
m.get_root().html.add_child(folium.Element(title_html))

# Add green circles for safe locations
for idx, row in safe_locations.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=10,
        color='green',
        fill=True,
        fill_color='green',
        fill_opacity=0.6,
        popup=f"Location: {row['Location']}<br>Safety Level: {row['SafetyLevel']}"
    ).add_to(m)

# Add heatmap with green gradient
HeatMap(
    safe_locations[['Latitude', 'Longitude']].values,
    radius=25,
    gradient={'0.4': 'green', '0.65': 'lime', '1': 'lightgreen'}
).add_to(m)

# Save to HTML file
m.save("safe_location_heatmap.html")
print("✅ Map saved as safe_location_heatmap.html")
