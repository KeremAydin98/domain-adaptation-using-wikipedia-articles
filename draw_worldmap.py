import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world.rename(columns={'name': 'country'}, inplace=True)
world_geo = world.copy()

df = pd.read_excel('true_scores.xlsx', index_col=0)
df_geo = pd.read_excel('geo_true_scores.xlsx', index_col=0)

df.rename(columns={'United States': 'United States of America',
                    'Czech Republic':'Czechia',
                    'Democratic Republic of the Congo': 'Dem. Rep. Congo',
                    'Central African Republic': 'Central African Rep.',
                    'South Sudan':'S. Sudan', 'Bosnia and Herzegovina':'Bosnia and Herz.',
                    'Dominican Republic':'Dominican Rep.'}, inplace=True)
df_geo.rename(columns={'United States': 'United States of America',
                    'Czech Republic':'Czechia',
                    'Democratic Republic of the Congo': 'Dem. Rep. Congo',
                    'Central African Republic': 'Central African Rep.',
                    'South Sudan':'S. Sudan', 'Bosnia and Herzegovina':'Bosnia and Herz.',
                    'Dominican Republic':'Dominican Rep.'}, inplace=True)

scores = []
geo_scores = []
differences = []
for country in world['country']:

    if country == 'Cyprus':
        score = 0
        geo_score = 0

        scores.append(score)
        geo_scores.append(geo_score)
        differences.append(geo_score - score)

        continue

    try:
        green = df.loc['green', country]
        yellow = df.loc['yellow', country]
        red = df.loc['red', country]

        score = int((green + int(0.5 * yellow)) / (green + yellow + red) * 100)

        geo_green = df_geo.loc['green', country]
        geo_yellow = df_geo.loc['yellow', country]
        geo_red = df_geo.loc['red', country] 

        geo_score = int((geo_green + int(0.5 * geo_yellow)) / (geo_green + geo_yellow + geo_red) * 100)

        scores.append(score)
        geo_scores.append(geo_score)

        if score != 0:
            differences.append(100 * (geo_score - score) / score)
        else:
            differences.append(100 * (geo_score - score) / 1)

    except:
        score = None
        geo_score = None

        scores.append(score)
        geo_scores.append(geo_score)
        differences.append(None)

    

world['scores'] = scores
world['geo_scores'] = geo_scores
world['differences'] = differences

# Plot the world map
world.plot()


world.to_excel('world_scores.xlsx', index=False)


# Customize the colors based on scores
fig, ax = plt.subplots(1, 1, figsize=(15, 10))  # Increase the size of the figure
world.plot(column='scores', cmap='coolwarm', vmax=world['scores'].max(), vmin=0,legend=False, ax=ax)
cbar = ax.get_figure().colorbar(ax.collections[0], ax=ax, shrink=0.7)  # Shrink the colorbar
world[world['scores'].isnull()].plot(ax=ax, color='black', hatch='///')
plt.xticks([])
plt.yticks([])
# Save the plot as a JPEG file
plt.savefig('scores.jpg', format='jpg', dpi=900)

# Customize the colors based on scores
fig, ax = plt.subplots(1, 1, figsize=(15, 10))  # Increase the size of the figure
world.plot(column='geo_scores', cmap='coolwarm', vmax=world['geo_scores'].max(), vmin=0, legend=False, ax=ax)
# Adjust the colorbar
cbar = ax.get_figure().colorbar(ax.collections[0], ax=ax, shrink=0.7)  # Shrink the colorbar
# Plot the countries with NaN geo_scores in black with hatching
world[world['geo_scores'].isnull()].plot(ax=ax, color='black', hatch='///')
plt.xticks([])
plt.yticks([])
# Save the plot as a JPEG file
plt.savefig('geo_scores.jpg', format='jpg', dpi=900)

# Customize the colors based on scores
fig, ax = plt.subplots(1, 1, figsize=(15, 10))  # Increase the size of the figure
world.plot(column='differences', cmap='RdYlBu', vmax=world['differences'].max(), vmin=-world['differences'].max(),legend=False, ax=ax)
cbar = ax.get_figure().colorbar(ax.collections[0], ax=ax, shrink=0.7)  # Shrink the colorbar
world[world['differences'].isnull()].plot(ax=plt.gca(), color='black', hatch='///')
plt.xticks([])
plt.yticks([])
# Save the plot as a JPEG file
plt.savefig('diff_scores.jpg', format='jpg', dpi=900)
