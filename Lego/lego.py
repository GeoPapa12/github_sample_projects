import pandas as pd

inv_parts = pd.read_csv("inventory_parts.csv")
inv = pd.read_csv("inventories.csv")
sets = pd.read_csv("sets.csv")
sets_years = sets[['year', 'set_num']]
colors = pd.read_csv("colors.csv")

# inv_new = sets_years.merge(inv, how='inner', left_on='set_num', right_on='set_num')
inv_parts_new = inv_parts.merge(inv, how='inner', left_on='inventory_id', right_on='id')


df = inv.merge(sets, how='inner', left_on='set_num', right_on='set_num')
df = df.merge(inv_parts, how='inner', left_on='id', right_on='inventory_id')

df['decade'] = df['year'] - (df['year'] % 10)

df = df[['decade', 'color_id', 'set_num']]
df = df.merge(colors, how='inner', left_on='color_id', right_on='id')
df = df[['decade', 'name', 'set_num']]


colors_list = ['Black', 'Gray', 'Silver', 'White',
               'Red', 'Pink', 'purple', 'Brown', 'Yellow', 
               'Green', 'Blue']

df['name_s'] = 'other'
for clr in colors_list:
    df.loc[df['name'].str.contains(clr), 'name_s'] = clr

df_other = df[df['name_s'] == 'other']

df.to_csv('decade_vs_color.csv', index=False)
# inv_new.to_csv('inventories.csv', index=False)
# =============================================================

df_sets_new = inv_parts_new.groupby(['id', 'year']).sum()[['quantity']]
df_sets_new.to_csv('sets_qty_year.csv', index=True)
