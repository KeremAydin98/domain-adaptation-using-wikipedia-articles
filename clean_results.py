import pandas as pd
import ast

df = pd.read_csv('/mnt/keremaydin/llava/scores.csv')
df_geo = pd.read_csv('/mnt/keremaydin/llava/geo_scores.csv')

correct_df = {}
correct_geo_df = {}
for col in df.columns:

    col_value = ast.literal_eval(df[col].dropna().iloc[0])
    geo_col_value = ast.literal_eval(df_geo[col].dropna().iloc[0])
    
    correct_df[col] = col_value
    correct_geo_df[col] = geo_col_value

df_true = pd.DataFrame(correct_df)
df_geo_true = pd.DataFrame(correct_geo_df)

df_true.to_excel('/mnt/keremaydin/llava/true_scores.xlsx')
df_geo_true.to_excel('/mnt/keremaydin/llava/geo_true_scores.xlsx')

