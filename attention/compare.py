import wandb
import pandas as pd

wandb.finish()
wandb.init(
    project="Comparison_vanilla_attention",
    name = 'Compare_attention_vanilla',
    resume="never",
    reinit=True,
)
vanilla_predictions = pd.read_csv("predictions_vanilla.csv")
attention_predictions = pd.read_csv("predictions_attention.csv")
comparison_df = vanilla_predictions.copy()
comparison_df['Vanilla_prediction'] = comparison_df['Predicted']
comparison_df['Attention_prediction'] = attention_predictions['Predicted']
comparison_df['Attention_Is_Correct'] = attention_predictions['Is_Correct']
filtered_df = comparison_df[
    (comparison_df['Is_Correct'] == "False❌") &
    (comparison_df['Attention_Is_Correct'] == "True✅")
][['Input', 'Target', 'Vanilla_prediction', 'Attention_prediction']]
filtered_df.reset_index(drop=True, inplace=True)
filtered_df.to_csv("comparison.csv", index=False)
table = wandb.Table(dataframe=filtered_df)
wandb.log({"Comparison_table": table})
# finish run
wandb.finish()