

from TFG.scripts_dataset.metrics import print_scores


print_scores(
    {
        "ALL": (10, 0.1),
        "BOOBS": (1, 0.01)
    },
    100
)