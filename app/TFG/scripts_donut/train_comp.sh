#!/bin/bash

# python app/TFG/scripts_donut/train_model.py -tr 5  -va 5 -ts 100 -o app/TFG/outputs/donut/donut_comp_1x5_5_100 -n donut_comp_1x5_5_100 -k
# python app/TFG/scripts_donut/train_model.py -tr 10 -va 5 -ts 100 -o app/TFG/outputs/donut/donut_comp_2x5_5_100 -n donut_comp_2x5_5_100 -k
# python app/TFG/scripts_donut/train_model.py -tr 15 -va 5 -ts 100 -o app/TFG/outputs/donut/donut_comp_3x5_5_100 -n donut_comp_3x5_5_100 -k
# python app/TFG/scripts_donut/train_model.py -tr 20 -va 5 -ts 100 -o app/TFG/outputs/donut/donut_comp_4x5_5_100 -n donut_comp_4x5_5_100 -k
# python app/TFG/scripts_donut/train_model.py -tr 25 -va 5 -ts 100 -o app/TFG/outputs/donut/donut_comp_5x5_5_100 -n donut_comp_5x5_5_100 -k


TRAIN_VALUES=(5 10 15 20 25 30)
N_TEMPLATES=5
VAL=(30 30 30 30 30 30)
TEST=100
BASE_OUTPUT="app/TFG/outputs/donut"
SCRIPT="app/TFG/scripts_donut/train_model.py"

echo "🔧 Starting training for ${#TRAIN_VALUES[@]} configurations..."

for i in "${!TRAIN_VALUES[@]}"; do
    TR=${TRAIN_VALUES[$i]}
    VAL_CURR=${VAL[$i]}
    IDX=$((i + 1))
    NAME="donut_comp_${IDX}x${N_TEMPLATES}_${VAL_CURR}_${TEST}"
    OUTPUT="${BASE_OUTPUT}/${NAME}"
    
    echo "🚀 Training model: ${NAME} (Train: ${TR}, Val: ${VAL_CURR}, Test: ${TEST})"
    python "$SCRIPT" -tr "$TR" -va "$VAL_CURR" -ts "$TEST" -o "$OUTPUT" -n "$NAME" -k
done

echo "✅ Trained ${#TRAIN_VALUES[@]} model(s) successfully!"