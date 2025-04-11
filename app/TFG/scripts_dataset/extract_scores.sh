#!/bin/bash
# Stop the script if any command fails
set -e

# python TFG/scripts_dataset/validate_model.py -o TFG/outputs/donut_v2/donut_comp_1x5_30_100

TESTING_ROUTES=(
    TFG/outputs/donut/donut_comp_1x5
    TFG/outputs/donut/donut_comp_2x5
    TFG/outputs/donut/donut_comp_3x5
    TFG/outputs/donut/donut_comp_4x5
    TFG/outputs/donut/donut_comp_5x1_t1_v1
    TFG/outputs/donut/donut_comp_5x5
    TFG/outputs/donut/donut_comp_5x5_v2
    TFG/outputs/donut/donut_comp_6x5
    TFG/outputs/donut/donut_comp_7x5


    TFG/outputs/donut_v2/donut_comp_1x5_30_100
    TFG/outputs/donut_v2/donut_comp_2x5_10_100
    TFG/outputs/donut_v2/donut_comp_3x5_10_100
    TFG/outputs/donut_v2/donut_comp_3x5_30_100
    TFG/outputs/donut_v2/donut_comp_4x5_30_100
    TFG/outputs/donut_v2/donut_comp_5x5_15_100
    TFG/outputs/donut_v2/donut_comp_5x5_30_100
    TFG/outputs/donut_v2/donut_comp_6x5_30_100

    TFG/outputs/orc_llm_keep/ocr_finetuned_2x5_v1
    TFG/outputs/orc_llm_keep/ocr_finetuned_3x5_v1
    TFG/outputs/orc_llm_keep/ocr_finetuned_4x5_v1
    TFG/outputs/orc_llm_keep/ocr_finetuned_4x5_v1_fail
    TFG/outputs/orc_llm_keep/ocr_finetuned_5x1_v1
    TFG/outputs/orc_llm_keep/ocr_finetuned_5x5_v1
    TFG/outputs/orc_llm_keep/ocr_finetuned_5x5_v1_205
    TFG/outputs/orc_llm_keep/ocr_finetuned_5x5_v1_fail
)

echo "🔧 Starting validation for ${#TESTING_ROUTES[@]} configurations..."

for i in "${!TESTING_ROUTES[@]}"; do
    TR=${TESTING_ROUTES[$i]}
    
    echo "Validating output at ${TR}"
    # If this command fails, the script will exit here
    python TFG/scripts_dataset/validate_model.py -o ${TR} -f 100
done

echo "✅ Validated ${#TESTING_ROUTES[@]} outputs successfully!"