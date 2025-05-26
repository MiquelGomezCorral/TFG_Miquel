#!/bin/bash
# Stop the script if any command fails
set -e

# python TFG/scripts_dataset/validate_model.py -o TFG/outputs/donut_v2/donut_comp_1x5_30_100

TESTING_ROUTES=(
    src/outputs/donut/donut_comp_1x5
    src/outputs/donut/donut_comp_2x5
    src/outputs/donut/donut_comp_3x5
    src/outputs/donut/donut_comp_4x5
    src/outputs/donut/donut_comp_5x1_t1_v1
    src/outputs/donut/donut_comp_5x5
    src/outputs/donut/donut_comp_5x5_v2
    src/outputs/donut/donut_comp_6x5
    src/outputs/donut/donut_comp_7x5


    src/outputs/donut_v2/donut_comp_1x5_30_100
    src/outputs/donut_v2/donut_comp_2x5_10_100
    src/outputs/donut_v2/donut_comp_3x5_10_100
    src/outputs/donut_v2/donut_comp_3x5_30_100
    src/outputs/donut_v2/donut_comp_4x5_30_100
    src/outputs/donut_v2/donut_comp_5x5_15_100
    src/outputs/donut_v2/donut_comp_5x5_30_100
    src/outputs/donut_v2/donut_comp_6x5_30_100

    src/outputs/orc_llm_keep/ocr_finetuned_2x5_v1
    src/outputs/orc_llm_keep/ocr_finetuned_3x5_v1
    src/outputs/orc_llm_keep/ocr_finetuned_4x5_v1
    src/outputs/orc_llm_keep/ocr_finetuned_5x1_v1
    src/outputs/orc_llm_keep/ocr_finetuned_5x5_v1
    src/outputs/orc_llm_keep/ocr_finetuned_5x5_v1_205
    src/outputs/orc_llm_keep/prebuilt-read_structured
    src/outputs/orc_llm_keep/prebuilt-read_raw_lines
    src/outputs/orc_llm_keep/prebuilt-invoice_structured
    src/outputs/orc_llm_keep/prebuilt-invoice_raw_lines
)

echo "🔧 Starting validation for ${#TESTING_ROUTES[@]} configurations..."

for i in "${!TESTING_ROUTES[@]}"; do
    TR=${TESTING_ROUTES[$i]}
    
    echo "=========================================================================================================="
    echo "                           Validating output at ${TR}"
    echo "=========================================================================================================="
    # If this command fails, the script will exit here
    python TFG/scripts_dataset/validate_model.py -o ${TR} -f 100
done

echo "✅ Validated ${#TESTING_ROUTES[@]} outputs successfully!"