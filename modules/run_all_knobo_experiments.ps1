# ====== RUN ALL KNOBO EXPERIMENTS ======

Write-Host "Running KnoBo (PubMed, Prior=True)"
python modules/cbm.py `
    --mode binary `
    --bottleneck PubMed `
    --number_of_features 150 `
    --add_prior True `
    --modality xray `
    --model_name whyxrayclip

Write-Host "Running Linear Probe (150 features)"
python modules/cbm.py `
    --mode linear_probe `
    --number_of_features 150 `
    --modality xray `
    --model_name whyxrayclip

Write-Host "Running Linear Probe (768 features)"
python modules/cbm.py `
    --mode linear_probe `
    --number_of_features 768 `
    --modality xray `
    --model_name whyxrayclip

Write-Host "Running PCBM (PubMed)"
python modules/cbm.py `
    --mode pcbm `
    --bottleneck PubMed `
    --number_of_features 150 `
    --modality xray `
    --model_name whyxrayclip

# NOTE: only the PubMed bottleneck has trained grounding functions in data/grounding_functions/xray.
# Textbooks/StatPearls/Wikipedia need modules/train_grounding.py run first, so they are not included here.

Write-Host "Running KnoBo (PubMed, Prior=False)"
python modules/cbm.py `
    --mode binary `
    --bottleneck PubMed `
    --number_of_features 150 `
    --add_prior False `
    --modality xray `
    --model_name whyxrayclip

Write-Host "`nAll experiments finished."
Write-Host "Result files:"
ls data/results/
