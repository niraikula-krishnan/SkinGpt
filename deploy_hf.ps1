# Deploy SkinGPT to Hugging Face Spaces
# Prerequisites: HF token with WRITE access from https://huggingface.co/settings/tokens

Write-Host "Step 1: Log in to Hugging Face (paste a Write token when prompted)"
hf auth login

Write-Host "`nStep 2: Upload app to Space..."
$staging = Join-Path $PSScriptRoot "..\hf_upload"
$src = $PSScriptRoot

if (Test-Path $staging) { Remove-Item -Recurse -Force $staging }
New-Item -ItemType Directory -Path $staging | Out-Null
Copy-Item "$src\app.py","$src\rag.py","$src\requirements.txt","$src\Dockerfile","$src\README.md","$src\class_names.json","$src\skin_model.h5","$src\.dockerignore" -Destination $staging
Copy-Item "$src\templates" -Destination $staging -Recurse
Copy-Item "$src\rag" -Destination $staging -Recurse

Push-Location $staging
hf upload niraikula-krishnan/SkinGpt . . --repo-type space --commit-message "Deploy SkinGPT Flask app"
Pop-Location

Write-Host "`nStep 3: Add secrets in Space settings (Settings -> Variables and secrets):"
Write-Host "  GROQ_API_KEY = your Groq API key (https://console.groq.com)"
Write-Host "  FLASK_SECRET_KEY = any random string"
Write-Host "`nDemo links:"
Write-Host "  https://huggingface.co/spaces/niraikula-krishnan/SkinGpt"
Write-Host "  https://niraikula-krishnan-skingpt.hf.space"
