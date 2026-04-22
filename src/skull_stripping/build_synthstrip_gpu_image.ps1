param(
    [string]$WorkDir = ".external",
    [string]$FreeSurferCommit = "9f94380e7b134a4dea660f60d412c4cb4dd34144",
    [string]$ImageTag = "synthstrip-cuda-local:0.1",
    [string]$SourceImage = "freesurfer/synthstrip"
)

$ErrorActionPreference = "Stop"

Write-Host "[info] Building CUDA SynthStrip Docker image"
Write-Host "[info] WorkDir=$WorkDir"
Write-Host "[info] Commit=$FreeSurferCommit"
Write-Host "[info] ImageTag=$ImageTag"
Write-Host "[info] SourceImage=$SourceImage"

$repoRoot = Get-Location
$workPath = Join-Path $repoRoot $WorkDir
$ctxPath = Join-Path $workPath "synthstrip_gpu_build"

if (-not (Test-Path $workPath)) {
    New-Item -ItemType Directory -Path $workPath | Out-Null
}
if (Test-Path $ctxPath) {
    Remove-Item -Recurse -Force $ctxPath
}
New-Item -ItemType Directory -Path $ctxPath | Out-Null

$baseRaw = "https://raw.githubusercontent.com/freesurfer/freesurfer/$FreeSurferCommit/mri_synthstrip"

Write-Host "[step] Downloading Dockerfile.gpu and mri_synthstrip from commit..."
Invoke-WebRequest -Uri "$baseRaw/Dockerfile.gpu" -OutFile (Join-Path $ctxPath "Dockerfile.gpu")
Invoke-WebRequest -Uri "$baseRaw/mri_synthstrip" -OutFile (Join-Path $ctxPath "mri_synthstrip")

Write-Host "[step] Ensuring source SynthStrip image is available..."
docker pull $SourceImage | Out-Null

$containerName = "synthstrip_model_extract_$([Guid]::NewGuid().ToString('N').Substring(0,8))"
try {
    Write-Host "[step] Extracting model files from $SourceImage ..."
    docker create --name $containerName $SourceImage | Out-Null

    docker cp "${containerName}:/freesurfer/models/synthstrip.1.pt" (Join-Path $ctxPath "synthstrip.1.pt")
    docker cp "${containerName}:/freesurfer/models/synthstrip.nocsf.1.pt" (Join-Path $ctxPath "synthstrip.nocsf.1.pt")
}
finally {
    docker rm -f $containerName | Out-Null
}

Write-Host "[step] Building Docker image $ImageTag from local context..."
docker build -f (Join-Path $ctxPath "Dockerfile.gpu") -t $ImageTag $ctxPath

Write-Host "[done] Built image: $ImageTag"
Write-Host "[next] Run: python src/skull_strip.py --data-dir data --output-dir data_stripped --synthstrip-image $ImageTag"
