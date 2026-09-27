param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$SourceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$TempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$WorkRoot = Join-Path $TempRoot ("paper3-samos-v1-" + [Guid]::NewGuid().ToString("N"))
$PreviousNoBytecode = $env:PYTHONDONTWRITEBYTECODE
$PreviousPythonPath = $env:PYTHONPATH

function Assert-ByteIdentical {
    param(
        [Parameter(Mandatory = $true)][string]$Expected,
        [Parameter(Mandatory = $true)][string]$Candidate
    )

    $ExpectedBytes = [System.IO.File]::ReadAllBytes($Expected)
    $CandidateBytes = [System.IO.File]::ReadAllBytes($Candidate)
    if (($ExpectedBytes.Length -ne $CandidateBytes.Length) -or
        ([Convert]::ToBase64String($ExpectedBytes) -cne [Convert]::ToBase64String($CandidateBytes))) {
        throw "Regenerated output differs from the frozen artifact: $Expected"
    }
}

try {
    & $Python (Join-Path $SourceRoot "verify_release.py")
    if ($LASTEXITCODE -ne 0) { throw "Release verification failed." }

    New-Item -ItemType Directory -Path $WorkRoot | Out-Null
    foreach ($Relative in Get-Content -LiteralPath (Join-Path $SourceRoot "FILES.txt")) {
        if ([string]::IsNullOrWhiteSpace($Relative)) { continue }
        $Source = Join-Path $SourceRoot $Relative
        $Target = Join-Path $WorkRoot $Relative
        New-Item -ItemType Directory -Path (Split-Path -Parent $Target) -Force | Out-Null
        Copy-Item -LiteralPath $Source -Destination $Target
    }

    $env:PYTHONDONTWRITEBYTECODE = "1"
    Push-Location $WorkRoot
    try {
        # E002 compact journal replay: no models or omitted raw tensor reads.
        $StrengtheningRoot = "results/paper3_strengthening_v2/public_v1"
        & $Python -B -m artifact.paper3_samos.build_strengthening_projection replay $StrengtheningRoot
        if ($LASTEXITCODE -ne 0) { throw "E002 compact replay failed" }
        & $Python -B -m artifact.paper3_samos.replay_packing_context --root (Join-Path $StrengtheningRoot "retained_context/packing")
        if ($LASTEXITCODE -ne 0) { throw "Historical packing-context replay failed" }
        $StrengtheningTables = Join-Path $WorkRoot ".review-strengthening-tables"
        & $Python -B -m src.report.paper3_strengthening_publication --source (Join-Path $StrengtheningRoot "private_audits/saved_record_audit.json") --output-directory $StrengtheningTables
        if ($LASTEXITCODE -ne 0) { throw "E002 table regeneration failed." }
        foreach ($Name in @("paper3_strengthening_service_v1.tex", "paper3_strengthening_geometry_v1.tex", "paper3_strengthening_publication_v1.json")) {
            Assert-ByteIdentical -Expected (Join-Path "docs/paper/generated" $Name) -Candidate (Join-Path $StrengtheningTables $Name)
        }
        & $Python -B -m pytest -q -p no:cacheprovider --basetemp (Join-Path $WorkRoot ".strengthening-tests") tests/test_paper3_strengthening_projection.py tests/test_paper3_strengthening_predictions.py tests/test_paper3_strengthening_publication.py tests/test_paper3_packing_context_replay.py
        if ($LASTEXITCODE -ne 0) { throw "E002 compact replay mutation tests failed." }
        & $Python -B -m src.report.paper3_transfer_publication_v1
        if ($LASTEXITCODE -ne 0) { throw "Transfer replay failed." }
        & $Python -B -m pytest tests/test_paper3_transfer_publication_v1.py -q -p no:cacheprovider
        if ($LASTEXITCODE -ne 0) { throw "Transfer mutation tests failed." }

        # Regenerate the causal analysis and its manifest in a separate root so
        # both exclusive-output CLIs operate on absent paths. The frozen release
        # files are read only and compared byte-for-byte afterward.
        $CausalRoot = Join-Path $WorkRoot ".review-causal-candidate"
        $CausalInputs = @(
            "configs/paper3_rk3576_fragmentation_pairs_v1.yaml",
            "docs/paper3_rk3576_fragmentation_pairs_v1.md",
            "results/rk3576/paper3_fragmentation_pairs_v1/paper3_rk3576_fragmentation_pairs_v1_raw.json",
            "scripts/rk3576/run_paper3_rk3576_fragmentation_pairs_v1.sh",
            "src/deployment/benchmark_rk3576_fragmentation_pairs.py",
            "src/deployment/benchmark_rk3576_sparse.py",
            "src/deployment/board_probe.py",
            "src/deployment/prepare_paper3_rk3576_fragmentation_pairs.py",
            "src/report/paper3_rk3576_fragmentation_pairs_manifest.py",
            "src/report/paper3_rk3576_fragmentation_pairs_v1.py",
            "src/runtime/sparse_refinement.py",
            "tests/test_benchmark_rk3576_fragmentation_pairs.py",
            "tests/test_paper3_rk3576_fragmentation_pairs_analysis.py",
            "tests/test_paper3_rk3576_fragmentation_pairs_contract.py",
            "tests/test_paper3_rk3576_fragmentation_pairs_manifest.py"
        )
        foreach ($Relative in $CausalInputs) {
            $Target = Join-Path $CausalRoot $Relative
            New-Item -ItemType Directory -Path (Split-Path -Parent $Target) -Force | Out-Null
            Copy-Item -LiteralPath (Join-Path $WorkRoot $Relative) -Destination $Target
        }
        $env:PYTHONPATH = $CausalRoot
        Push-Location $CausalRoot
        try {
            & $Python -m src.report.paper3_rk3576_fragmentation_pairs_v1 `
                --contract "configs/paper3_rk3576_fragmentation_pairs_v1.yaml" `
                --raw "results/rk3576/paper3_fragmentation_pairs_v1/paper3_rk3576_fragmentation_pairs_v1_raw.json" `
                --json-output "results/tables/paper3_rk3576_fragmentation_pairs_v1.json" `
                --markdown-output "results/tables/paper3_rk3576_fragmentation_pairs_v1.md" `
                --latex-output "docs/paper/generated/paper3_rk3576_fragmentation_pairs_v1.tex"
            if ($LASTEXITCODE -ne 0) { throw "Causal analysis regeneration failed." }
            & $Python -m src.report.paper3_rk3576_fragmentation_pairs_manifest `
                --contract "configs/paper3_rk3576_fragmentation_pairs_v1.yaml" `
                --output "results/manifests/paper3_rk3576_fragmentation_pairs_v1.json"
            if ($LASTEXITCODE -ne 0) { throw "Causal manifest regeneration failed." }
        } finally {
            Pop-Location
            $env:PYTHONPATH = $PreviousPythonPath
        }

        foreach ($Relative in @(
            "results/tables/paper3_rk3576_fragmentation_pairs_v1.json",
            "results/tables/paper3_rk3576_fragmentation_pairs_v1.md",
            "docs/paper/generated/paper3_rk3576_fragmentation_pairs_v1.tex",
            "results/manifests/paper3_rk3576_fragmentation_pairs_v1.json"
        )) {
            Assert-ByteIdentical `
                -Expected (Join-Path $WorkRoot $Relative) `
                -Candidate (Join-Path $CausalRoot $Relative)
        }

        # Recompute the successful second-graph result directly from the frozen
        # 120-sample capture. The output path is new and the frozen candidate is
        # compared byte-for-byte, so replay cannot rewrite evidence.
        $V22AnalysisCandidate = Join-Path $CausalRoot "paper3_shared_v2_fp16_rk3576_analysis_v22.json"
        & $Python -m src.deployment.run_paper3_shared_v2_fp16_v22 analyze `
            --capture "results/manifests/paper3_shared_v2_fp16_rk3576_capture_v22.json" `
            --output $V22AnalysisCandidate
        if ($LASTEXITCODE -ne 0) { throw "V22 second-graph analysis regeneration failed." }
        Assert-ByteIdentical `
            -Expected (Join-Path $WorkRoot "results/manifests/paper3_shared_v2_fp16_rk3576_analysis_v22.json") `
            -Candidate $V22AnalysisCandidate

        # Recompute the exact-graph RK3566 replication from its frozen
        # 120-sample capture. This uses the same value-blind offline analyzer;
        # no board backend is constructed or authorized.
        $V24AnalysisCandidate = Join-Path $CausalRoot "paper3_shared_v2_fp16_rk3566_analysis_v24.json"
        & $Python -m src.deployment.run_paper3_shared_v2_fp16_rk3566_v24 analyze `
            --capture "results/manifests/paper3_shared_v2_fp16_rk3566_capture_v24.json" `
            --output $V24AnalysisCandidate
        if ($LASTEXITCODE -ne 0) { throw "V24 RK3566 analysis regeneration failed." }
        Assert-ByteIdentical `
            -Expected (Join-Path $WorkRoot "results/manifests/paper3_shared_v2_fp16_rk3566_analysis_v24.json") `
            -Candidate $V24AnalysisCandidate

        $V32AnalysisCandidate = Join-Path $CausalRoot "paper3_width24_v32.json"
        $V32LatexCandidate = Join-Path $CausalRoot "paper3_width24_v32.tex"
        & $Python -m src.report.paper3_width24_rk3566_v32 `
            --json-output $V32AnalysisCandidate --latex-output $V32LatexCandidate
        if ($LASTEXITCODE -ne 0) { throw "V32 compact replay failed." }
        Assert-ByteIdentical -Expected (Join-Path $WorkRoot "results/tables/paper3_width24_int8_rk3566_analysis_v32.json") -Candidate $V32AnalysisCandidate
        Assert-ByteIdentical -Expected (Join-Path $WorkRoot "docs/paper/generated/paper3_width24_int8_rk3566_v32.tex") -Candidate $V32LatexCandidate
        & $Python -m pytest -p no:cacheprovider --basetemp (Join-Path $CausalRoot "v32-tests") tests/test_paper3_width24_rk3566_analysis_v32.py -q
        if ($LASTEXITCODE -ne 0) { throw "V32 compact tests failed." }

        # Replay all 5,120 dispatch timing/decision rows. This verifies the
        # compact projection, not omitted full raw numerical arrays.
        $DispatchJson = Join-Path $CausalRoot "paper3_dispatch_publication_v1.json"
        $DispatchTex = Join-Path $CausalRoot "paper3_dispatch_publication_v1.tex"
        & $Python -m src.report.paper3_dispatch_publication_v1 `
            --json-output $DispatchJson --latex-output $DispatchTex
        if ($LASTEXITCODE -ne 0) { throw "Dispatch compact replay failed." }
        Assert-ByteIdentical -Expected (Join-Path $WorkRoot "results/tables/paper3_dispatch_publication_v1.json") -Candidate $DispatchJson
        Assert-ByteIdentical -Expected (Join-Path $WorkRoot "docs/paper/generated/paper3_dispatch_publication_v1.tex") -Candidate $DispatchTex
        & $Python -m pytest -p no:cacheprovider --basetemp (Join-Path $CausalRoot "dispatch-tests") tests/test_paper3_dispatch_publication_v1.py -q
        if ($LASTEXITCODE -ne 0) { throw "Dispatch compact tests failed." }

        # Regenerate the self-contained NPU-only organization summary. Its
        # accounting comes from Paper 3 masks and dispatched RKNN shapes only.
        $NpuSummaryCandidate = Join-Path $CausalRoot "paper3_npu_fragmentation_summary_v1.json"
        $NpuSummaryMarkdownCandidate = Join-Path $CausalRoot "paper3_npu_fragmentation_summary_v1.md"
        & $Python -m src.report.paper3_npu_fragmentation_summary_v1 `
            --input "results/tables/codesign_matched_masks_rk3576_v1_public.json" `
            --json-output $NpuSummaryCandidate `
            --markdown-output $NpuSummaryMarkdownCandidate
        if ($LASTEXITCODE -ne 0) { throw "Paper 3 NPU-only summary regeneration failed." }
        Assert-ByteIdentical `
            -Expected (Join-Path $WorkRoot "results/tables/paper3_npu_fragmentation_summary_v1.json") `
            -Candidate $NpuSummaryCandidate
        Assert-ByteIdentical `
            -Expected (Join-Path $WorkRoot "results/tables/paper3_npu_fragmentation_summary_v1.md") `
            -Candidate $NpuSummaryMarkdownCandidate


        $PytestTemp = Join-Path $WorkRoot ".review-pytest-temp"
        & $Python -m pytest -p no:cacheprovider `
            --basetemp $PytestTemp `
            tests/test_paper3_rk3576_fragmentation_pairs_contract.py `
            tests/test_benchmark_rk3576_fragmentation_pairs.py `
            tests/test_paper3_rk3576_fragmentation_pairs_analysis.py `
            tests/test_paper3_rk3576_fragmentation_pairs_manifest.py `
            -k "not offline_preflight_hashes_every_accepted_dependency_without_authorizing_samples" `
            -q
        if ($LASTEXITCODE -ne 0) { throw "Paper 3 fragmentation tests failed." }

        & $Python -m pytest -p no:cacheprovider `
            --basetemp $PytestTemp `
            tests/test_paper3_npu_fragmentation_summary.py `
            -q
        if ($LASTEXITCODE -ne 0) { throw "Paper 3 independent-analysis/manuscript tests failed." }

        & $Python -m pytest -p no:cacheprovider `
            --basetemp $PytestTemp `
            tests/test_paper3_shared_v2_fp16_v22.py `
            -q
        if ($LASTEXITCODE -ne 0) { throw "Paper 3 V22 second-graph tests failed." }


    } finally {
        Pop-Location
    }
} finally {
    $env:PYTHONDONTWRITEBYTECODE = $PreviousNoBytecode
    $env:PYTHONPATH = $PreviousPythonPath
    $ResolvedWork = [IO.Path]::GetFullPath($WorkRoot)
    if ((Test-Path -LiteralPath $ResolvedWork) -and $ResolvedWork.StartsWith($TempRoot) -and $ResolvedWork -ne $TempRoot) {
        Remove-Item -LiteralPath $ResolvedWork -Recurse -Force
    }
}

Write-Host "PAPER3_SAMOS_REPRODUCTION_PASS"

& $Python (Join-Path $SourceRoot "verify_release.py")
if ($LASTEXITCODE -ne 0) { throw "Release mutated during replay." }
