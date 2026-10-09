$customNodes = Join-Path $PSScriptRoot 'custom_nodes'
$failed = @()

foreach ($node in Get-ChildItem $customNodes -Directory) {
    if (-not (Test-Path (Join-Path $node.FullName '.git'))) {
        continue
    }

    Write-Host "== $($node.Name)"
    Push-Location $node.FullName

    git fetch --all --prune --tags

    $upstream = git symbolic-ref -q refs/remotes/origin/HEAD
    if ($LASTEXITCODE -ne 0) {
        $upstream = 'refs/remotes/origin/main'
    }

    git rebase ($upstream -replace '^refs/remotes/', '')
    if ($LASTEXITCODE -ne 0) {
        $failed += $node.Name
    }

    Pop-Location
}

if ($failed.Count -gt 0) {
    Write-Host "Failed: $($failed -join ', ')"
}
