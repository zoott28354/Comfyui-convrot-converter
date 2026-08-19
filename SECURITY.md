# Security policy

## Supported version

Security fixes are applied to the latest version on the `main` branch. Older snapshots are not maintained separately.

## Untrusted model files

Safetensors is the recommended input format. PyTorch `.pth`, `.pt`, `.ckpt`, and `.bin` checkpoints have a larger attack surface even when loaded with `weights_only=True`. Open them only when they come from a source you trust. The GUI displays a warning before opening these formats, and the command-line converter requires `--allow-pytorch-checkpoint`.

The converter runs with the permissions of the current Windows user. It is not a sandbox for malicious model files. For untrusted inputs, use a disposable virtual machine or another isolated environment.

## Reporting a vulnerability

Do not publish exploit details in a normal issue. Use GitHub's private **Report a vulnerability** feature when it is available for this repository. For non-sensitive security improvements, open a regular issue and label it as security-related.

Include the affected version or commit, reproduction steps, expected impact, and whether a proof of concept accesses or modifies local files.
