# Security policy

This research simulator does not accept untrusted network model payloads. Its
VGAE CLI loader uses PyTorch's restricted `weights_only=True` mode and a
tensor/primitive-only payload. Still obtain checkpoints from trusted sources:
restricted loading does not prevent resource-exhaustion tensors or guarantee
the scientific provenance of model weights. Other ad-hoc PyTorch files should
never be loaded with unrestricted pickle deserialization.

Report security issues privately to the repository owner rather than opening a
public issue. External UERANSIM/Open5GS adapters must add authentication,
transport security, schema validation and payload-size limits before deployment.
