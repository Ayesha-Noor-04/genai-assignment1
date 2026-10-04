from pathlib import Path
from torchvision.datasets import OxfordIIITPet

out = Path("backend/samples")
out.mkdir(parents=True, exist_ok=True)
ds = OxfordIIITPet("data", split="test", download=True)
for n, i in enumerate(range(0, len(ds), 150)):
    ds[i][0].convert("RGB").resize((128, 128)).save(out / f"sample_{n:02d}.png")
print("saved", n + 1, "samples")