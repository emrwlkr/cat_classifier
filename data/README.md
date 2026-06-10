# Data

This directory is gitignored — photos live here locally but are never committed.

## Layout

```
data/
├── luna/      ← JPEG photos of your first cat  (name the folder whatever you like)
└── mochi/     ← JPEG photos of your second cat
```

The folder name becomes the class label — name them after your cats.
Add at least a handful of photos per cat (10+ is ideal).
The training script reads `*.jpeg` from every subdirectory it finds here.
