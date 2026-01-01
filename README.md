# FLUX.1-Schnell RunPod Serverless Worker

Fast text-to-image generation using FLUX.1-Schnell.

## Features

- **Model Baked-In** - No download on cold start (~30s startup)
- **Network Volume Support** - Even faster loading if volume attached
- **4-Step Inference** - ~3 seconds per image
- **24GB VRAM Optimized** - RTX 4090, A10, L40S

## Deploy from GitHub

### 1. Push to GitHub

```bash
# Create new repo on GitHub: flux-schnell-runpod

cd runpod-flux
git init
git add .
git commit -m "FLUX.1-Schnell RunPod Worker"
git remote add origin https://github.com/YOUR_USERNAME/flux-schnell-runpod.git
git push -u origin main
```

### 2. RunPod Serverless Setup

1. Go to https://console.runpod.io/serverless
2. Click **"New Endpoint"**
3. Select **"GitHub Repo"** as source
4. Connect your GitHub account
5. Select your `flux-schnell-runpod` repo
6. Settings:

| Setting | Value |
|---------|-------|
| **Branch** | main |
| **Dockerfile Path** | Dockerfile |
| **Container Disk** | 50 GB |
| **GPU Type** | 24 GB VRAM (RTX 4090) |
| **Max Workers** | 1-3 |
| **Idle Timeout** | 5 seconds |
| **Execution Timeout** | 120 seconds |

7. Click **Deploy**

Build takes ~15-20 minutes (downloads 25GB model).

## API Usage

### Request

```json
{
  "input": {
    "prompt": "a beautiful sunset over mountains, photorealistic, 8k",
    "width": 1024,
    "height": 1024,
    "seed": 12345,
    "num_inference_steps": 4,
    "guidance": 0.0
  }
}
```

### Response

```json
{
  "image": "base64_encoded_png...",
  "image_base64": "base64_encoded_png...",
  "seed": 12345,
  "width": 1024,
  "height": 1024,
  "inference_time": 2.8
}
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Image description |
| `width` | int | 1024 | Width (256-1536, divisible by 8) |
| `height` | int | 1024 | Height (256-1536, divisible by 8) |
| `seed` | int | random | Reproducibility seed |
| `num_inference_steps` | int | 4 | Denoising steps (4 optimal for Schnell) |
| `guidance` | float | 0.0 | Guidance scale (0.0 for Schnell) |

## Performance

| Resolution | Steps | Time (RTX 4090) |
|------------|-------|-----------------|
| 576x1024 | 4 | ~2.5s |
| 1024x1024 | 4 | ~3.5s |
| 1024x1536 | 4 | ~5s |

## Network Volume (Optional)

If you attach a Network Volume, the model will be cached there for even faster loading:

- Path: `/runpod-volume/models/flux-schnell`
- First run saves model to volume
- Subsequent cold starts load from NVMe (~15s vs ~30s)

## Troubleshooting

### CUDA OOM
- Reduce dimensions to max 1024x1024
- Or use 48GB GPU

### Slow first request
- Normal: Model loads to GPU (~30s)
- Subsequent requests: ~3s

### Build fails
- Check Dockerfile syntax
- Ensure enough disk space (50GB+)
