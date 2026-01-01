"""
FLUX.1-Schnell RunPod Serverless Handler
- Model baked into Docker image
- Network Volume support for faster loading
- Optimized for 24GB VRAM
"""

import os
import io
import base64
import time
import torch
import runpod
from PIL import Image

# Global model reference
pipe = None
MODEL_LOADED = False


def get_model_path():
    """Get best model path - Network Volume > Baked-in cache"""
    # Network Volume paths (faster NVMe)
    volume_paths = [
        "/runpod-volume/models/flux-schnell",
        "/runpod-volume/flux-schnell",
        "/workspace/models/flux-schnell"
    ]

    for path in volume_paths:
        if os.path.exists(path) and os.path.isdir(path):
            # Check if model files exist
            if os.path.exists(os.path.join(path, "model_index.json")):
                print(f"Found model in Network Volume: {path}")
                return path

    # Fallback to HuggingFace cache (baked into image)
    return "black-forest-labs/FLUX.1-schnell"


def save_to_volume(pipe):
    """Save model to Network Volume for faster future loads"""
    volume_path = "/runpod-volume/models/flux-schnell"

    if os.path.exists("/runpod-volume") and not os.path.exists(volume_path):
        try:
            print(f"Saving model to Network Volume: {volume_path}")
            os.makedirs(volume_path, exist_ok=True)
            pipe.save_pretrained(volume_path)
            print("Model saved to Network Volume!")
        except Exception as e:
            print(f"Could not save to volume: {e}")


def load_model():
    """Load FLUX.1-Schnell model with optimizations"""
    global pipe, MODEL_LOADED

    if MODEL_LOADED and pipe is not None:
        return pipe

    print("=" * 60)
    print("FLUX.1-Schnell - Loading Model")
    print("=" * 60)

    start = time.time()

    from diffusers import FluxPipeline

    model_path = get_model_path()
    print(f"Loading from: {model_path}")

    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    )

    # Move to GPU
    pipe.to("cuda")

    # Memory optimizations for 24GB
    pipe.enable_attention_slicing(slice_size="auto")

    # Try xformers
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xformers memory efficient attention enabled")
    except Exception:
        print("xformers not available, using default attention")

    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.1f}s")

    # Save to Network Volume for faster future loads
    if model_path == "black-forest-labs/FLUX.1-schnell":
        save_to_volume(pipe)

    print("=" * 60)

    MODEL_LOADED = True
    return pipe


def handler(job):
    """
    RunPod handler for FLUX.1-Schnell image generation

    Input:
        prompt: str - Image description
        width: int - Image width (default: 1024, max: 1536)
        height: int - Image height (default: 1024, max: 1536)
        seed: int - Random seed (optional)
        num_inference_steps: int - Steps (default: 4)
        guidance: float - Guidance scale (default: 0.0)

    Output:
        image: str - Base64 encoded PNG
        image_base64: str - Alias for compatibility
        seed: int - Used seed
    """
    job_input = job.get("input", {})

    # Extract parameters
    prompt = job_input.get("prompt", "A beautiful landscape")
    width = job_input.get("width", 1024)
    height = job_input.get("height", 1024)
    seed = job_input.get("seed")

    # Schnell defaults
    num_inference_steps = job_input.get("num_inference_steps", 4)
    guidance_scale = job_input.get("guidance", job_input.get("guidance_scale", 0.0))

    # Validate dimensions (divisible by 8)
    width = (width // 8) * 8
    height = (height // 8) * 8

    # Clamp for memory (24GB safe limits)
    width = max(256, min(width, 1536))
    height = max(256, min(height, 1536))

    print(f"\n{'='*60}")
    print(f"FLUX.1-Schnell Generation Request")
    print(f"{'='*60}")
    print(f"Size: {width}x{height}")
    print(f"Steps: {num_inference_steps}")
    print(f"Guidance: {guidance_scale}")
    print(f"Prompt: {prompt[:100]}...")

    # Load model
    try:
        model = load_model()
    except Exception as e:
        print(f"Model load error: {e}")
        return {"error": f"Failed to load model: {str(e)}"}

    # Setup seed
    if seed is None:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()

    generator = torch.Generator(device="cuda").manual_seed(seed)
    print(f"Seed: {seed}")

    # Generate
    try:
        gen_start = time.time()

        with torch.inference_mode():
            result = model(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )

        gen_time = time.time() - gen_start
        print(f"Generated in {gen_time:.2f}s")

        # Convert to base64
        image = result.images[0]
        buffered = io.BytesIO()
        image.save(buffered, format="PNG", optimize=True)
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        print(f"Output: {len(image_base64) // 1024}KB base64")
        print(f"{'='*60}\n")

        return {
            "image": image_base64,
            "image_base64": image_base64,
            "seed": seed,
            "width": width,
            "height": height,
            "inference_time": round(gen_time, 2)
        }

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("CUDA OOM - clearing cache")
        return {"error": "Out of memory. Try smaller dimensions (max 1024x1024)."}

    except Exception as e:
        print(f"Generation error: {e}")
        return {"error": str(e)}


# ============================================
# WORKER STARTUP
# ============================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FLUX.1-Schnell RunPod Serverless Worker")
    print("=" * 60)
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Not available")
    print("VRAM:", f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if torch.cuda.is_available() else "N/A")
    print("=" * 60 + "\n")

    # Pre-load model for warm start
    print("Pre-loading model...")
    load_model()
    print("Worker ready!\n")

    # Start RunPod handler
    runpod.serverless.start({"handler": handler})
