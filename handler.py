"""
FLUX.1-Schnell RunPod Serverless Handler
Uses HF_TOKEN environment variable for authentication
Model cached to Network Volume for fast subsequent loads
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


def load_model():
    """Load FLUX.1-Schnell model with HF_TOKEN auth"""
    global pipe, MODEL_LOADED

    if MODEL_LOADED and pipe is not None:
        return pipe

    print("=" * 60)
    print("FLUX.1-Schnell - Loading Model")
    print("=" * 60)

    # Login with HF_TOKEN
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Logging in to HuggingFace...")
        from huggingface_hub import login
        login(token=hf_token)
        print("Logged in successfully!")
    else:
        print("WARNING: No HF_TOKEN found!")

    start = time.time()

    from diffusers import FluxPipeline

    # Check Network Volume cache first
    cache_path = "/runpod-volume/models/flux-schnell"

    if os.path.exists(cache_path) and os.path.exists(os.path.join(cache_path, "model_index.json")):
        print(f"Loading from Network Volume: {cache_path}")
        model_path = cache_path
    else:
        print("Downloading from HuggingFace...")
        model_path = "black-forest-labs/FLUX.1-schnell"

    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    )

    # Move to GPU
    pipe.to("cuda")

    # Memory optimizations
    pipe.enable_attention_slicing(slice_size="auto")

    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.1f}s")

    # Save to Network Volume for faster future loads
    if model_path == "black-forest-labs/FLUX.1-schnell" and os.path.exists("/runpod-volume"):
        try:
            print(f"Caching model to: {cache_path}")
            os.makedirs(cache_path, exist_ok=True)
            pipe.save_pretrained(cache_path)
            print("Model cached!")
        except Exception as e:
            print(f"Cache failed: {e}")

    print("=" * 60)

    MODEL_LOADED = True
    return pipe


def handler(job):
    """
    RunPod handler for FLUX.1-Schnell image generation
    """
    job_input = job.get("input", {})

    prompt = job_input.get("prompt", "A beautiful landscape")
    width = job_input.get("width", 1024)
    height = job_input.get("height", 1024)
    seed = job_input.get("seed")
    num_inference_steps = job_input.get("num_inference_steps", 4)
    guidance_scale = job_input.get("guidance", job_input.get("guidance_scale", 0.0))

    # Validate dimensions
    width = (width // 8) * 8
    height = (height // 8) * 8
    width = max(256, min(width, 1536))
    height = max(256, min(height, 1536))

    print(f"\n{'='*60}")
    print(f"FLUX.1-Schnell Generation")
    print(f"Size: {width}x{height}, Steps: {num_inference_steps}")
    print(f"Prompt: {prompt[:80]}...")

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

        print(f"Output: {len(image_base64) // 1024}KB")
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
        return {"error": "Out of memory. Try smaller dimensions."}

    except Exception as e:
        print(f"Generation error: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FLUX.1-Schnell RunPod Worker")
    print("=" * 60)

    try:
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            print("WARNING: CUDA not available!")
    except Exception as e:
        print(f"GPU check error: {e}")

    hf_token = os.environ.get("HF_TOKEN")
    print(f"HF_TOKEN: {'Set (' + hf_token[:10] + '...)' if hf_token else 'NOT SET - WILL FAIL!'}")
    print("=" * 60 + "\n")

    # Start worker WITHOUT pre-loading (lazy load on first request)
    # This allows the container to start even if model download fails
    print("Starting worker (model will load on first request)...")
    runpod.serverless.start({"handler": handler})
