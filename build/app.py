# Copyright © 2025-2026 Quadux IT GmbH
#    ____                  __              __________   ______          __    __  __
#   / __ \__  ______ _____/ /_  ___  __   /  _/_  __/  / ____/___ ___  / /_  / / / /
#  / / / / / / / __ `/ __  / / / / |/_/   / /  / /    / / __/ __ `__ \/ __ \/ /_/ /
# / /_/ / /_/ / /_/ / /_/ / /_/ />  <   _/ /  / /    / /_/ / / / / / / /_/ / __  /
# \___\_\__,_/\__,_/\__,_/\__,_/_/|_|  /___/ /_/     \____/_/ /_/ /_/_.___/_/ /_/
# License: Quadux files Apache 2.0 (see LICENSE), Jina model: Qwen Research License
# Author: Walter Hoffmann

"""
Jina Embeddings v4 API - FastAPI server for text and image embeddings.
"""
import os
import sys
from io import BytesIO
from typing import List

# Suppress transformers warnings (Mistral regex etc.)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HUGGINGFACE_HUB_ALLOW_CODE_EVERYWHERE", "1")

# Disable tqdm progress bars - they break in Docker logs due to timestamp injection
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

# Configuration via environment variables
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
FORCE_CPU = os.getenv("FORCE_CPU", "").lower() in ("1", "true", "yes")
OFFLINE_MODE = os.getenv("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes") or \
               os.getenv("TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes")

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

MODEL_NAME = os.getenv("MODEL_NAME", "jinaai/jina-embeddings-v4")
VERSION = "1.2.0"

# Print copyright banner on startup
print(f"""
================================================================================
 Copyright (c) 2025-2026 Quadux IT GmbH
    ____                  __              __________   ______          __    __  __
   / __ \\__  ______ _____/ /_  ___  __   /  _/_  __/  / ____/___ ___  / /_  / / / /
  / / / / / / / __ `/ __  / / / / |/_/   / /  / /    / / __/ __ `__ \\/ __ \\/ /_/ /
 / /_/ / /_/ / /_/ / /_/ / /_/ />  <   _/ /  / /    / /_/ / / / / / / /_/ / __  /
 \\___\\_\\__,_/\\__,_/\\__,_/\\__,_/_/|_|  /___/ /_/     \\____/_/ /_/ /_/_.___/_/ /_/

 Jina Embeddings v4 API - Version {VERSION}
 License: Quadux files Apache 2.0, Jina model: Qwen Research License
 Author: Walter Hoffmann
================================================================================

 API Endpoints:
 ------------------------------------------------------------------------------
 GET  /health        - Health check (device, model status)
 
 Single-Vector (2048 dims, Matryoshka truncation supported):
 POST /embed/text    - Text embeddings  (params: texts[], task, dimensions?)
 POST /embed/image   - Image embeddings (params: file, task, dimensions?)
 
 Multi-Vector (128 dims per token/patch, for MaxSim/Late Interaction):
 POST /embed/textMV  - Text multi-vector  (params: texts[], task)
 POST /embed/imageMV - Image multi-vector (params: file, task)
 ------------------------------------------------------------------------------
""")

# Determine device - respect FORCE_CPU environment variable
if FORCE_CPU:
    DEVICE = torch.device("cpu")
    print("CPU mode forced via FORCE_CPU environment variable")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# float32 ist der Standard für numerische Stabilität (keine NaN bei langen Texten)
TORCH_DTYPE = torch.float32
print(f"Using device: {DEVICE} with dtype: float32")

app = FastAPI(title="Jina Embeddings v4 API")

print(f"Loading model {MODEL_NAME} on {DEVICE}...")

# ============================================================================
# STEP 1: Install import hook to patch dynamic modules as they load
# ============================================================================

import importlib
import importlib.abc
import importlib.machinery


class DynamicModulePatcher(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Import hook that patches Jina/Qwen dynamic modules as they're loaded."""
    
    def __init__(self):
        self._patched = set()
    
    def find_module(self, fullname, path=None):
        # Only intercept qwen2_5_vl modules from jina
        if "qwen2_5_vl" in fullname and "jina" in fullname.lower():
            return self
        return None
    
    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        
        # Let the default loader handle it first
        # Remove ourselves temporarily to avoid recursion
        sys.meta_path = [p for p in sys.meta_path if p is not self]
        try:
            module = importlib.import_module(fullname)
        finally:
            sys.meta_path.insert(0, self)
        
        # Now patch the module
        self._patch_module(module, fullname)
        return module
    
    def _patch_module(self, module, fullname):
        if fullname in self._patched:
            return
        self._patched.add(fullname)
        
        # Patch ROPE_INIT_FUNCTIONS
        if hasattr(module, "ROPE_INIT_FUNCTIONS"):
            rope = module.ROPE_INIT_FUNCTIONS
            if "default" not in rope:
                if "base" in rope:
                    rope["default"] = rope["base"]
                elif len(rope) > 0:
                    first_key = next(iter(rope.keys()))
                    rope["default"] = rope[first_key]


# Install the import hook
_patcher = DynamicModulePatcher()
sys.meta_path.insert(0, _patcher)

# ============================================================================
# STEP 2: Patch transformers before importing
# ============================================================================

from transformers import processing_utils, modeling_utils

# Global tokenizer reference
_tokenizer = None

# Patch ProcessorMixin.from_args_and_dict to inject tokenizer
_orig_from_args_and_dict = processing_utils.ProcessorMixin.from_args_and_dict.__func__

@classmethod
def _patched_from_args_and_dict(cls, args, processor_dict, **kwargs):
    """Inject tokenizer for Jina/Qwen processors."""
    global _tokenizer
    if _tokenizer is not None:
        # Force tokenizer into kwargs
        kwargs["tokenizer"] = _tokenizer
        
        # Also inject into args - the processor expects (image_processor, tokenizer, ...)
        args = list(args)
        if len(args) == 0:
            args = [None, _tokenizer]
        elif len(args) == 1:
            args.append(_tokenizer)
        elif len(args) >= 2:
            if args[1] is None:
                args[1] = _tokenizer
        args = tuple(args)
    return _orig_from_args_and_dict(cls, args, processor_dict, **kwargs)

processing_utils.ProcessorMixin.from_args_and_dict = _patched_from_args_and_dict

# Patch tied_weights helper for dynamic modules
_orig_get_tied = modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys

def _safe_get_tied(self, *a, **kw):
    try:
        return _orig_get_tied(self, *a, **kw)
    except AttributeError:
        return []

modeling_utils.PreTrainedModel.get_expanded_tied_weights_keys = _safe_get_tied

# Patch mark_tied_weights_as_initialized for list vs dict issue
_orig_mark_tied = modeling_utils.PreTrainedModel.mark_tied_weights_as_initialized

def _safe_mark_tied(self, *a, **kw):
    # Check if all_tied_weights_keys is a list instead of dict
    tied_keys = getattr(self, "all_tied_weights_keys", None)
    if isinstance(tied_keys, list):
        # Convert to dict format expected by the method
        self.all_tied_weights_keys = {k: True for k in tied_keys}
    try:
        return _orig_mark_tied(self, *a, **kw)
    except (AttributeError, TypeError):
        pass  # Silently ignore if it still fails

modeling_utils.PreTrainedModel.mark_tied_weights_as_initialized = _safe_mark_tied

# ============================================================================
# STEP 3: Import and patch transformers.models ROPE
# ============================================================================

from transformers import AutoTokenizer, AutoConfig, AutoModel

try:
    from transformers.models.qwen2_5.modeling_qwen2_5 import ROPE_INIT_FUNCTIONS
    if "default" not in ROPE_INIT_FUNCTIONS and "base" in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = ROPE_INIT_FUNCTIONS["base"]
except Exception:
    pass

# ============================================================================
# STEP 4: Load tokenizer FIRST
# ============================================================================

print("Loading tokenizer...")
if OFFLINE_MODE:
    print("  (offline mode - using cached files only)")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True,
    local_files_only=OFFLINE_MODE
)
    
_tokenizer = tokenizer  # Set global for processor patch

# ============================================================================
# STEP 5: Load and fix config
# ============================================================================

print("Loading config...")
config = AutoConfig.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True,
    local_files_only=OFFLINE_MODE
)


def _fix_rope_config(cfg):
    """Fix rope_type and rope_scaling in config."""
    if hasattr(cfg, "rope_type") and cfg.rope_type == "default":
        cfg.rope_type = "base"
    if hasattr(cfg, "rope_scaling"):
        if cfg.rope_scaling is None:
            cfg.rope_scaling = {"type": "linear", "factor": 1.0}
        elif isinstance(cfg.rope_scaling, dict):
            cfg.rope_scaling.setdefault("type", "linear")
            cfg.rope_scaling.setdefault("factor", 1.0)
    if hasattr(cfg, "text_config"):
        _fix_rope_config(cfg.text_config)


_fix_rope_config(config)

# ============================================================================
# STEP 6: Patch loaded dynamic modules (they get loaded during config)
# ============================================================================


def _patch_all_loaded_modules():
    """Patch all already-loaded qwen2_5_vl modules."""
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        if "qwen2_5_vl" in name.lower() and "jina" in name.lower():
            if hasattr(module, "ROPE_INIT_FUNCTIONS"):
                rope = module.ROPE_INIT_FUNCTIONS
                if "default" not in rope:
                    if "base" in rope:
                        rope["default"] = rope["base"]


_patch_all_loaded_modules()

# ============================================================================
# STEP 7: Load model
# ============================================================================

print("Loading model...")
model = AutoModel.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True, 
    config=config,
    torch_dtype=TORCH_DTYPE,
    local_files_only=OFFLINE_MODE
)
model.to(DEVICE)
model.eval()

print(f"Model loaded successfully on {DEVICE}")

# ============================================================================
# PEFT-compatible encode methods that route through the PEFT wrapper
# ============================================================================

def peft_encode_image(peft_model, images, task=None, truncate_dim=None, return_numpy=False):
    """
    Encode images through PEFT model, ensuring LoRA adapters are applied.
    """
    import numpy as np
    from PIL import Image
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    # Get inner model for config/processor access
    inner = peft_model.base_model.model if hasattr(peft_model, "base_model") else peft_model
    
    # Validate task
    if task is None:
        task = "text-matching"
    valid_tasks = ["retrieval", "text-matching", "code"]
    if task not in valid_tasks:
        raise ValueError(f"Invalid task: {task}. Must be one of {valid_tasks}")
    
    # Validate truncate_dim
    if truncate_dim is not None:
        valid_dims = [128, 256, 512, 1024, 2048]
        if truncate_dim not in valid_dims:
            raise ValueError(f"Invalid truncate_dim: {truncate_dim}. Must be one of {valid_dims}")
    
    # Get processor
    proc = inner.processor if hasattr(inner, 'processor') else None
    if proc is None:
        raise ValueError("No processor found on inner model")
    
    # Convert to list
    if isinstance(images, (str, Image.Image)):
        images = [images]
    
    # Load images if URLs/paths
    loaded_images = []
    for img in images:
        if isinstance(img, str):
            if img.startswith("http"):
                import requests
                from io import BytesIO
                response = requests.get(img)
                img = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                img = Image.open(img).convert("RGB")
        loaded_images.append(img)
    
    # Process batches - KEY: call peft_model() which routes through PEFT
    dataloader = DataLoader(
        dataset=loaded_images,
        batch_size=8,
        shuffle=False,
        collate_fn=proc.process_images,
    )
    
    results = []
    peft_model.eval()
    
    for batch in tqdm(dataloader, desc="Encoding images...", disable=True):
        with torch.no_grad():
            batch = {k: v.to(peft_model.device) for k, v in batch.items()}
            
            # Forward through PEFT wrapper with task_label
            output = peft_model(**batch, task_label=task)
            
            embeddings = output.single_vec_emb
            if truncate_dim is not None:
                embeddings = embeddings[:, :truncate_dim]
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            
            embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
            results.append(embeddings.cpu())
    
    if return_numpy:
        return np.concatenate([r.numpy() for r in results], axis=0)
    return [item for sublist in results for item in torch.unbind(sublist)]


def peft_encode_text(peft_model, texts, task=None, truncate_dim=None, prompt_name=None, return_numpy=False):
    """
    Encode texts through PEFT model, ensuring LoRA adapters are applied.
    Uses proper prefixes (Query:/Passage:) for retrieval tasks.
    """
    import numpy as np
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    # Get inner model for config/processor access
    inner = peft_model.base_model.model if hasattr(peft_model, "base_model") else peft_model
    
    # Validate task
    if task is None:
        task = "retrieval"
    valid_tasks = ["retrieval", "text-matching", "code"]
    if task not in valid_tasks:
        raise ValueError(f"Invalid task: {task}. Must be one of {valid_tasks}")
    
    # Validate truncate_dim
    if truncate_dim is not None:
        valid_dims = [128, 256, 512, 1024, 2048]
        if truncate_dim not in valid_dims:
            raise ValueError(f"Invalid truncate_dim: {truncate_dim}. Must be one of {valid_dims}")
    
    # Get processor
    proc = inner.processor if hasattr(inner, 'processor') else None
    if proc is None:
        raise ValueError("No processor found on inner model")
    
    # Convert to list
    if isinstance(texts, str):
        texts = [texts]
    
    # Determine prefix based on task and prompt_name
    # NOTE: process_texts adds ": " after prefix, so use just "Query" or "Passage"
    # For text-matching task, always use "Query" prefix (as per Jina docs)
    if task == "text-matching":
        prefix = "Query"  # text-matching always uses Query prefix
    elif task == "retrieval":
        if prompt_name == "passage":
            prefix = "Passage"
        else:
            prefix = "Query"
    else:  # code task
        prefix = None  # No prefix for code
    
    # Create processor function with prefix
    from functools import partial
    processor_fn = partial(proc.process_texts, prefix=prefix)
    
    # Process batches - KEY: call peft_model() which routes through PEFT
    dataloader = DataLoader(
        dataset=texts,  # Raw texts, prefix is applied by processor
        batch_size=32,
        shuffle=False,
        collate_fn=processor_fn,
    )
    
    results = []
    peft_model.eval()
    
    for batch in tqdm(dataloader, desc="Encoding texts...", disable=True):
        with torch.no_grad():
            batch = {k: v.to(peft_model.device) for k, v in batch.items()}
            
            # Forward through PEFT wrapper with task_label
            output = peft_model(**batch, task_label=task)
            
            embeddings = output.single_vec_emb
            if truncate_dim is not None:
                embeddings = embeddings[:, :truncate_dim]
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            
            embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
            results.append(embeddings.cpu())
    
    if return_numpy:
        return np.concatenate([r.numpy() for r in results], axis=0)
    return [item for sublist in results for item in torch.unbind(sublist)]


# Load processor for image embeddings
print("Loading processor...")
from transformers import AutoProcessor, AutoImageProcessor
processor = None
try:
    # Lade den Hauptprozessor
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True,
        local_files_only=OFFLINE_MODE
    )
    # Der JinaEmbeddingsV4Processor (erbt von Qwen2_5_VLProcessor) braucht tokenizer + image_processor
    # Injiziere fehlende Komponenten
    if not hasattr(processor, 'tokenizer') or processor.tokenizer is None:
        processor.tokenizer = tokenizer
        print("  (tokenizer injected into processor)")
    if not hasattr(processor, 'image_processor') or processor.image_processor is None:
        try:
            image_proc = AutoImageProcessor.from_pretrained(
                MODEL_NAME, 
                trust_remote_code=True,
                local_files_only=OFFLINE_MODE
            )
            processor.image_processor = image_proc
            print("  (image_processor injected into processor)")
        except Exception as ip_err:
            print(f"  Warning: Could not load image_processor: {ip_err}")
    
    # Injiziere den vollständigen Processor ins innere Model
    # damit encode_image funktioniert
    inner_model = model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        inner_model = model.base_model.model
    if hasattr(inner_model, 'processor'):
        if not hasattr(inner_model.processor, 'tokenizer') or inner_model.processor.tokenizer is None:
            inner_model.processor.tokenizer = tokenizer
            print("  (tokenizer injected into inner_model.processor)")
        if not hasattr(inner_model.processor, 'image_processor') or inner_model.processor.image_processor is None:
            if processor is not None and hasattr(processor, 'image_processor'):
                inner_model.processor.image_processor = processor.image_processor
                print("  (image_processor injected into inner_model.processor)")
    
    print("Processor loaded successfully")
except Exception as e:
    print(f"Warning: Could not load processor: {e}")
    processor = None

# ============================================================================
# API Endpoints
# ============================================================================


# Debug endpoints removed for production - API is ready!


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class TextRequest(BaseModel):
    texts: List[str]
    task: str = "retrieval.passage"
    dimensions: int = None  # Matryoshka: truncate to this dimension (default: full 2048)


class TextResponse(BaseModel):
    embeddings: List[List[float]]


@app.get("/health")
async def health():
    return {"status": "ok", "version": VERSION, "device": str(DEVICE), "model": MODEL_NAME}


@app.post("/embed/text", response_model=TextResponse)
async def embed_text(req: TextRequest):
    """
    Generate single-vector embeddings for a batch of texts via the Jina encode_text path.

    The request body mirrors `TextRequest`: a list of texts, a dot-suffixed task (e.g.
    "retrieval.passage", "text-matching.query", "code"), and an optional Matryoshka
    `dimensions` value to truncate the 2048-dim vectors. The handler validates the base
    task and optionally derives a prompt_name, then routes through the PEFT-aware
    `peft_encode_text` helper to keep LoRA adapters active.

    Returns a `TextResponse` carrying the cleaned embeddings.
    """
    try:
        # Parse task name: "retrieval.query" -> task="retrieval", prompt_name="query"
        # Valid tasks: "retrieval", "text-matching", "code"
        task_parts = req.task.split(".")
        base_task = task_parts[0]
        
        # Validate base task
        valid_tasks = ["retrieval", "text-matching", "code"]
        if base_task not in valid_tasks:
            raise HTTPException(status_code=400, detail=f"Invalid task: {base_task}. Must be one of {valid_tasks}")
        
        # Determine prompt_name based on task suffix
        prompt_name = "query"  # Default
        if len(task_parts) > 1:
            if task_parts[1] in ["query", "passage"]:
                prompt_name = task_parts[1]
        
        # Use PEFT-compatible encode function that routes through LoRA adapters
        embeddings = peft_encode_text(
            peft_model=model,
            texts=req.texts,
            task=base_task,
            truncate_dim=req.dimensions,
            prompt_name=prompt_name,
            return_numpy=True
        )
        
        import numpy as np
        if isinstance(embeddings, np.ndarray):
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
            return TextResponse(embeddings=embeddings.tolist())
        elif isinstance(embeddings, list):
            return TextResponse(embeddings=[e.tolist() if hasattr(e, 'tolist') else e for e in embeddings])
        else:
            return TextResponse(embeddings=[embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings])
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class ImageRequest(BaseModel):
    task: str = "text-matching"


# =============================================================================
# IMAGE EMBEDDING ENDPOINTS
# =============================================================================

@app.post("/embed/image")
async def embed_image(
    file: UploadFile = File(...), 
    task: str = Form("retrieval"), 
    dimensions: int = Form(None)
):
    """
    Generate Single-Vector embeddings for an image.
    
    Single-Vector mode returns one embedding per image with 2048 dimensions (or truncated via Matryoshka).
    Use this for:
    - MongoDB Atlas Vector Search
    - Any vector database (Qdrant, Milvus, Pinecone, etc.)
    - Simple cosine similarity search
    
    Args:
        file: Image file (PNG, JPEG, etc.)
        task: "retrieval" (recommended), "text-matching", or "code"
        dimensions: Optional Matryoshka truncation (128, 256, 512, 1024, 2048)
    
    Returns:
        {"embeddings": [[...2048 floats...]]}
    """
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        base_task = task.split(".")[0]
        valid_tasks = ["retrieval", "text-matching", "code"]
        if base_task not in valid_tasks:
            raise HTTPException(status_code=400, detail=f"Invalid task: {base_task}. Must be one of {valid_tasks}")
        
        embeddings = peft_encode_image(
            peft_model=model,
            images=[image],
            task=base_task,
            truncate_dim=dimensions,
            return_numpy=True
        )
        
        import numpy as np
        if isinstance(embeddings, np.ndarray):
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
            return {"embeddings": embeddings.tolist() if embeddings.ndim == 2 else [embeddings.tolist()]}
        elif isinstance(embeddings, list):
            return {"embeddings": [e.tolist() if hasattr(e, 'tolist') else e for e in embeddings]}
        return {"embeddings": [embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings]}
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class MultiVectorResponse(BaseModel):
    """Response for multi-vector embeddings."""
    embeddings: List[List[List[float]]]  # List of (num_patches x 128) embeddings
    shapes: List[List[int]]  # Shape info for each embedding


@app.post("/embed/imageMV")
async def embed_image_multi_vector(
    file: UploadFile = File(...), 
    task: str = Form("retrieval")
):
    """
    Generate Multi-Vector embeddings for an image (Late Interaction / ColPali style).
    
    Multi-Vector mode returns multiple 128-dim embeddings per image - one per image patch.
    Use this for:
    - Visual Document Retrieval (PDFs, charts, tables, diagrams)
    - MaxSim similarity (late interaction)
    - Reranking with fine-grained matching
    
    Args:
        file: Image file (PNG, JPEG, etc.)
        task: "retrieval" (recommended), "text-matching", or "code"
    
    Returns:
        {
            "embeddings": [[[...128 floats...], [...], ...]],  // N patches x 128 dims
            "shapes": [[N, 128]]  // Number of patches varies by image size
        }
    """
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        base_task = task.split(".")[0]
        valid_tasks = ["retrieval", "text-matching", "code"]
        if base_task not in valid_tasks:
            raise HTTPException(status_code=400, detail=f"Invalid task: {base_task}. Must be one of {valid_tasks}")
        
        # Get inner model for multi-vector encoding
        inner = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model
        
        embeddings = inner.encode_image(
            [image],
            task=base_task,
            return_multivector=True,
            return_numpy=True
        )
        
        import numpy as np
        result_embeddings = []
        result_shapes = []
        
        for emb in embeddings:
            if hasattr(emb, 'cpu'):
                emb = emb.cpu().numpy()
            emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
            result_embeddings.append(emb.tolist())
            result_shapes.append(list(emb.shape))
        
        return {"embeddings": result_embeddings, "shapes": result_shapes}
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/textMV")
async def embed_text_multi_vector(req: TextRequest):
    """
    Generate Multi-Vector embeddings for text (Late Interaction / ColBERT style).
    
    Multi-Vector mode returns multiple 128-dim embeddings per text - one per token.
    Use this for:
    - MaxSim similarity with image multi-vectors
    - Fine-grained text-document matching
    
    Args:
        texts: List of text strings
        task: "retrieval" (recommended), "text-matching", or "code"
    
    Returns:
        {
            "embeddings": [[[...128 floats...], [...], ...]],  // N tokens x 128 dims per text
            "shapes": [[N, 128], ...]  // Number of tokens varies by text length
        }
    """
    try:
        task_parts = req.task.split(".")
        base_task = task_parts[0]
        prompt_name = task_parts[1] if len(task_parts) > 1 else "query"
        
        valid_tasks = ["retrieval", "text-matching", "code"]
        if base_task not in valid_tasks:
            raise HTTPException(status_code=400, detail=f"Invalid task: {base_task}. Must be one of {valid_tasks}")
        
        # Get inner model for multi-vector encoding
        inner = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model
        
        embeddings = inner.encode_text(
            req.texts,
            task=base_task,
            prompt_name=prompt_name,
            return_multivector=True,
            return_numpy=True
        )
        
        import numpy as np
        result_embeddings = []
        result_shapes = []
        
        for emb in embeddings:
            if hasattr(emb, 'cpu'):
                emb = emb.cpu().numpy()
            emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
            result_embeddings.append(emb.tolist())
            result_shapes.append(list(emb.shape))
        
        return {"embeddings": result_embeddings, "shapes": result_shapes}
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
